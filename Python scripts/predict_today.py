"""
Predict Today — NCAAB Prediction System
=========================================
Fetches today's CBS games, loads team features from existing CSVs, loads trained
models from the model cache, and outputs win-probability predictions for every
scheduled game.

Steps:
  1. Scrape CBS Sports for today's games (ALL games — no Final-only filter)
  2. Load team feature lookup from existing CSVs (sentiment + efficiency + kenpom)
  3. Load models from Python scripts/model_cache/ (produced by model_training.py)
  4. Build matchup vectors (is_home=1.0 — prediction perspective)
  5. Run all 6 models and compute ensemble probabilities
  6. Display regression table, probability table, confidence summary, and picks

Model ensemble (from model_cache/metadata.joblib):
    Regression : 30% Ridge + 30% GB + 40% NN Reg  → reg_implied_prob (logistic transform)
    Classif.   : 35% Logistic + 20% Bayes + 45% NN Cls → cls_ensemble_prob
    Final      : 55% reg_implied + 45% cls_ensemble → final_win_prob

Confidence tiers (§5.3 Requirements):
    HIGH : prob_std < 0.08  AND  |final_prob − 0.5| > 0.15
    MED  : prob_std < 0.15  AND  |final_prob − 0.5| > 0.08
    LOW  : all other cases
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import date, datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import feedparser as _feedparser
    _FEEDPARSER_AVAILABLE = True
except ImportError:
    _feedparser = None  # type: ignore
    _FEEDPARSER_AVAILABLE = False

try:
    import odds_features as _odds_features
    _ODDS_AVAILABLE = True
except ImportError:
    _odds_features = None  # type: ignore
    _ODDS_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

# Input paths
SENTIMENT_CSV_PATH  = os.path.join(_SCRIPT_DIR, "sentiment_features.csv")
EFFICIENCY_CSV_PATH = os.path.join(_SCRIPT_DIR, "efficiency_metrics.csv")
EFF_MAPPING_PATH    = os.path.join(_SCRIPT_DIR, "team_name_mapping.csv")
KENPOM_CSV_PATH     = os.path.join(_SCRIPT_DIR, "kenpom_ratings.csv")
CACHE_DIR           = os.path.join(_SCRIPT_DIR, "model_cache")

TODAY = date.today()

CBS_BASE_URL     = "https://www.cbssports.com/college-basketball/scoreboard/FBS/{date_str}/"
REQUEST_TIMEOUT  = 20
RETRY_ATTEMPTS   = 3
RETRY_BACKOFF    = [2, 5, 10]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Confidence thresholds (§5.3 Requirements)
CONF_HIGH_STD    = 0.08
CONF_HIGH_MARGIN = 0.15
CONF_MED_STD     = 0.15
CONF_MED_MARGIN  = 0.08

# Efficiency / KenPom feature columns
EFFICIENCY_FEATURE_COLS = ["efg_pct", "tov_pct", "orb_pct", "ftr"]
KENPOM_FEATURE_COLS     = ["kenpom_rank", "adj_em", "adj_o", "adj_d"]

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# =============================================================================
# CELL 2 (ADAPTED FROM CELL 5 OF PREDICTOR NOTEBOOK) — CBS TODAY'S GAMES
# ALL STATUSES INCLUDED (no Final-only filter — we predict upcoming games)
# =============================================================================

def _fetch_page(url: str) -> Optional[str]:
    """Fetch a URL with retry/backoff. Returns HTML text or None on failure."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            log.warning("HTTP %d on attempt %d — %s", resp.status_code, attempt + 1, url)
        except requests.exceptions.Timeout:
            log.warning("Timeout on attempt %d — %s", attempt + 1, url)
        except requests.exceptions.RequestException as exc:
            log.warning("Request error on attempt %d: %s", attempt + 1, exc)
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(RETRY_BACKOFF[attempt])
    return None


def get_todays_games(game_date: date) -> list[dict]:
    """
    Scrape CBS Sports scoreboard for game_date.

    KEY DIFFERENCE from training scraper: the Final-only filter is NOT applied.
    All game cards (scheduled, live, in-progress, final) are returned so that
    every matchup of the day has a prediction.

    Scores may be None for upcoming games — predictions are still produced.
    """
    date_str = game_date.strftime("%Y%m%d")
    url      = CBS_BASE_URL.format(date_str=date_str)

    html = _fetch_page(url)
    if html is None:
        log.error("Failed to fetch CBS page for %s", date_str)
        return []

    soup         = BeautifulSoup(html, "lxml")
    cards        = soup.find_all("div", class_="single-score-card")
    seen_abbrevs = set()
    games        = []

    log.info("CBS cards found     : %d  (%s)", len(cards), game_date.isoformat())

    for card in cards:
        try:
            abbrev = card.get("data-abbrev", "")
            if not abbrev:
                continue

            # Deduplication (Business Rule 2 — one row per game)
            if abbrev in seen_abbrevs:
                log.warning("Duplicate data-abbrev '%s' — skipping", abbrev)
                continue
            seen_abbrevs.add(abbrev)

            # Status (any value — not filtered)
            status_tag = card.find("div", class_="game-status")
            status_raw = status_tag.get_text(strip=True) if status_tag else "Scheduled"

            # Team rows (row 0 = away, row 1 = home — per data-abbrev convention)
            tr_rows = [tr for tr in card.find_all("tr") if tr.get("class") is not None]
            if len(tr_rows) < 2:
                continue

            away_row, home_row = tr_rows[0], tr_rows[1]

            def _name(row) -> Optional[str]:
                tag = row.find("a", class_="team-name-link")
                return tag.get_text(strip=True) if tag else None

            def _score(row) -> Optional[int]:
                tag = row.find("td", class_="total")
                if tag is None:
                    return None
                try:
                    return int(tag.get_text(strip=True))
                except (ValueError, TypeError):
                    return None

            away_name  = _name(away_row)
            home_name  = _name(home_row)
            away_score = _score(away_row)
            home_score = _score(home_row)

            if not away_name or not home_name:
                continue

            games.append({
                "date":      game_date.isoformat(),
                "away_name": away_name,
                "away_score": away_score,
                "home_name":  home_name,
                "home_score": home_score,
                "status":     status_raw,
            })

        except Exception as exc:
            log.warning("Card parse error [%s]: %s", card.get("data-abbrev", "?"), exc)
            continue

    log.info("Games found today   : %d", len(games))
    return games

# =============================================================================
# CELL 7 (ADAPTED) — BUILD TEAM FEATURE LOOKUP FROM EXISTING CSVs
# =============================================================================

def _load_sentiment(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sentiment CSV not found: {path}")
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != "team_name"]
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def _load_efficiency(eff_path: str, mapping_path: str) -> pd.DataFrame:
    if not os.path.isfile(eff_path) or not os.path.isfile(mapping_path):
        log.warning("Efficiency CSVs not found — efficiency features will be 0.0")
        return pd.DataFrame(columns=["cbs_name"] + EFFICIENCY_FEATURE_COLS)
    eff_df     = pd.read_csv(eff_path)
    mapping_df = pd.read_csv(mapping_path)
    merged = eff_df.merge(
        mapping_df[["source_name", "cbs_name"]],
        left_on="team_id", right_on="source_name", how="left",
    )
    result = merged[["cbs_name"] + EFFICIENCY_FEATURE_COLS].copy()
    for col in EFFICIENCY_FEATURE_COLS:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    result = result.dropna(subset=["cbs_name"])
    dup_count = result["cbs_name"].duplicated().sum()
    if dup_count > 0:
        result = result.groupby("cbs_name", as_index=False)[EFFICIENCY_FEATURE_COLS].mean()
    return result.reset_index(drop=True)


def _load_kenpom(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        log.warning("KenPom CSV not found — kenpom features will be 0.0")
        return pd.DataFrame(columns=["cbs_name"] + KENPOM_FEATURE_COLS)
    df     = pd.read_csv(path)
    result = df[["cbs_name"] + KENPOM_FEATURE_COLS].copy()
    for col in KENPOM_FEATURE_COLS:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    result = result[result["cbs_name"].str.strip() != ""].dropna(subset=["cbs_name"])
    dup_count = result["cbs_name"].duplicated().sum()
    if dup_count > 0:
        result = result.groupby("cbs_name", as_index=False)[KENPOM_FEATURE_COLS].mean()
    return result.reset_index(drop=True)


def build_team_feature_lookup(numeric_cols: list[str]) -> dict:
    """
    Build a per-team feature dict by joining sentiment + efficiency + kenpom CSVs.

    numeric_cols is the ordered feature list from metadata.joblib — this ensures
    the lookup dict keys match the column order the models were trained on.

    Returns dict[team_name → dict[feature_name → float]]
    Teams missing from a source receive 0.0 for that source's features.
    """
    sentiment_df  = _load_sentiment(SENTIMENT_CSV_PATH)
    efficiency_df = _load_efficiency(EFFICIENCY_CSV_PATH, EFF_MAPPING_PATH)
    kenpom_df     = _load_kenpom(KENPOM_CSV_PATH)

    # Join efficiency and kenpom onto sentiment (base — all teams present)
    df = sentiment_df.copy()
    df = df.merge(
        efficiency_df.rename(columns={"cbs_name": "team_name"}),
        on="team_name", how="left",
    )
    df = df.merge(
        kenpom_df.rename(columns={"cbs_name": "team_name"}),
        on="team_name", how="left",
    )
    feat_cols = [c for c in df.columns if c != "team_name"]
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Build lookup, keeping only columns that match numeric_cols from training
    lookup = {}
    for _, row in df.iterrows():
        team = str(row["team_name"])
        lookup[team] = {
            col: float(row[col]) if col in df.columns else 0.0
            for col in numeric_cols
        }

    log.info("Team feature lookup : %d teams  (%d features each)",
             len(lookup), len(numeric_cols))
    return lookup

# =============================================================================
# MATCHUP VECTOR HELPERS (mirrors model_training.py exactly)
# =============================================================================

def _get_feature_vector(team_name: str, lookup: dict, cols: list[str]) -> np.ndarray:
    if team_name in lookup:
        return np.array(
            [float(lookup[team_name].get(c, 0.0)) for c in cols],
            dtype=np.float32,
        )
    return np.zeros(len(cols), dtype=np.float32)


def _build_matchup_vector(
    home_name: str,
    away_name: str,
    lookup: dict,
    cols: list[str],
) -> np.ndarray:
    """Build prediction matchup vector with is_home=1.0 (prediction perspective)."""
    home_vec = _get_feature_vector(home_name, lookup, cols)
    away_vec = _get_feature_vector(away_name, lookup, cols)
    diff_vec = home_vec - away_vec
    return np.concatenate([home_vec, away_vec, diff_vec, [1.0]])  # is_home=1.0

# =============================================================================
# CELL 4 — LOAD MODELS FROM CACHE
# =============================================================================

def load_models(cache_dir: str) -> tuple[dict, dict]:
    """
    Load all models and metadata from cache_dir.
    Raises FileNotFoundError if cache is missing (run model_training.py first).
    """
    metadata_path = os.path.join(cache_dir, "metadata.joblib")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Model cache not found at: {cache_dir}\n"
            "Run model_training.py first to train and cache models."
        )

    metadata = joblib.load(metadata_path)

    models: dict = {
        "scaler":         joblib.load(os.path.join(cache_dir, "scaler.joblib")),
        "lr_model":       joblib.load(os.path.join(cache_dir, "ridge_model.joblib")),
        "sgd_model":      joblib.load(os.path.join(cache_dir, "gb_model.joblib")),
        "logistic_model": joblib.load(os.path.join(cache_dir, "logistic_model.joblib")),
        "bayes_model":    joblib.load(os.path.join(cache_dir, "bayes_model.joblib")),
        "nn_reg_model":   None,
        "nn_cls_model":   None,
    }

    if TF_AVAILABLE:
        nn_reg_path = os.path.join(cache_dir, "nn_regressor.keras")
        nn_cls_path = os.path.join(cache_dir, "nn_classifier.keras")
        if os.path.isfile(nn_reg_path):
            models["nn_reg_model"] = keras.models.load_model(nn_reg_path)
        if os.path.isfile(nn_cls_path):
            models["nn_cls_model"] = keras.models.load_model(nn_cls_path)

    log.info(
        "Models loaded       : schema=%s  as_of=%s  TF=%s",
        metadata.get("schema_version", "?"),
        metadata.get("as_of_date", "?"),
        TF_AVAILABLE,
    )
    return models, metadata

# =============================================================================
# CELL 11 (ADAPTED) — PREDICTION FUNCTIONS
# =============================================================================

def _prob_to_moneyline(p: float) -> str:
    """Convert win probability to American moneyline string."""
    p = float(np.clip(p, 0.01, 0.99))
    if p >= 0.5:
        return f"-{round((p / (1.0 - p)) * 100)}"
    return f"+{round(((1.0 - p) / p) * 100)}"


def _assign_confidence(prob_std: float, margin: float) -> str:
    """Assign HIGH / MED / LOW confidence tier."""
    if prob_std < CONF_HIGH_STD and margin > CONF_HIGH_MARGIN:
        return "HIGH"
    if prob_std < CONF_MED_STD and margin > CONF_MED_MARGIN:
        return "MED"
    return "LOW"


def predict_game(
    game: dict,
    models: dict,
    lookup: dict,
    numeric_cols: list[str],
    ensemble_weights: dict,
) -> dict:
    """
    Run all 6 models for a single game and return a full prediction dict.

    game keys required: home_name, away_name, status, date
    """
    home_name = game["home_name"]
    away_name = game["away_name"]

    # Warn if either team has no features (zero vector will be used)
    if home_name not in lookup:
        log.warning("No features for home team '%s' — using zero vector", home_name)
    if away_name not in lookup:
        log.warning("No features for away team '%s' — using zero vector", away_name)

    vec      = _build_matchup_vector(home_name, away_name, lookup, numeric_cols)
    X_game   = models["scaler"].transform(vec.reshape(1, -1))

    # ── Regression models → score differential ────────────────────────────────
    lr_diff  = float(models["lr_model"].predict(X_game)[0])
    sgd_diff = float(models["sgd_model"].predict(X_game)[0])

    ew = ensemble_weights
    if models["nn_reg_model"] is not None:
        nn_diff  = float(models["nn_reg_model"].predict(X_game, verbose=0)[0][0])
        reg_diff = ew["reg_w_ridge"] * lr_diff + ew["reg_w_gb"] * sgd_diff + ew["reg_w_nn"] * nn_diff
    else:
        nn_diff  = float("nan")
        _r = ew["reg_w_ridge"] / (ew["reg_w_ridge"] + ew["reg_w_gb"])
        _g = ew["reg_w_gb"]    / (ew["reg_w_ridge"] + ew["reg_w_gb"])
        reg_diff = _r * lr_diff + _g * sgd_diff

    # Regression-implied win probability (logistic transform, scale=10)
    reg_implied_prob = float(1.0 / (1.0 + np.exp(-reg_diff / 10.0)))

    # ── Classification models → win probability ────────────────────────────────
    log_prob   = float(models["logistic_model"].predict_proba(X_game)[0][1])
    bayes_prob = float(models["bayes_model"].predict_proba(X_game)[0][1])

    if models["nn_cls_model"] is not None:
        nn_cls_prob = float(models["nn_cls_model"].predict(X_game, verbose=0)[0][0])
        cls_prob    = (
            ew["cls_w_logistic"] * log_prob
            + ew["cls_w_bayes"]  * bayes_prob
            + ew["cls_w_nn"]     * nn_cls_prob
        )
    else:
        nn_cls_prob = float("nan")
        _total = ew["cls_w_logistic"] + ew["cls_w_bayes"]
        cls_prob = (
            (ew["cls_w_logistic"] / _total) * log_prob
            + (ew["cls_w_bayes"]  / _total) * bayes_prob
        )

    # ── Final ensemble ─────────────────────────────────────────────────────────
    final_prob = float(np.clip(
        ew["final_w_reg"] * reg_implied_prob + ew["final_w_cls"] * cls_prob,
        0.01, 0.99,
    ))
    away_prob = 1.0 - final_prob

    # ── Confidence ────────────────────────────────────────────────────────────
    all_probs = [reg_implied_prob, log_prob, bayes_prob]
    if not np.isnan(nn_cls_prob):
        all_probs.append(nn_cls_prob)
    prob_std  = float(np.std(all_probs))
    margin    = abs(final_prob - 0.5)
    confidence = _assign_confidence(prob_std, margin)

    # ── Predicted winner and moneylines ───────────────────────────────────────
    predicted_winner = home_name if final_prob >= 0.5 else away_name
    home_ml = _prob_to_moneyline(final_prob)
    away_ml = _prob_to_moneyline(away_prob)

    return {
        # Game info
        "Date":               game["date"],
        "Home Team":          home_name,
        "Away Team":          away_name,
        "Status":             game.get("status", "?"),
        # Regression outputs
        "Ridge Diff":         round(lr_diff, 2),
        "GB Diff":            round(sgd_diff, 2),
        "NN Reg Diff":        round(nn_diff, 2) if not np.isnan(nn_diff) else "N/A",
        "Reg Ensemble Diff":  round(reg_diff, 2),
        # Classification outputs
        "Logistic Prob":      round(log_prob, 3),
        "Bayes Prob":         round(bayes_prob, 3),
        "NN Cls Prob":        round(nn_cls_prob, 3) if not np.isnan(nn_cls_prob) else "N/A",
        "Cls Ensemble Prob":  round(cls_prob, 3),
        # Final ensemble
        "Reg Implied Prob":   round(reg_implied_prob, 3),
        "Final Win Prob (Home)": round(final_prob, 3),
        "Final Win Prob (Away)": round(away_prob, 3),
        # Output
        "Predicted Winner":   predicted_winner,
        "Home ML":            home_ml,
        "Away ML":            away_ml,
        "Model Std Dev":      round(prob_std, 3),
        "Confidence":         confidence,
    }

# =============================================================================
# CELL 12 — DISPLAY RESULTS
# =============================================================================

def display_results(results: list[dict]) -> None:
    """Print four output tables to stdout."""
    if not results:
        print("\n  No games to display.\n")
        return

    print()
    print("=" * 75)
    print(f"  NCAAB PREDICTIONS — {TODAY.strftime('%A, %B %-d, %Y').upper()}")
    print(f"  {len(results)} game(s)  |  TF={TF_AVAILABLE}")
    print("=" * 75)

    # ── Table 1: Regression model score differentials ─────────────────────────
    print("\n  TABLE 1 — REGRESSION MODELS (Predicted Score Differential: Home − Away)\n")
    reg_rows = [
        [
            r["Home Team"],
            r["Away Team"],
            r["Ridge Diff"],
            r["GB Diff"],
            r["NN Reg Diff"],
            r["Reg Ensemble Diff"],
        ]
        for r in results
    ]
    print(tabulate(
        reg_rows,
        headers=["Home", "Away", "Ridge", "GB", "NN Reg", "Ensemble Diff"],
        tablefmt="rounded_outline",
        floatfmt=".2f",
        numalign="right",
    ))

    # ── Table 2: Classification model win probabilities ───────────────────────
    print("\n  TABLE 2 — CLASSIFICATION MODELS (Home Win Probability)\n")
    cls_rows = [
        [
            r["Home Team"],
            r["Away Team"],
            f"{r['Logistic Prob']:.1%}",
            f"{r['Bayes Prob']:.1%}",
            f"{r['NN Cls Prob']:.1%}" if r["NN Cls Prob"] != "N/A" else "N/A",
            f"{r['Cls Ensemble Prob']:.1%}",
        ]
        for r in results
    ]
    print(tabulate(
        cls_rows,
        headers=["Home", "Away", "Logistic", "Bayes", "NN Cls", "Cls Ensemble"],
        tablefmt="rounded_outline",
    ))

    # ── Table 3: Final ensemble + confidence summary ──────────────────────────
    print("\n  TABLE 3 — FINAL ENSEMBLE + CONFIDENCE\n")
    final_rows = [
        [
            r["Home Team"],
            r["Away Team"],
            f"{r['Reg Implied Prob']:.1%}",
            f"{r['Cls Ensemble Prob']:.1%}",
            f"{r['Final Win Prob (Home)']:.1%}",
            r["Predicted Winner"],
            r["Home ML"],
            r["Away ML"],
            f"{r['Model Std Dev']:.3f}",
            r["Confidence"],
        ]
        for r in results
    ]
    print(tabulate(
        final_rows,
        headers=[
            "Home", "Away", "Reg→Prob", "Cls Prob",
            "Final%", "Predicted Winner",
            "Home ML", "Away ML", "Std Dev", "Conf",
        ],
        tablefmt="rounded_outline",
    ))

    # ── Table 4: High + Med confidence picks ──────────────────────────────────
    picks = [r for r in results if r["Confidence"] in ("HIGH", "MED")]
    picks_sorted = sorted(
        picks,
        key=lambda r: (0 if r["Confidence"] == "HIGH" else 1, -r["Final Win Prob (Home)"]),
    )

    if picks_sorted:
        print(f"\n  TABLE 4 — HIGH / MED CONFIDENCE PICKS ({len(picks_sorted)} game(s))\n")
        pick_rows = [
            [
                r["Confidence"],
                r["Predicted Winner"],
                f"{r['Away Team']} @ {r['Home Team']}",
                f"{r['Final Win Prob (Home)']:.1%}" if r["Predicted Winner"] == r["Home Team"]
                    else f"{r['Final Win Prob (Away)']:.1%}",
                r["Home ML"] if r["Predicted Winner"] == r["Home Team"] else r["Away ML"],
                f"{r['Model Std Dev']:.3f}",
            ]
            for r in picks_sorted
        ]
        print(tabulate(
            pick_rows,
            headers=["Conf", "Pick", "Matchup", "Win Prob", "Moneyline", "Std Dev"],
            tablefmt="rounded_outline",
        ))
    else:
        print("\n  TABLE 4 — No HIGH or MED confidence picks today.\n")

    # ── Summary stats ─────────────────────────────────────────────────────────
    home_picks = sum(1 for r in results if r["Predicted Winner"] == r["Home Team"])
    away_picks = len(results) - home_picks
    conf_counts = {t: sum(1 for r in results if r["Confidence"] == t)
                   for t in ("HIGH", "MED", "LOW")}

    print()
    print("  SUMMARY")
    print(f"    Total games       : {len(results)}")
    print(f"    Home team picks   : {home_picks}   Away team picks: {away_picks}")
    print(f"    Confidence dist.  : HIGH={conf_counts['HIGH']}  MED={conf_counts['MED']}  LOW={conf_counts['LOW']}")
    print()

# =============================================================================
# FEATURE ENRICHMENT — mirrors Cell 14B + 14C of predictor notebook
# =============================================================================

def _format_feature_label(fname: str) -> str:
    """Convert raw feature name to a readable label, e.g. 'diff_kenpom_rank' → 'Δ KenPom Rank'."""
    label = (fname
             .replace("diff_", "Δ ")
             .replace("home_", "Home ")
             .replace("away_", "Away "))
    return " ".join(w if w == "Δ" else w.replace("_", " ").title() for w in label.split())


def _compute_feature_drivers(
    home_name: str,
    away_name: str,
    models: dict,
    lookup: dict,
    numeric_cols: list[str],
    feature_names: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Average contribution across Ridge, GB, and Logistic for the top N features.
    Mirrors Cell 14B of the predictor notebook.
    Positive contribution = favors home team. For away picks the website flips the label.
    """
    home_vec    = _get_feature_vector(home_name, lookup, numeric_cols)
    away_vec    = _get_feature_vector(away_name, lookup, numeric_cols)
    diff_vec    = home_vec - away_vec
    feat_vec    = np.concatenate([home_vec, away_vec, diff_vec, [0.0]])  # is_home=0
    feat_scaled = models["scaler"].transform(feat_vec.reshape(1, -1))[0]

    scores: dict[str, list[float]] = {}

    # Ridge: coefficient × scaled value → score-diff contribution
    try:
        ridge_c = np.clip(
            np.array(models["lr_model"].coef_).flatten() * feat_scaled, -1e4, 1e4
        )
        for i, c in enumerate(ridge_c):
            fn = feature_names[i]
            if fn != "is_home" and np.isfinite(c):
                scores.setdefault(fn, []).append(float(c))
    except Exception:
        pass

    # GB: importance × |scaled value| → unsigned magnitude
    try:
        gb_c = models["sgd_model"].feature_importances_ * np.abs(feat_scaled)
        for i, c in enumerate(gb_c):
            fn = feature_names[i]
            if fn != "is_home" and np.isfinite(c):
                scores.setdefault(fn, []).append(float(c))
    except Exception:
        pass

    # Logistic: coefficient × scaled value → log-odds contribution
    try:
        log_c = np.clip(
            np.array(models["logistic_model"].coef_[0]).flatten() * feat_scaled, -1e4, 1e4
        )
        for i, c in enumerate(log_c):
            fn = feature_names[i]
            if fn != "is_home" and np.isfinite(c):
                scores.setdefault(fn, []).append(float(c))
    except Exception:
        pass

    averaged = [
        (fn, float(np.mean(vs)))
        for fn, vs in scores.items()
        if vs and np.isfinite(np.mean(vs))
    ]
    averaged.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            "feature":      fn,
            "label":        _format_feature_label(fn),
            "contribution": round(avg, 4),
            "favors_home":  bool(avg > 0),
        }
        for fn, avg in averaged[:top_n]
    ]


def _fetch_team_articles(team_name: str, max_articles: int = 3) -> list[dict]:
    """
    Fetch Google News RSS articles for a team.
    Falls back to a Google News search URL if feedparser is unavailable or fails.
    Mirrors Cell 14C of the predictor notebook.
    """
    fallback_url = (
        "https://news.google.com/search?q="
        + requests.utils.quote(f"{team_name} mens basketball NCAAB")
        + "&hl=en-US"
    )
    fallback = [{"title": f"Search: {team_name} NCAAB news",
                 "url": fallback_url, "source": "search"}]

    if not _FEEDPARSER_AVAILABLE:
        return fallback

    try:
        q   = requests.utils.quote(f"{team_name} men's basketball NCAAB")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = _feedparser.parse(url)
        out: list[dict] = []
        seen: set[str]  = set()
        for entry in feed.entries:
            link  = getattr(entry, "link", "")
            title = getattr(entry, "title", "")[:120]
            if link and link not in seen:
                seen.add(link)
                out.append({"title": title, "url": link, "source": "Google News"})
            if len(out) >= max_articles:
                break
        return out if out else fallback
    except Exception:
        return fallback


def _ml_to_decimal(ml_str: str) -> Optional[float]:
    """Convert American moneyline string to decimal odds."""
    try:
        ml = int(str(ml_str).replace("+", "").strip())
        return round(1 + ml / 100, 4) if ml > 0 else round(1 + 100 / abs(ml), 4)
    except Exception:
        return None


def _build_top_picks(
    results: list[dict],
    odds_lookup: Optional[dict] = None,
    stake: float = 5.0,
    top_n: int = 3,
) -> list[dict]:
    """
    Top N straight picks sorted by confidence → EV → win_prob.
    Mirrors Cell 14 of the predictor notebook.
    Only includes MED/HIGH confidence picks with win_prob ≥ 55%.

    odds_lookup: dict[cbs_home_name → {home_ml, away_ml, spread}] from odds_features.
    If provided, each pick is enriched with vegas_ml and vegas_spread.
    """
    conf_rank = {"HIGH": 3, "MED": 2}
    candidates = []
    for r in results:
        conf = r["Confidence"]
        if conf not in conf_rank:
            continue
        if r["Predicted Winner"] == r["Home Team"]:
            win_prob = r["Final Win Prob (Home)"]
            ml_str   = r["Home ML"]
            bet_side = "HOME"
        else:
            win_prob = r["Final Win Prob (Away)"]
            ml_str   = r["Away ML"]
            bet_side = "AWAY"
        if win_prob < 0.55:
            continue
        dec = _ml_to_decimal(str(ml_str))
        if dec is None or dec <= 1.0:
            continue
        profit = round((dec - 1) * stake, 2)
        ev     = round(win_prob * profit - (1 - win_prob) * stake, 4)

        # Vegas lines enrichment (no-op if odds_lookup not available)
        odds_entry = None
        if odds_lookup and r["Home Team"] in odds_lookup:
            odds_entry    = odds_lookup[r["Home Team"]]
            vegas_ml      = odds_entry["home_ml"] if bet_side == "HOME" else odds_entry["away_ml"]
            vegas_spread  = odds_entry["spread"]
            odds_movement = (
                odds_entry["home_ml_movement"] if bet_side == "HOME"
                else odds_entry["away_ml_movement"]
            )
        else:
            vegas_ml      = "N/A"
            vegas_spread  = "N/A"
            odds_movement = "—"

        home_ml_mov = odds_entry.get("home_ml_movement", "—") if odds_entry else "—"
        away_ml_mov = odds_entry.get("away_ml_movement", "—") if odds_entry else "—"
        candidates.append({
            "pick":             r["Predicted Winner"],
            "matchup":          f"{r['Away Team']} @ {r['Home Team']}",
            "home_team":        r["Home Team"],
            "away_team":        r["Away Team"],
            "bet_side":         bet_side,
            "ml":               str(ml_str),
            "win_prob":         round(win_prob, 3),
            "confidence":       conf,
            "stake":            stake,
            "if_win":           profit,
            "ev":               ev,
            "vegas_ml":         vegas_ml,
            "vegas_spread":     vegas_spread,
            "odds_movement":    odds_movement,
            "home_ml_movement": home_ml_mov,
            "away_ml_movement": away_ml_mov,
            "_cr":              conf_rank[conf],
        })
    candidates.sort(key=lambda x: (x["_cr"], x["ev"], x["win_prob"]), reverse=True)
    out = []
    for i, c in enumerate(candidates[:top_n], 1):
        c = {k: v for k, v in c.items() if k != "_cr"}
        c["rank"] = i
        out.append(c)
    return out


# =============================================================================
# JSON EXPORT (§6.3 Requirements)
# =============================================================================

def export_json(
    results: list[dict],
    script_dir: str,
    project_dir: str,
    models: Optional[dict] = None,
    metadata: Optional[dict] = None,
    lookup: Optional[dict] = None,
    odds_lookup: Optional[dict] = None,
) -> str:
    """
    Serialise predictions to predictions_latest.json (§6.3 schema).
    When models/metadata/lookup are provided, enriches MED/HIGH picks with:
      - feature_drivers  (top 5 feature contributions — Cell 14B)
      - articles_home / articles_away  (Google News RSS — Cell 14C)
    Also adds a top_picks section (betting optimizer — Cell 14).
    Writes to Python scripts/ and copies to website/.
    """
    home_picks  = sum(1 for r in results if r["Predicted Winner"] == r["Home Team"])
    conf_counts = {t: sum(1 for r in results if r["Confidence"] == t)
                   for t in ("HIGH", "MED", "LOW")}

    def _safe_float(v):
        if v == "N/A" or v is None:
            return None
        try:
            f = float(v)
            return None if (f != f) else f
        except (TypeError, ValueError):
            return None

    # ── Betting optimizer top picks (Cell 14) ─────────────────────────────────
    top_picks    = _build_top_picks(results, odds_lookup=odds_lookup)
    combined_ev  = round(sum(tp["ev"] for tp in top_picks), 4)

    # ── Article cache: fetch only for top-pick teams (Cell 14C) ───────────────
    article_cache: dict[str, list[dict]] = {}
    top_pick_team_set: set[str] = set()
    for tp in top_picks:
        top_pick_team_set.add(tp["home_team"])
        top_pick_team_set.add(tp["away_team"])

    if top_pick_team_set:
        log.info("Fetching articles for %d top-pick teams…", len(top_pick_team_set))
        for team in sorted(top_pick_team_set):
            article_cache[team] = _fetch_team_articles(team)
            time.sleep(0.3)

    # ── Feature name list from metadata ───────────────────────────────────────
    feature_names = metadata.get("feature_names", []) if metadata else []
    numeric_cols_meta = metadata.get("numeric_cols", []) if metadata else []

    # ── Build enriched predictions ────────────────────────────────────────────
    predictions = []
    for r in results:
        # Feature drivers for MED/HIGH picks (Cell 14B)
        feature_drivers: list[dict] = []
        if models and feature_names and lookup and r["Confidence"] in ("HIGH", "MED"):
            try:
                feature_drivers = _compute_feature_drivers(
                    r["Home Team"], r["Away Team"],
                    models, lookup, numeric_cols_meta, feature_names,
                )
            except Exception as exc:
                log.debug("Feature drivers failed for %s @ %s: %s",
                          r["Away Team"], r["Home Team"], exc)

        # Articles for top-pick teams only
        articles_home = article_cache.get(r["Home Team"], [])
        articles_away = article_cache.get(r["Away Team"], [])

        odds_entry = odds_lookup.get(r["Home Team"]) if odds_lookup else None
        predictions.append({
            "home_team":           r["Home Team"],
            "away_team":           r["Away Team"],
            "status":              r.get("Status", ""),
            "ridge_diff":          _safe_float(r["Ridge Diff"]),
            "gb_diff":             _safe_float(r["GB Diff"]),
            "nn_reg_diff":         _safe_float(r["NN Reg Diff"]),
            "reg_ensemble_diff":   _safe_float(r["Reg Ensemble Diff"]),
            "logistic_prob":       _safe_float(r["Logistic Prob"]),
            "bayes_prob":          _safe_float(r["Bayes Prob"]),
            "nn_cls_prob":         _safe_float(r["NN Cls Prob"]),
            "cls_ensemble_prob":   _safe_float(r["Cls Ensemble Prob"]),
            "reg_implied_prob":    _safe_float(r["Reg Implied Prob"]),
            "final_win_prob_home": _safe_float(r["Final Win Prob (Home)"]),
            "final_win_prob_away": _safe_float(r["Final Win Prob (Away)"]),
            "predicted_winner":    r["Predicted Winner"],
            "home_ml":             r["Home ML"],
            "away_ml":             r["Away ML"],
            "model_std_dev":       _safe_float(r["Model Std Dev"]),
            "confidence":          r["Confidence"],
            "home_ml_movement":    odds_entry.get("home_ml_movement", "—") if odds_entry else "—",
            "away_ml_movement":    odds_entry.get("away_ml_movement", "—") if odds_entry else "—",
            "feature_drivers":     feature_drivers,
            "articles_home":       articles_home,
            "articles_away":       articles_away,
        })

    perf = metadata.get("validation_metrics", {}) if metadata else {}
    payload = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "date":         TODAY.isoformat(),
        "summary": {
            "total_games":          len(results),
            "home_picks":           home_picks,
            "away_picks":           len(results) - home_picks,
            "tensorflow_available": TF_AVAILABLE,
            "confidence":           conf_counts,
            "combined_ev":          combined_ev,
        },
        "model_performance": {
            "schema_version": metadata.get("schema_version") if metadata else None,
            "metrics":        perf,
        },
        "top_picks":   top_picks,
        "predictions": predictions,
    }

    scripts_path = os.path.join(script_dir, "predictions_latest.json")
    with open(scripts_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("JSON exported       : %s  (%d predictions, %d top picks)",
             scripts_path, len(predictions), len(top_picks))

    website_path = os.path.join(project_dir, "website", "predictions_latest.json")
    if os.path.isdir(os.path.dirname(website_path)):
        shutil.copy2(scripts_path, website_path)
        log.info("JSON copied to      : %s", website_path)

    return scripts_path

# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="NCAAB Predict Today")
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Write predictions_latest.json to Python scripts/ and website/",
    )
    args = parser.parse_args()

    log.info("=" * 68)
    log.info("NCAAB Predict Today  —  %s", TODAY.isoformat())
    log.info("TF=%s  export_json=%s", TF_AVAILABLE, args.export_json)
    log.info("=" * 68)

    # ── Step 1: Fetch today's games ────────────────────────────────────────────
    log.info("Step 1/4 — Fetching today's CBS games (all statuses)")
    games = get_todays_games(TODAY)
    if not games:
        log.warning("No games found for %s — check CBS or try a different date.", TODAY)
        print(f"\n  No games found on CBS for {TODAY.isoformat()}.\n")
        return

    # ── Step 1.5: Fetch Vegas lines ────────────────────────────────────────────
    log.info("Step 1.5 — Fetching Vegas lines from The Odds API")
    odds_lookup: dict = {}
    if _ODDS_AVAILABLE:
        cbs_names_today = list(
            {g["home_name"] for g in games} | {g["away_name"] for g in games}
        )
        odds_lookup = _odds_features.build_odds_lookup(
            _odds_features.fetch_odds(), cbs_names_today
        )
        log.info("Vegas odds coverage : %d/%d games matched", len(odds_lookup), len(games))
    else:
        log.warning("odds_features module not available — skipping Vegas lines")

    # ── Step 2: Load models from cache ────────────────────────────────────────
    log.info("Step 2/4 — Loading models from cache")
    models, metadata = load_models(CACHE_DIR)

    numeric_cols     = metadata["numeric_cols"]
    ensemble_weights = metadata["ensemble_weights"]
    is_home_idx      = metadata["is_home_idx"]

    log.info(
        "Cache metadata      : n_features=%d  features/team=%d  is_home_idx=%d",
        metadata["n_features"], len(numeric_cols), is_home_idx,
    )

    # ── Step 3: Build team feature lookup from CSVs ────────────────────────────
    log.info("Step 3/4 — Building team feature lookup from CSVs")
    lookup = build_team_feature_lookup(numeric_cols)

    today_teams  = set(g["home_name"] for g in games) | set(g["away_name"] for g in games)
    missing      = [t for t in sorted(today_teams) if t not in lookup]
    if missing:
        log.warning(
            "%d today's teams not in feature lookup (zero vectors used): %s",
            len(missing), ", ".join(missing),
        )
    else:
        log.info("Feature coverage    : all %d teams found in lookup", len(today_teams))

    # ── Step 4: Predict all games ──────────────────────────────────────────────
    log.info("Step 4/4 — Running predictions (%d games)", len(games))
    results = []
    for game in games:
        try:
            pred = predict_game(game, models, lookup, numeric_cols, ensemble_weights)
            results.append(pred)
        except Exception as exc:
            log.error(
                "Prediction failed for %s @ %s: %s",
                game["away_name"], game["home_name"], exc,
            )

    # ── Display ────────────────────────────────────────────────────────────────
    display_results(results)

    # ── JSON export ────────────────────────────────────────────────────────────
    if args.export_json:
        export_json(results, _SCRIPT_DIR, _PROJECT_DIR,
                    models=models, metadata=metadata, lookup=lookup,
                    odds_lookup=odds_lookup)


if __name__ == "__main__":
    main()
