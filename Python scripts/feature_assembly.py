"""
Feature Assembly — Joins all data sources into a single training_df
====================================================================
Reads four source CSVs produced by the standalone data scripts, then
replicates the notebook Cell 5 merge logic to produce training_df.csv.

Input files (all in the same directory as this script):
    cbs_games.csv            — 709 rows × 12 cols  (game outcomes)
    sentiment_features.csv   — 365 teams × 32 cols (31 features + team_name)
    efficiency_metrics.csv   — 365 rows × 9 cols   (4 efficiency metrics)
    team_name_mapping.csv    — 365 rows             (Sports Ref → CBS name map)
    kenpom_ratings.csv       — 365 rows × 9 cols   (4 T-Rank ratings)

Output:
    training_df.csv  — one row per final game with home/away features joined

Schema:
    12 game cols + 39 home_* feature cols + 39 away_* feature cols = 90 cols
    (31 sentiment + 4 efficiency + 4 kenpom = 39 features per team)

Schema version: 1.0
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

# cbs_games.csv is written to the project root by cbs_scraper.py
CBS_CSV_PATH        = os.path.join(_PROJECT_DIR, "cbs_games.csv")
SENTIMENT_CSV_PATH  = os.path.join(_SCRIPT_DIR, "sentiment_features.csv")
EFFICIENCY_CSV_PATH = os.path.join(_SCRIPT_DIR, "efficiency_metrics.csv")
EFF_MAPPING_PATH    = os.path.join(_SCRIPT_DIR, "team_name_mapping.csv")
KENPOM_CSV_PATH     = os.path.join(_SCRIPT_DIR, "kenpom_ratings.csv")
TRAINING_OUT_PATH   = os.path.join(_SCRIPT_DIR, "training_df.csv")

AS_OF_DATE     = date.today().isoformat()
SCHEMA_VERSION = "1.0"

# Feature columns from each source (metadata columns excluded)
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
# LOADERS
# =============================================================================

def load_games(path: str) -> pd.DataFrame:
    """
    Load cbs_games.csv and enforce numeric types on score/outcome columns.
    Expected columns: date, away_name, away_score, home_name, home_score,
        status, home_team_won, winner_name, winner_score, loser_name,
        loser_score, score_diff
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CBS games CSV not found: {path}")

    df = pd.read_csv(path)

    for col in ["away_score", "home_score", "winner_score", "loser_score",
                "home_team_won", "score_diff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("Games loaded        : %d rows × %d cols", len(df), df.shape[1])
    return df


def load_sentiment(path: str) -> pd.DataFrame:
    """
    Load sentiment_features.csv.
    Join key: team_name (CBS names — same strings used in cbs_games.csv).
    Returns 31 feature cols + team_name col.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sentiment features CSV not found: {path}")

    df = pd.read_csv(path)

    feature_cols = [c for c in df.columns if c != "team_name"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    log.info(
        "Sentiment loaded    : %d teams × %d features",
        len(df), len(feature_cols),
    )
    return df


def load_efficiency(eff_path: str, mapping_path: str) -> pd.DataFrame:
    """
    Load efficiency_metrics.csv and resolve Sports Reference team names to
    CBS names using team_name_mapping.csv.

    Returns DataFrame with columns: cbs_name + EFFICIENCY_FEATURE_COLS.
    Teams that cannot be mapped to a CBS name are dropped with a warning.
    """
    for p in (eff_path, mapping_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Efficiency source not found: {p}")

    eff_df     = pd.read_csv(eff_path)
    mapping_df = pd.read_csv(mapping_path)

    # mapping_df columns: source_name, cbs_name, match_confidence
    merged = eff_df.merge(
        mapping_df[["source_name", "cbs_name"]],
        left_on="team_id",
        right_on="source_name",
        how="left",
    )

    unmapped = merged["cbs_name"].isna().sum()
    if unmapped > 0:
        log.warning(
            "Efficiency mapping  : %d/%d teams could not map to CBS names "
            "(features will be 0.0 for those teams)",
            unmapped, len(merged),
        )

    result = merged[["cbs_name"] + EFFICIENCY_FEATURE_COLS].copy()
    for col in EFFICIENCY_FEATURE_COLS:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    result = result.dropna(subset=["cbs_name"])

    # Multiple Sports Reference source names can fuzzy-match to the same CBS
    # name. Deduplicate by averaging numeric columns per cbs_name.
    dup_count = result["cbs_name"].duplicated().sum()
    if dup_count > 0:
        log.warning(
            "Efficiency: %d duplicate cbs_name rows detected — averaging values",
            dup_count,
        )
        result = (
            result.groupby("cbs_name", as_index=False)[EFFICIENCY_FEATURE_COLS]
            .mean()
        )

    result = result.reset_index(drop=True)
    log.info(
        "Efficiency loaded   : %d teams × %d features (after CBS name mapping)",
        len(result), len(EFFICIENCY_FEATURE_COLS),
    )
    return result


def load_kenpom(path: str) -> pd.DataFrame:
    """
    Load kenpom_ratings.csv. cbs_name column already contains mapped CBS names
    (produced by kenpom_ratings.py). No secondary mapping step required.

    Returns DataFrame with columns: cbs_name + KENPOM_FEATURE_COLS.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"KenPom ratings CSV not found: {path}")

    df = pd.read_csv(path)

    result = df[["cbs_name"] + KENPOM_FEATURE_COLS].copy()
    for col in KENPOM_FEATURE_COLS:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    result = result.dropna(subset=["cbs_name"])

    # Multiple Torvik source names can fuzzy-match to the same CBS name.
    # Deduplicate by averaging numeric columns per cbs_name.
    dup_count = result["cbs_name"].duplicated().sum()
    if dup_count > 0:
        log.warning(
            "KenPom: %d duplicate cbs_name rows detected — averaging values",
            dup_count,
        )
        result = (
            result.groupby("cbs_name", as_index=False)[KENPOM_FEATURE_COLS]
            .mean()
        )

    result = result.reset_index(drop=True)
    log.info(
        "KenPom loaded       : %d teams × %d features",
        len(result), len(KENPOM_FEATURE_COLS),
    )
    return result

# =============================================================================
# FEATURE ASSEMBLY
# =============================================================================

def build_master_features(
    sentiment_df:  pd.DataFrame,
    efficiency_df: pd.DataFrame,
    kenpom_df:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all feature sources into a single master_features_df.
    Join key: team_name (CBS names).

    Merge order:
        1. Sentiment features (base — all 365 CBS teams present)
        2. LEFT JOIN efficiency features  on team_name == cbs_name
        3. LEFT JOIN kenpom features      on team_name == cbs_name

    Missing values from failed joins are filled with 0.0 per project convention:
        pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    """
    df = sentiment_df.copy()

    # Join efficiency (Sports Ref names already resolved to CBS names)
    df = df.merge(
        efficiency_df.rename(columns={"cbs_name": "team_name"}),
        on="team_name",
        how="left",
    )

    # Join kenpom (cbs_name already mapped)
    df = df.merge(
        kenpom_df.rename(columns={"cbs_name": "team_name"}),
        on="team_name",
        how="left",
    )

    # Enforce numeric + fill nulls from unmatched joins (project convention)
    feature_cols = [c for c in df.columns if c != "team_name"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    log.info(
        "master_features_df  : %d teams × %d features (%d sentiment, "
        "%d efficiency, %d kenpom)",
        len(df), len(feature_cols),
        len([c for c in df.columns if c in sentiment_df.columns and c != "team_name"]),
        len(EFFICIENCY_FEATURE_COLS),
        len(KENPOM_FEATURE_COLS),
    )
    return df


def assemble_training_df(
    games_df:    pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replicates notebook Cell 5 merge logic exactly:

        home_feats = features_df.add_prefix('home_')
                                .rename(columns={'home_team_name': 'home_name'})
        away_feats = features_df.add_prefix('away_')
                                .rename(columns={'away_team_name': 'away_name'})
        training_df = games_df.merge(home_feats, on='home_name', how='left')
        training_df = training_df.merge(away_feats, on='away_name', how='left')
        training_df = training_df.dropna(subset=['home_team_won'])
        training_df['home_team_won'] = training_df['home_team_won'].astype(int)
    """
    home_feats = features_df.add_prefix("home_").rename(
        columns={"home_team_name": "home_name"}
    )
    away_feats = features_df.add_prefix("away_").rename(
        columns={"away_team_name": "away_name"}
    )

    training_df = games_df.merge(home_feats, on="home_name", how="left")
    training_df = training_df.merge(away_feats, on="away_name", how="left")

    # Drop rows with no outcome (non-final games produce null home_team_won)
    training_df = (
        training_df
        .dropna(subset=["home_team_won"])
        .reset_index(drop=True)
    )
    training_df["home_team_won"] = training_df["home_team_won"].astype(int)

    log.info(
        "training_df         : %d rows × %d cols",
        len(training_df), training_df.shape[1],
    )
    return training_df

# =============================================================================
# VALIDATION
# =============================================================================

def validate_training_df(training_df: pd.DataFrame) -> int:
    """
    Run basic sanity checks on the assembled training_df.
    Returns the number of warnings issued.
    """
    warn_count = 0

    # Check for unexpected nulls in feature columns
    feature_cols = [c for c in training_df.columns
                    if c.startswith("home_") or c.startswith("away_")]
    null_counts = training_df[feature_cols].isna().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        log.warning(
            "Unexpected nulls in %d feature column(s) after assembly:",
            len(cols_with_nulls),
        )
        for col, n in cols_with_nulls.items():
            log.warning("  %s: %d nulls", col, n)
        warn_count += len(cols_with_nulls)

    # Check score_diff integrity
    if "score_diff" in training_df.columns:
        computed = training_df["home_score"] - training_df["away_score"]
        mismatch = (training_df["score_diff"] - computed).abs() > 0.01
        if mismatch.sum() > 0:
            log.warning(
                "score_diff mismatch in %d rows (home_score - away_score != score_diff)",
                mismatch.sum(),
            )
            warn_count += 1

    # Check home win rate is within reasonable bounds (35–75%)
    home_win_rate = training_df["home_team_won"].mean()
    if not (0.35 <= home_win_rate <= 0.75):
        log.warning(
            "Home win rate %.1f%% is outside expected range [35%%, 75%%]",
            home_win_rate * 100,
        )
        warn_count += 1

    if warn_count == 0:
        log.info("Validation passed — no issues detected")
    else:
        log.warning("Validation complete — %d issue(s) flagged", warn_count)

    return warn_count

# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary(
    games_df:    pd.DataFrame,
    features_df: pd.DataFrame,
    training_df: pd.DataFrame,
    warn_count:  int,
) -> None:
    """Print a human-readable summary of the assembled training data."""
    feature_cols = [c for c in features_df.columns if c != "team_name"]
    n_sent = len([c for c in feature_cols
                  if c not in EFFICIENCY_FEATURE_COLS + KENPOM_FEATURE_COLS])

    home_win_rate  = training_df["home_team_won"].mean()
    score_diff_min = training_df["score_diff"].min() if "score_diff" in training_df.columns else float("nan")
    score_diff_max = training_df["score_diff"].max() if "score_diff" in training_df.columns else float("nan")

    # Coverage: count games where at least one home feature is 0 (possible unmatched team)
    if "home_sent_overall" in training_df.columns:
        unmatched_home = (training_df["home_sent_overall"] == 0.0).sum()
        unmatched_away = (training_df["away_sent_overall"] == 0.0).sum()
    else:
        unmatched_home = unmatched_away = "N/A"

    kenpom_missing = (
        (training_df["home_adj_em"] == 0.0).sum()
        if "home_adj_em" in training_df.columns
        else "N/A"
    )
    eff_missing = (
        (training_df["home_efg_pct"] == 0.0).sum()
        if "home_efg_pct" in training_df.columns
        else "N/A"
    )

    print()
    print("=" * 60)
    print("FEATURE ASSEMBLY SUMMARY")
    print("=" * 60)
    print(f"  As-of date            : {AS_OF_DATE}")
    print(f"  Schema version        : {SCHEMA_VERSION}")
    print()
    print(f"  Source CSVs loaded:")
    print(f"    cbs_games           : {len(games_df)} rows × {games_df.shape[1]} cols")
    print(f"    sentiment_features  : {len(features_df)} teams")
    print()
    print(f"  Features per team     : {len(feature_cols)}")
    print(f"    Sentiment (news)    : {n_sent}")
    print(f"    Efficiency (4-Fac.) : {len(EFFICIENCY_FEATURE_COLS)}")
    print(f"    KenPom / T-Rank     : {len(KENPOM_FEATURE_COLS)}")
    print()
    print(f"  master_features_df    : {features_df.shape}")
    print(f"  training_df           : {training_df.shape}")
    print()
    print(f"  Home win rate         : {home_win_rate:.1%}")
    print(f"  Score diff range      : {score_diff_min:.0f} to {score_diff_max:.0f} pts")
    print()
    print(f"  Feature coverage (0.0 = unmatched team):")
    print(f"    Sentiment missing (home) : {unmatched_home} games")
    print(f"    Sentiment missing (away) : {unmatched_away} games")
    print(f"    Efficiency missing (home): {eff_missing} games")
    print(f"    KenPom missing (home)    : {kenpom_missing} games")
    print()
    print(f"  Validation warnings   : {warn_count}")
    print()
    print(f"  Output: {TRAINING_OUT_PATH}")
    print("=" * 60)

    # Top-5 game preview
    preview_cols = ["date", "away_name", "away_score",
                    "home_name", "home_score", "home_team_won", "score_diff"]
    available = [c for c in preview_cols if c in training_df.columns]
    if available:
        print()
        print("  Sample (first 5 rows, game cols only):")
        print(training_df[available].head().to_string(index=False))
        print()

# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    log.info("=" * 60)
    log.info("NCAAB Feature Assembly")
    log.info("As-of: %s  |  Schema: %s", AS_OF_DATE, SCHEMA_VERSION)
    log.info("=" * 60)

    # Step 1: Load all source CSVs
    log.info("Step 1/4 — Loading source CSVs")
    games_df      = load_games(CBS_CSV_PATH)
    sentiment_df  = load_sentiment(SENTIMENT_CSV_PATH)
    efficiency_df = load_efficiency(EFFICIENCY_CSV_PATH, EFF_MAPPING_PATH)
    kenpom_df     = load_kenpom(KENPOM_CSV_PATH)

    # Step 2: Build master_features_df (one row per team, all features joined)
    log.info("Step 2/4 — Building master_features_df")
    features_df = build_master_features(sentiment_df, efficiency_df, kenpom_df)

    # Step 3: Assemble training_df (notebook Cell 5 merge logic)
    log.info("Step 3/4 — Assembling training_df")
    training_df = assemble_training_df(games_df, features_df)

    # Step 4: Validate + save
    log.info("Step 4/4 — Validating and saving")
    warn_count = validate_training_df(training_df)
    training_df.to_csv(TRAINING_OUT_PATH, index=False)
    log.info("Saved -> %s  (%d rows × %d cols)",
             TRAINING_OUT_PATH, len(training_df), training_df.shape[1])

    print_summary(games_df, features_df, training_df, warn_count)


if __name__ == "__main__":
    main()
