# -*- coding: utf-8 -*-
"""
March Madness Bracket Predictor — 2026 NCAA Tournament
=======================================================
Reuses the existing NCAAB ML models and feature infrastructure from predict_today.py
to simulate all 6 tournament rounds and output a completed bracket.

Key differences from predict_today.py:
  - is_home = 0.0 for all games (neutral site)
  - Lower seed occupies team1 slot (win prob = "team1 wins")
  - No CBS scraping — bracket is hard-coded from Selection Sunday results
  - Optional feature refresh for tournament teams before predicting

Usage:
  python "Python scripts/march_madness_bracket.py"
  python "Python scripts/march_madness_bracket.py" --export-json
  python "Python scripts/march_madness_bracket.py" --region East
  python "Python scripts/march_madness_bracket.py" --skip-refresh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime
from typing import Optional

import numpy as np

# Allow imports from the Python scripts directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_today import (
    TF_AVAILABLE,
    _assign_confidence,
    _get_feature_vector,
    _prob_to_moneyline,
    build_team_feature_lookup,
    load_models,
)

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(_SCRIPT_DIR, "model_cache")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# =============================================================================
# CBS NAME MAPPINGS
# Maps bracket team name → CBS team name used in feature CSVs
# Teams that match exactly need no entry.
# =============================================================================

BRACKET_TO_CBS: dict[str, str] = {
    "Michigan State":     "Michigan St.",
    "North Carolina":     "N. Carolina",
    "Iowa State":         "Iowa St.",
    "St. Mary's":         "Saint Mary's",
    "Utah State":         "Utah St.",
    "North Dakota State": "N. Dakota St.",
    "Kennesaw State":     "Kennesaw St.",
    "Wright State":       "Wright St.",
    "Tennessee State":    "Tennessee St.",
    "St. John's":         "St. John's (NY)",
    "LIU":                "Long Island",
    "Cal Baptist":        "Cal Baptist",
    "High Point":         "High Point",
    "Hawaii":             "Hawaii",
    "Queens":             "Queens (NC)",
}


def cbs(name: str) -> str:
    """Translate bracket team name to CBS feature CSV name."""
    return BRACKET_TO_CBS.get(name, name)


# =============================================================================
# 2026 BRACKET DEFINITION (Selection Sunday 2026-03-15)
# =============================================================================

# First Four results — defaults used when --first-four flag not supplied.
# Keys: "{Region}_{seed_slot}" → predicted winner (bracket name)
FIRST_FOUR_DEFAULTS: dict[str, str] = {
    "South_16":   "Lehigh",    # Prairie View A&M vs Lehigh
    "Midwest_16": "Howard",    # UMBC vs Howard
    "West_11":    "NC State",  # Texas vs NC State
    "Midwest_11": "SMU",       # Miami (OH) vs SMU
}

# First Four game details (for output display)
FIRST_FOUR_GAMES: list[dict] = [
    {"slot": "South_16",   "team1": "Prairie View A&M", "team2": "Lehigh"},
    {"slot": "Midwest_16", "team1": "UMBC",             "team2": "Howard"},
    {"slot": "West_11",    "team1": "Texas",            "team2": "NC State"},
    {"slot": "Midwest_11", "team1": "Miami (OH)",       "team2": "SMU"},
]

# Full bracket: region → seed → team name (bracket name, not CBS name)
# None = filled by First Four winner
BRACKET_2026: dict[str, dict[int, Optional[str]]] = {
    "East": {
        1:  "Duke",
        2:  "UConn",
        3:  "Michigan State",
        4:  "Kansas",
        5:  "St. John's",
        6:  "Louisville",
        7:  "UCLA",
        8:  "Ohio State",
        9:  "TCU",
        10: "UCF",
        11: "South Florida",
        12: "Northern Iowa",
        13: "Cal Baptist",
        14: "North Dakota State",
        15: "Furman",
        16: "Siena",
    },
    "West": {
        1:  "Arizona",
        2:  "Purdue",
        3:  "Gonzaga",
        4:  "Arkansas",
        5:  "Wisconsin",
        6:  "BYU",
        7:  "Miami",
        8:  "Villanova",
        9:  "Utah State",
        10: "Missouri",
        11: None,           # First Four: West_11
        12: "High Point",
        13: "Hawaii",
        14: "Kennesaw State",
        15: "Queens",
        16: "LIU",
    },
    "South": {
        1:  "Florida",
        2:  "Houston",
        3:  "Illinois",
        4:  "Nebraska",
        5:  "Vanderbilt",
        6:  "North Carolina",
        7:  "St. Mary's",
        8:  "Clemson",
        9:  "Iowa",
        10: "Texas A&M",
        11: "VCU",
        12: "McNeese",
        13: "Troy",
        14: "Penn",
        15: "Idaho",
        16: None,           # First Four: South_16
    },
    "Midwest": {
        1:  "Michigan",
        2:  "Iowa State",
        3:  "Virginia",
        4:  "Alabama",
        5:  "Texas Tech",
        6:  "Tennessee",
        7:  "Kentucky",
        8:  "Georgia",
        9:  "Saint Louis",
        10: "Santa Clara",
        11: None,           # First Four: Midwest_11
        12: "Akron",
        13: "Hofstra",
        14: "Wright State",
        15: "Tennessee State",
        16: None,           # First Four: Midwest_16
    },
}

# Standard Round-of-64 seed matchup pairs (lower seed vs higher seed)
R64_MATCHUPS: list[tuple[int, int]] = [
    (1, 16), (8, 9),
    (5, 12), (4, 13),
    (6, 11), (3, 14),
    (7, 10), (2, 15),
]

# Final Four matchups: East vs West, South vs Midwest
FF_MATCHUPS: list[tuple[str, str]] = [
    ("East",    "West"),
    ("South",   "Midwest"),
]

# =============================================================================
# NEUTRAL-SITE PREDICTION
# =============================================================================

def predict_neutral(
    team1: str,          # bracket name — occupies "home" slot (lower seed)
    team2: str,          # bracket name — occupies "away" slot (higher seed)
    seed1: int,
    seed2: int,
    models: dict,
    lookup: dict,
    numeric_cols: list[str],
    ensemble_weights: dict,
    round_name: str = "",
    region: str = "",
) -> dict:
    """
    Predict a neutral-site game. is_home=0.0 for both teams.

    team1 is the lower-seeded (favored) team and occupies the home feature slot.
    Win probability returned is "team1 wins".
    """
    t1_cbs = cbs(team1)
    t2_cbs = cbs(team2)

    if t1_cbs not in lookup:
        log.warning("No features for '%s' (CBS: '%s') — zero vector used", team1, t1_cbs)
    if t2_cbs not in lookup:
        log.warning("No features for '%s' (CBS: '%s') — zero vector used", team2, t2_cbs)

    v1 = _get_feature_vector(t1_cbs, lookup, numeric_cols)
    v2 = _get_feature_vector(t2_cbs, lookup, numeric_cols)

    # Neutral site: is_home=0.0 (no home advantage)
    vec    = np.concatenate([v1, v2, v1 - v2, [0.0]])
    X_game = models["scaler"].transform(vec.reshape(1, -1))

    ew = ensemble_weights

    # ── Regression models ─────────────────────────────────────────────────────
    lr_diff  = float(models["lr_model"].predict(X_game)[0])
    sgd_diff = float(models["sgd_model"].predict(X_game)[0])

    if models["nn_reg_model"] is not None:
        nn_diff  = float(models["nn_reg_model"].predict(X_game, verbose=0)[0][0])
        reg_diff = (ew["reg_w_ridge"] * lr_diff
                    + ew["reg_w_gb"]    * sgd_diff
                    + ew["reg_w_nn"]    * nn_diff)
    else:
        nn_diff  = float("nan")
        _r = ew["reg_w_ridge"] / (ew["reg_w_ridge"] + ew["reg_w_gb"])
        _g = ew["reg_w_gb"]    / (ew["reg_w_ridge"] + ew["reg_w_gb"])
        reg_diff = _r * lr_diff + _g * sgd_diff

    reg_implied_prob = float(1.0 / (1.0 + np.exp(-reg_diff / 10.0)))

    # ── Classification models ──────────────────────────────────────────────────
    log_prob   = float(models["logistic_model"].predict_proba(X_game)[0][1])
    bayes_prob = float(models["bayes_model"].predict_proba(X_game)[0][1])

    if models["nn_cls_model"] is not None:
        nn_cls_prob = float(models["nn_cls_model"].predict(X_game, verbose=0)[0][0])
        cls_prob    = (ew["cls_w_logistic"] * log_prob
                       + ew["cls_w_bayes"]  * bayes_prob
                       + ew["cls_w_nn"]     * nn_cls_prob)
    else:
        nn_cls_prob = float("nan")
        _total   = ew["cls_w_logistic"] + ew["cls_w_bayes"]
        cls_prob = ((ew["cls_w_logistic"] / _total) * log_prob
                    + (ew["cls_w_bayes"]  / _total) * bayes_prob)

    # ── Final ensemble ─────────────────────────────────────────────────────────
    final_prob = float(np.clip(
        ew["final_w_reg"] * reg_implied_prob + ew["final_w_cls"] * cls_prob,
        0.01, 0.99,
    ))
    team2_prob = 1.0 - final_prob

    # ── Confidence ────────────────────────────────────────────────────────────
    all_probs = [reg_implied_prob, log_prob, bayes_prob]
    if not np.isnan(nn_cls_prob):
        all_probs.append(nn_cls_prob)
    prob_std   = float(np.std(all_probs))
    margin     = abs(final_prob - 0.5)
    confidence = _assign_confidence(prob_std, margin)

    winner      = team1 if final_prob >= 0.5 else team2
    winner_prob = final_prob if final_prob >= 0.5 else team2_prob
    winner_seed = seed1 if winner == team1 else seed2
    loser_seed  = seed2 if winner == team1 else seed1
    # Upset = higher seed number (weaker team) wins
    is_upset    = winner_seed > loser_seed

    return {
        "region":       region,
        "round":        round_name,
        "seed1":        seed1,
        "team1":        team1,
        "seed2":        seed2,
        "team2":        team2,
        "team1_prob":   round(final_prob, 4),
        "team2_prob":   round(team2_prob, 4),
        "winner":       winner,
        "winner_seed":  winner_seed,
        "winner_prob":  round(winner_prob, 4),
        "confidence":   confidence,
        "prob_std":     round(prob_std, 4),
        "reg_diff":     round(reg_diff, 2),
        "team1_ml":     _prob_to_moneyline(final_prob),
        "team2_ml":     _prob_to_moneyline(team2_prob),
        "is_upset":     is_upset,
    }


# =============================================================================
# ROUND SIMULATION ENGINE
# =============================================================================

def _resolve_first_four(
    bracket: dict[str, dict[int, Optional[str]]],
    first_four_results: dict[str, str],
) -> dict[str, dict[int, str]]:
    """
    Replace None seed slots with the First Four winner for that slot.
    Returns a new bracket with all slots filled.
    """
    resolved: dict[str, dict[int, str]] = {}
    for region, seeds in bracket.items():
        resolved[region] = {}
        for seed, team in seeds.items():
            if team is None:
                slot_key = f"{region}_{seed}"
                winner = first_four_results.get(slot_key)
                if winner is None:
                    raise ValueError(
                        f"First Four slot '{slot_key}' not resolved. "
                        f"Provide --first-four or update FIRST_FOUR_DEFAULTS."
                    )
                resolved[region][seed] = winner
            else:
                resolved[region][seed] = team
    return resolved


def simulate_region(
    region_name: str,
    seeds: dict[int, str],
    models: dict,
    lookup: dict,
    numeric_cols: list[str],
    ensemble_weights: dict,
) -> tuple[str, int, list[dict]]:
    """
    Simulate 4 rounds for one region.

    Returns: (regional_champion_name, regional_champion_seed, list_of_game_results)
    """
    all_results: list[dict] = []

    # Current field: list of (seed, team_name)
    field = [(seed, team) for seed, team in seeds.items()]
    field.sort(key=lambda x: x[0])  # sort by seed ascending

    round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]

    for round_idx, round_name in enumerate(round_names):
        round_results: list[dict] = []
        next_field: list[tuple[int, str]] = []

        if round_idx == 0:
            # R64: fixed seed matchups
            seed_map = {s: t for s, t in field}
            for s_low, s_high in R64_MATCHUPS:
                t_low  = seed_map[s_low]
                t_high = seed_map[s_high]
                res = predict_neutral(
                    t_low, t_high, s_low, s_high,
                    models, lookup, numeric_cols, ensemble_weights,
                    round_name=round_name, region=region_name,
                )
                round_results.append(res)
                next_field.append((res["winner_seed"], res["winner"]))
        else:
            # R32 → Elite 8: winners play in bracket order (top half vs bottom half)
            # Pair consecutive winners: (0,1), (2,3), (4,5), (6,7)
            for i in range(0, len(field), 2):
                s1, t1 = field[i]
                s2, t2 = field[i + 1]
                # Lower seed number = favored = team1 slot
                if s1 <= s2:
                    res = predict_neutral(
                        t1, t2, s1, s2,
                        models, lookup, numeric_cols, ensemble_weights,
                        round_name=round_name, region=region_name,
                    )
                else:
                    res = predict_neutral(
                        t2, t1, s2, s1,
                        models, lookup, numeric_cols, ensemble_weights,
                        round_name=round_name, region=region_name,
                    )
                round_results.append(res)
                next_field.append((res["winner_seed"], res["winner"]))

        all_results.extend(round_results)
        field = next_field

    # field should now have exactly 1 team: the regional champion
    assert len(field) == 1, f"Expected 1 champion, got {field}"
    champ_seed, champ_name = field[0]
    return champ_name, champ_seed, all_results


def simulate_tournament(
    bracket: dict[str, dict[int, Optional[str]]],
    first_four_results: dict[str, str],
    models: dict,
    lookup: dict,
    numeric_cols: list[str],
    ensemble_weights: dict,
    region_filter: Optional[str] = None,
) -> dict:
    """
    Run all 6 rounds. Returns full results tree:
    {
        "first_four": [...],
        "regions": { "East": {...}, ... },
        "final_four": [...],
        "championship": {...},
        "champion": str,
    }
    """
    # ── Predict First Four games ───────────────────────────────────────────────
    ff_game_results: list[dict] = []
    for game in FIRST_FOUR_GAMES:
        slot   = game["slot"]
        region = slot.split("_")[0]
        seed   = int(slot.split("_")[1])
        t1, t2 = game["team1"], game["team2"]
        res = predict_neutral(
            t1, t2, seed, seed,
            models, lookup, numeric_cols, ensemble_weights,
            round_name="First Four", region=region,
        )
        res["ff_slot"]  = slot
        res["ff_winner_overridden"] = (first_four_results.get(slot) != res["winner"]
                                       and first_four_results.get(slot) is not None)
        # Override with actual result if provided
        if first_four_results.get(slot) is not None:
            res["winner"]       = first_four_results[slot]
            res["winner_seed"]  = seed
        ff_game_results.append(res)

    # ── Resolve First Four slots ───────────────────────────────────────────────
    resolved_bracket = _resolve_first_four(bracket, first_four_results)

    # ── Simulate each region ───────────────────────────────────────────────────
    region_results: dict[str, dict] = {}
    champions: dict[str, tuple[str, int]] = {}  # region → (name, seed)

    regions_to_run = (
        [region_filter] if region_filter else list(resolved_bracket.keys())
    )

    for region in regions_to_run:
        seeds = resolved_bracket[region]
        champ_name, champ_seed, game_list = simulate_region(
            region, seeds, models, lookup, numeric_cols, ensemble_weights
        )
        region_results[region] = {
            "champion":      champ_name,
            "champion_seed": champ_seed,
            "games":         game_list,
        }
        champions[region] = (champ_name, champ_seed)
        log.info("%-8s champion : [%d] %s", region, champ_seed, champ_name)

    # ── Final Four (only if all 4 regions simulated) ───────────────────────────
    ff_results: list[dict] = []
    championship_result: Optional[dict] = None
    champion_name: Optional[str]        = None

    if not region_filter and len(champions) == 4:
        ff_winners: list[tuple[str, int]] = []

        for r1, r2 in FF_MATCHUPS:
            n1, s1 = champions[r1]
            n2, s2 = champions[r2]
            if s1 <= s2:
                res = predict_neutral(
                    n1, n2, s1, s2,
                    models, lookup, numeric_cols, ensemble_weights,
                    round_name="Final Four", region=f"{r1}/{r2}",
                )
            else:
                res = predict_neutral(
                    n2, n1, s2, s1,
                    models, lookup, numeric_cols, ensemble_weights,
                    round_name="Final Four", region=f"{r1}/{r2}",
                )
            ff_results.append(res)
            ff_winners.append((res["winner"], res["winner_seed"]))
            log.info("Final Four %s/%s : [%d] %s  def. [%d] %s  (%.1f%%  %s)",
                     r1, r2, res["winner_seed"], res["winner"],
                     s1 if res["winner"] == n1 else s2,
                     n1 if res["winner"] == n2 else n2,
                     res["winner_prob"] * 100, res["confidence"])

        # ── Championship ──────────────────────────────────────────────────────
        (n1, s1), (n2, s2) = ff_winners
        if s1 <= s2:
            championship_result = predict_neutral(
                n1, n2, s1, s2,
                models, lookup, numeric_cols, ensemble_weights,
                round_name="Championship", region="National",
            )
        else:
            championship_result = predict_neutral(
                n2, n1, s2, s1,
                models, lookup, numeric_cols, ensemble_weights,
                round_name="Championship", region="National",
            )
        champion_name = championship_result["winner"]
        log.info("Champion            : [%d] %s  (%.1f%%  %s)",
                 championship_result["winner_seed"], champion_name,
                 championship_result["winner_prob"] * 100,
                 championship_result["confidence"])

    return {
        "first_four":    ff_game_results,
        "regions":       region_results,
        "final_four":    ff_results,
        "championship":  championship_result,
        "champion":      champion_name,
    }


# =============================================================================
# FEATURE REFRESH FOR TOURNAMENT TEAMS
# =============================================================================

def refresh_tournament_features(tournament_cbs_names: list[str]) -> None:
    """
    Refresh feature data for tournament teams before predicting.

    1. Re-run kenpom_ratings.main()     — full T-Rank scrape (~30s)
    2. Re-run efficiency_metrics.main() — full efficiency scrape (~30s)
    3. Upsert sentiment rows for tournament teams in sentiment_features.csv
    """
    import importlib
    import pandas as pd

    # KenPom / T-Rank refresh
    try:
        kenpom = importlib.import_module("kenpom_ratings")
        log.info("Refreshing KenPom ratings (full re-scrape)…")
        kenpom.main()
    except Exception as exc:
        log.warning("KenPom refresh failed: %s — using existing CSV", exc)

    # Efficiency metrics refresh
    try:
        efficiency = importlib.import_module("efficiency_metrics")
        log.info("Refreshing efficiency metrics (full re-scrape)…")
        efficiency.main()
    except Exception as exc:
        log.warning("Efficiency refresh failed: %s — using existing CSV", exc)

    # Sentiment refresh — upsert rows for tournament teams
    try:
        from sentiment_features import extract_team_features  # type: ignore
    except ImportError:
        log.warning("sentiment_features module not importable — skipping sentiment refresh")
        return

    sent_path = os.path.join(_SCRIPT_DIR, "sentiment_features.csv")
    df = pd.read_csv(sent_path) if os.path.isfile(sent_path) else pd.DataFrame()

    log.info("Refreshing sentiment for %d tournament teams…", len(tournament_cbs_names))
    updated = 0
    for name in tournament_cbs_names:
        try:
            feats = extract_team_features(name)
            feats["team_name"] = name
            row_df = pd.DataFrame([feats])
            if not df.empty and "team_name" in df.columns and name in df["team_name"].values:
                df = df[df["team_name"] != name]
            df = pd.concat([df, row_df], ignore_index=True)
            updated += 1
        except Exception as exc:
            log.warning("Sentiment refresh failed for %s: %s — keeping existing row", name, exc)

    df.to_csv(sent_path, index=False)
    log.info("Sentiment CSV updated: %d/%d rows upserted → %s",
             updated, len(tournament_cbs_names), sent_path)


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def _fmt_prob(p: float) -> str:
    return f"{p * 100:.1f}%"


def _fmt_game_line(res: dict) -> str:
    """Format a single game result as a compact one-liner."""
    seed1, t1 = res["seed1"], res["team1"]
    seed2, t2 = res["seed2"], res["team2"]
    winner     = res["winner"]
    loser      = t2 if winner == t1 else t1
    w_seed     = res["winner_seed"]
    l_seed     = seed2 if winner == t1 else seed1
    prob       = _fmt_prob(res["winner_prob"])
    conf       = res["confidence"]
    upset_tag  = "  ⚡ UPSET" if res["is_upset"] else ""
    return f"  [{w_seed:2d}] {winner:<26} def. [{l_seed:2d}] {loser:<26}  ({prob}  {conf}){upset_tag}"


def print_bracket_summary(results: dict, region_filter: Optional[str] = None) -> None:
    """Print a full formatted bracket summary to stdout."""
    width = 72

    print()
    print("═" * width)
    print("  2026 NCAA TOURNAMENT — ML BRACKET PREDICTIONS")
    print(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC  |  TF={TF_AVAILABLE}")
    print("═" * width)

    # ── First Four ────────────────────────────────────────────────────────────
    if results["first_four"] and not region_filter:
        print()
        print("  FIRST FOUR")
        print("  " + "─" * (width - 2))
        for res in results["first_four"]:
            print(_fmt_game_line(res))

    # ── Regional rounds ───────────────────────────────────────────────────────
    for region_name, region_data in results["regions"].items():
        games      = region_data["games"]
        champion   = region_data["champion"]
        champ_seed = region_data["champion_seed"]

        print()
        print(f"  {region_name.upper()} REGION")
        print("  " + "─" * (width - 2))

        round_groups: dict[str, list[dict]] = {}
        for g in games:
            round_groups.setdefault(g["round"], []).append(g)

        for round_name in ["Round of 64", "Round of 32", "Sweet 16", "Elite 8"]:
            rg = round_groups.get(round_name, [])
            if not rg:
                continue
            print(f"\n  {round_name}:")
            for res in rg:
                print(_fmt_game_line(res))

        print(f"\n  ► {region_name.upper()} CHAMPION: [{champ_seed}] {champion}")

    # ── Final Four ────────────────────────────────────────────────────────────
    if results["final_four"]:
        print()
        print("  FINAL FOUR")
        print("  " + "─" * (width - 2))
        for res in results["final_four"]:
            print(_fmt_game_line(res))

    # ── Championship ──────────────────────────────────────────────────────────
    if results["championship"]:
        cr = results["championship"]
        print()
        print("  CHAMPIONSHIP")
        print("  " + "─" * (width - 2))
        print(_fmt_game_line(cr))
        print()
        print("  " + "★" * (width - 2))
        print(f"  ★  PREDICTED CHAMPION: [{cr['winner_seed']}] {cr['winner']}  "
              f"({_fmt_prob(cr['winner_prob'])}  {cr['confidence']})  ★")
        print("  " + "★" * (width - 2))

    # ── Upset alerts ──────────────────────────────────────────────────────────
    all_games: list[dict] = list(results.get("first_four", []))
    for rd in results["regions"].values():
        all_games.extend(rd["games"])
    all_games.extend(results.get("final_four", []))
    if results.get("championship"):
        all_games.append(results["championship"])

    upsets = [g for g in all_games if g.get("is_upset") and g["confidence"] in ("HIGH", "MED")]
    if upsets:
        print()
        print("  UPSET ALERTS  (model favors higher seed with MED/HIGH confidence)")
        print("  " + "─" * (width - 2))
        for res in upsets:
            seed1, t1 = res["seed1"], res["team1"]
            seed2, t2 = res["seed2"], res["team2"]
            winner  = res["winner"]
            w_seed  = res["winner_seed"]
            loser   = t2 if winner == t1 else t1
            l_seed  = seed2 if winner == t1 else seed1
            rnd     = res["round"]
            region  = res.get("region", "")
            print(f"  ⚡ [{w_seed}] {winner} def. [{l_seed}] {loser}  "
                  f"({_fmt_prob(res['winner_prob'])}  {res['confidence']})  "
                  f"— {region} {rnd}")

    # ── Coverage report ───────────────────────────────────────────────────────
    zero_vec_teams = [
        g["team1"] for g in all_games if g["team1"] not in _LOOKUP_CACHE
    ] + [
        g["team2"] for g in all_games if g["team2"] not in _LOOKUP_CACHE
    ]
    zero_cbs_teams = sorted(set(cbs(t) for t in zero_vec_teams))
    if zero_cbs_teams:
        print()
        print(f"  ZERO-VECTOR TEAMS ({len(zero_cbs_teams)}) — no features in CSV:")
        for t in zero_cbs_teams:
            print(f"    • {t}")

    print()


# Global cache for coverage check (populated in main)
_LOOKUP_CACHE: dict = {}


# =============================================================================
# JSON EXPORT
# =============================================================================

def export_bracket_json(results: dict, output_path: str) -> None:
    """Serialise full bracket results to JSON."""

    def _ser(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_ser)
    log.info("Bracket JSON exported: %s", output_path)

    project_dir  = os.path.dirname(_SCRIPT_DIR)
    website_path = os.path.join(project_dir, "website", "bracket_predictions_2026.json")
    if os.path.isdir(os.path.dirname(website_path)):
        shutil.copy2(output_path, website_path)
        log.info("Bracket JSON copied to: %s", website_path)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="2026 March Madness Bracket Predictor")
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Write bracket_predictions_2026.json to Python scripts/",
    )
    parser.add_argument(
        "--region",
        choices=["East", "West", "South", "Midwest"],
        default=None,
        help="Simulate only one region (skips Final Four/Championship)",
    )
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Skip feature refresh (use existing CSVs — faster)",
    )
    parser.add_argument(
        "--first-four",
        nargs="*",
        metavar="SLOT=WINNER",
        help=(
            "Override First Four results. E.g.: "
            "--first-four South_16=Lehigh West_11='NC State'"
        ),
    )
    args = parser.parse_args()

    log.info("=" * 68)
    log.info("2026 March Madness Bracket Predictor")
    log.info("TF=%-5s  skip_refresh=%s  region=%s",
             TF_AVAILABLE, args.skip_refresh, args.region or "ALL")
    log.info("=" * 68)

    # ── Parse First Four overrides ─────────────────────────────────────────────
    first_four_results = dict(FIRST_FOUR_DEFAULTS)
    if args.first_four:
        for item in args.first_four:
            if "=" in item:
                slot, winner = item.split("=", 1)
                first_four_results[slot.strip()] = winner.strip()
                log.info("First Four override : %s → %s", slot.strip(), winner.strip())

    # ── Collect all tournament team CBS names ──────────────────────────────────
    all_bracket_teams: list[str] = []
    for region_seeds in BRACKET_2026.values():
        for team in region_seeds.values():
            if team is not None:
                all_bracket_teams.append(team)
    # Add First Four teams
    for game in FIRST_FOUR_GAMES:
        all_bracket_teams.extend([game["team1"], game["team2"]])

    all_cbs_names = sorted(set(cbs(t) for t in all_bracket_teams))
    log.info("Tournament teams    : %d unique CBS names", len(all_cbs_names))

    # ── Feature refresh ────────────────────────────────────────────────────────
    if not args.skip_refresh:
        log.info("Step 1/3 — Refreshing features for tournament teams")
        refresh_tournament_features(all_cbs_names)
    else:
        log.info("Step 1/3 — Skipping feature refresh (--skip-refresh)")

    # ── Load models ────────────────────────────────────────────────────────────
    log.info("Step 2/3 — Loading models from cache")
    models, metadata = load_models(CACHE_DIR)

    numeric_cols     = metadata["numeric_cols"]
    ensemble_weights = metadata["ensemble_weights"]

    # ── Build feature lookup ───────────────────────────────────────────────────
    log.info("Step 3/3 — Building team feature lookup")
    lookup = build_team_feature_lookup(numeric_cols)

    # Populate global cache for coverage check
    global _LOOKUP_CACHE
    _LOOKUP_CACHE = lookup

    # Check coverage
    missing = [t for t in all_cbs_names if t not in lookup]
    if missing:
        log.warning(
            "%d tournament teams not in feature lookup (zero vectors): %s",
            len(missing), ", ".join(missing),
        )
    else:
        log.info("Feature coverage    : all %d tournament teams found", len(all_cbs_names))

    # ── Simulate tournament ────────────────────────────────────────────────────
    log.info("Simulating tournament…")
    results = simulate_tournament(
        BRACKET_2026,
        first_four_results,
        models,
        lookup,
        numeric_cols,
        ensemble_weights,
        region_filter=args.region,
    )

    # ── Print summary ──────────────────────────────────────────────────────────
    print_bracket_summary(results, region_filter=args.region)

    # ── JSON export ────────────────────────────────────────────────────────────
    if args.export_json:
        output_path = os.path.join(_SCRIPT_DIR, "bracket_predictions_2026.json")
        # Add metadata to payload
        results["generated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        results["tournament_year"] = 2026
        results["model_schema_version"] = metadata.get("schema_version", "?")
        results["tf_available"] = TF_AVAILABLE
        export_bracket_json(results, output_path)


if __name__ == "__main__":
    main()
