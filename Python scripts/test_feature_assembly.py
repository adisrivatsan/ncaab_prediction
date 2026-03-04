"""
Test — Feature Assembly & X Input Matrix
=========================================
Runs feature_assembly pipeline and reports:
  1. Shape of each intermediate DataFrame
  2. Shape of the final training_df
  3. Shape of X (the numeric feature matrix fed to model training)
  4. Feature column inventory
  5. Basic sanity checks (nulls, ranges)
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import numpy as np

# Make sure imports resolve relative to this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import feature_assembly as fa

# =============================================================================
# Run assembly pipeline
# =============================================================================

print("=" * 60)
print("TEST: Feature Assembly & X Matrix")
print("=" * 60)

# Load all sources
games_df      = fa.load_games(fa.CBS_CSV_PATH)
sentiment_df  = fa.load_sentiment(fa.SENTIMENT_CSV_PATH)
efficiency_df = fa.load_efficiency(fa.EFFICIENCY_CSV_PATH, fa.EFF_MAPPING_PATH)
kenpom_df     = fa.load_kenpom(fa.KENPOM_CSV_PATH)

# Build master features
features_df = fa.build_master_features(sentiment_df, efficiency_df, kenpom_df)

# Assemble training_df
training_df = fa.assemble_training_df(games_df, features_df)

# =============================================================================
# Intermediate shapes
# =============================================================================

print()
print("── Intermediate DataFrames ──────────────────────────────")
print(f"  games_df           : {games_df.shape}")
print(f"  sentiment_df       : {sentiment_df.shape}")
print(f"  efficiency_df      : {efficiency_df.shape}")
print(f"  kenpom_df          : {kenpom_df.shape}")
print(f"  master_features_df : {features_df.shape}")
print(f"  training_df        : {training_df.shape}")

# =============================================================================
# X matrix (feature columns only — what gets fed to the models)
# =============================================================================

GAME_COLS = [
    "date", "away_name", "away_score", "home_name", "home_score",
    "status", "home_team_won", "winner_name", "winner_score",
    "loser_name", "loser_score", "score_diff",
]

# Feature cols = everything except game metadata
feature_cols = [c for c in training_df.columns if c not in GAME_COLS]

# Home and away feature columns
home_feat_cols = [c for c in feature_cols if c.startswith("home_")]
away_feat_cols = [c for c in feature_cols if c.startswith("away_")]

# X matrix: all numeric feature columns
X = training_df[feature_cols].values

# y targets
y_diff   = training_df["score_diff"].values
y_binary = training_df["home_team_won"].values

print()
print("── X Input Matrix ───────────────────────────────────────")
print(f"  X.shape             : {X.shape}   ← (n_games × n_features)")
print(f"  y_diff.shape        : {y_diff.shape}   (score differential regression target)")
print(f"  y_binary.shape      : {y_binary.shape}   (home_team_won classification target)")
print()
print(f"  Total feature cols  : {len(feature_cols)}")
print(f"    home_* cols       : {len(home_feat_cols)}")
print(f"    away_* cols       : {len(away_feat_cols)}")

# =============================================================================
# Feature breakdown by source
# =============================================================================

sent_cols   = [c for c in feature_cols
               if c not in [f"home_{x}" for x in fa.EFFICIENCY_FEATURE_COLS +
                                                    fa.KENPOM_FEATURE_COLS]
               and c not in [f"away_{x}" for x in fa.EFFICIENCY_FEATURE_COLS +
                                                    fa.KENPOM_FEATURE_COLS]]
eff_cols    = [c for c in feature_cols
               if any(c.endswith(f) for f in fa.EFFICIENCY_FEATURE_COLS)]
kenpom_cols = [c for c in feature_cols
               if any(c.endswith(f) for f in fa.KENPOM_FEATURE_COLS)]

print()
print("── Feature Breakdown ────────────────────────────────────")
print(f"  Sentiment cols      : {len(sent_cols)}  ({len(sent_cols)//2} per team × 2 sides)")
print(f"  Efficiency cols     : {len(eff_cols)}   ({len(eff_cols)//2} per team × 2 sides)")
print(f"  KenPom cols         : {len(kenpom_cols)}   ({len(kenpom_cols)//2} per team × 2 sides)")

# =============================================================================
# Null / zero checks on X
# =============================================================================

null_count  = np.isnan(X).sum()
zero_rows   = (X == 0).all(axis=1).sum()
null_cols   = [feature_cols[i] for i in range(X.shape[1]) if np.isnan(X[:, i]).any()]

print()
print("── Sanity Checks ────────────────────────────────────────")
print(f"  NaN values in X     : {null_count}")
print(f"  All-zero rows in X  : {zero_rows}   (games where team had no feature data)")
print(f"  Cols with any NaN   : {len(null_cols)}")
if null_cols:
    for c in null_cols[:5]:
        print(f"    - {c}")
    if len(null_cols) > 5:
        print(f"    ... and {len(null_cols) - 5} more")

print()
print(f"  home_team_won dist  : {int(y_binary.sum())} home wins / "
      f"{len(y_binary) - int(y_binary.sum())} away wins  "
      f"({y_binary.mean():.1%} home win rate)")
print(f"  score_diff range    : {y_diff.min():.0f} to {y_diff.max():.0f} pts  "
      f"(mean {y_diff.mean():.1f})")

# =============================================================================
# Feature column listing
# =============================================================================

print()
print("── Feature Columns (home side only, away mirrors) ───────")
for i, col in enumerate(home_feat_cols):
    base = col.replace("home_", "")
    print(f"  [{i+1:02d}] {base}")

print()
print("=" * 60)
print(f"RESULT: X.shape = {X.shape}")
print("=" * 60)
