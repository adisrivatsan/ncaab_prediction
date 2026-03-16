"""
Model Training — NCAAB Prediction System
=========================================
Reads training_df.csv (produced by feature_assembly.py) and trains 6 ML models:

    lr_model       : Ridge(alpha=RIDGE_ALPHA)                  — regressor
    sgd_model      : XGBoostRegressor (XGBRegressor)            — regressor  [misnomer]
    nn_reg_model   : Neural Network Regressor                  — regressor
    logistic_model : LogisticRegression(elasticnet)            — classifier
    bayes_model    : GaussianNB (Bayesian probability)         — classifier  [NEW]
    nn_cls_model   : Neural Network Classifier                 — classifier

Matchup vector dimensions (standalone pipeline, 39 features/team):
    home_feats(39) + away_feats(39) + diff_feats(39) + is_home(1) = 118 dims

Dataset mirroring: each game produces 2 training rows (home + away perspectives),
so X.shape = (n_games * 2, 118).

is_home is ZEROED to 0.0 before fitting any model (business rule §9 — must never
be removed). It is set to 1.0 only at prediction time.

Classification ensemble weights (with Bayes added):
    35% Logistic  +  20% Bayes  +  45% NN Cls

Final ensemble (unchanged from notebook):
    55% regression-implied prob  +  45% classification ensemble

Model cache: Python scripts/model_cache/
    scaler.joblib, ridge_model.joblib, gb_model.joblib,
    logistic_model.joblib, bayes_model.joblib,
    nn_regressor.keras, nn_classifier.keras, metadata.joblib

Schema version: 2.0  (bump and clear cache when feature schema changes)
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from datetime import date
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# XGBoost is optional — falls back to GradientBoostingRegressor if not installed
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# TensorFlow is optional — NNs are skipped gracefully if not installed
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CELL 1 — CONFIGURATION
# =============================================================================

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

TRAINING_CSV_PATH = os.path.join(_SCRIPT_DIR, "training_df.csv")
CACHE_DIR         = os.path.join(_SCRIPT_DIR, "model_cache")

AS_OF_DATE     = date.today().isoformat()
SCHEMA_VERSION = "2.0"

# ── Cache control ──────────────────────────────────────────────────────────────
USE_CACHE     = True   # load models from cache if they exist
FORCE_RETRAIN = False  # if True: always retrain and overwrite cache

# ── Regularization hyperparameters ────────────────────────────────────────────
RIDGE_ALPHA      = 20.0

# XGBoost hyperparameters (used when XGB_AVAILABLE; otherwise GB fallback is used)
XGB_N_ESTIMATORS   = 300
XGB_LEARNING_RATE  = 0.05
XGB_MAX_DEPTH      = 4
XGB_MIN_CHILD_WT   = 3
XGB_SUBSAMPLE      = 0.85
XGB_COLSAMPLE      = 0.85
XGB_L2_REG         = 1.0
XGB_L1_REG         = 0.1
XGB_EARLY_STOPPING = 20

# GradientBoosting fallback (used only when XGBoost not installed)
GB_N_ESTIMATORS  = 200
GB_LEARNING_RATE = 0.05
GB_MAX_DEPTH     = 4
GB_MIN_SAMPLES   = 3
GB_SUBSAMPLE     = 0.85
LOGISTIC_C       = 0.5
BAYES_VAR_SMOOTH = 1e-9   # GaussianNB variance smoothing floor
NN_L2            = 0.01
NN_DROPOUT       = 0.05
NN_HIDDEN_1      = 128
NN_HIDDEN_2      = 64
NN_LEARNING_RATE = 0.0003
NN_BATCH_SIZE    = 32
NN_MAX_EPOCHS    = 300
NN_PATIENCE      = 20

# ── Home bias audit thresholds (§5.4 Requirements) ───────────────────────────
HOME_BIAS_RED    = 0.72   # > 72% predicted home wins → 🔴 stop
HOME_BIAS_YELLOW = 0.62   # 62–72% → 🟡 caution
HOME_BIAS_LOW    = 0.50   # < 50% → 🟡 undervaluing home

# ── Ensemble weights ──────────────────────────────────────────────────────────
# Regression ensemble (score differential)
REG_W_RIDGE = 0.30
REG_W_GB    = 0.30
REG_W_NN    = 0.40

# Classification ensemble — Bayes added; weights sum to 1.0
CLS_W_LOGISTIC = 0.35
CLS_W_BAYES    = 0.20
CLS_W_NN       = 0.45

# Final ensemble (regression-implied prob + classification ensemble)
FINAL_W_REG = 0.55
FINAL_W_CLS = 0.45

# Non-feature columns present in training_df (excluded from feature matrix)
_GAME_COLS = {
    "date", "away_name", "away_score", "home_name", "home_score",
    "status", "home_team_won", "winner_name", "winner_score",
    "loser_name", "loser_score", "score_diff",
}

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
# CELL 2 — LOAD TRAINING DATA + RECONSTRUCT TEAM LOOKUP
# =============================================================================

def load_training_df(path: str) -> pd.DataFrame:
    """
    Load training_df.csv and enforce numeric types on all feature columns.
    Raises FileNotFoundError if the file does not exist (run feature_assembly.py first).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"training_df.csv not found: {path}\n"
            "Run feature_assembly.py first to generate it."
        )
    df = pd.read_csv(path)

    feature_cols = [c for c in df.columns if c not in _GAME_COLS]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    log.info("training_df loaded  : %d rows × %d cols", len(df), df.shape[1])
    return df


def build_team_lookup(training_df: pd.DataFrame) -> tuple[dict, list[str]]:
    """
    Reconstruct a per-team feature dict from training_df.

    For each game row:
      - extract home_* feature cols → strip 'home_' prefix → store for home_name
      - extract away_* feature cols → strip 'away_' prefix → store for away_name

    When a team appears in multiple rows, the first occurrence is used; features
    are static per team within a training window.

    Returns:
        lookup       : dict[team_name → dict[feature_name → float]]
        numeric_cols : ordered list of per-team feature names (without prefix)
    """
    # home_* feature cols excluding game-identity columns
    _exclude = {"home_name", "home_score", "home_team_won"}
    home_feature_cols = [
        c for c in training_df.columns
        if c.startswith("home_") and c not in _exclude
    ]
    numeric_cols = [c[len("home_"):] for c in home_feature_cols]
    away_feature_cols = [f"away_{nc}" for nc in numeric_cols]

    lookup: dict = {}
    for _, row in training_df.iterrows():
        home_name = row["home_name"]
        away_name = row["away_name"]

        if home_name not in lookup:
            lookup[home_name] = {
                nc: float(row[hc])
                for nc, hc in zip(numeric_cols, home_feature_cols)
            }
        if away_name not in lookup:
            lookup[away_name] = {
                nc: float(row[ac])
                for nc, ac in zip(numeric_cols, away_feature_cols)
            }

    log.info("Team lookup         : %d unique teams, %d features each",
             len(lookup), len(numeric_cols))
    return lookup, numeric_cols

# =============================================================================
# CELL 9 (ADAPTED) — MATCHUP VECTOR CONSTRUCTION
# =============================================================================

def get_feature_vector(
    team_name: str,
    lookup: dict,
    cols: list[str],
) -> np.ndarray:
    """Return the feature vector for a team, or zeros if team is not in lookup."""
    if team_name in lookup:
        return np.array(
            [float(lookup[team_name].get(c, 0.0)) for c in cols],
            dtype=np.float32,
        )
    return np.zeros(len(cols), dtype=np.float32)


def build_matchup_vector(
    home_name: str,
    away_name: str,
    lookup: dict,
    cols: list[str],
    is_home: float = 1.0,
) -> np.ndarray:
    """
    Build a matchup vector for a single game.

    Structure: [home_feats | away_feats | diff_feats | is_home]
    Dimensions: len(cols)*3 + 1

    is_home = 1.0 during prediction (home-team perspective)
    is_home = 0.0 during training (zeroed in build_training_arrays)
    """
    home_vec = get_feature_vector(home_name, lookup, cols)
    away_vec = get_feature_vector(away_name, lookup, cols)
    diff_vec = home_vec - away_vec
    return np.concatenate([home_vec, away_vec, diff_vec, [is_home]])


def build_training_arrays(
    training_df: pd.DataFrame,
    lookup: dict,
    numeric_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """
    Build X, y_diff, y_binary from training_df with dataset mirroring.

    Mirroring (eliminates home positional bias during training):
      Row A — home perspective : [home_feats, away_feats,  diff,  is_home=1.0]
      Row B — away perspective : [away_feats, home_feats, -diff,  is_home=0.0]

    NOTE: is_home is set here but zeroed to 0.0 in train_models() before fitting,
    per business rule §9 (is_home zeroing must never be removed).

    Returns:
        X             : (n_games*2, n_features) raw matchup vectors
        y_diff        : (n_games*2,) score differential targets
        y_binary      : (n_games*2,) binary home-win targets (0 or 1)
        feature_names : list of feature column names (length = n_features)
        is_home_idx   : index of the is_home flag in the feature vector
    """
    n_feats = len(numeric_cols) * 3 + 1
    n_games = len(training_df)

    X      = np.zeros((n_games * 2, n_feats), dtype=np.float32)
    y_diff = np.zeros(n_games * 2, dtype=np.float32)
    y_bin  = np.zeros(n_games * 2, dtype=np.float32)

    for i, (_, row) in enumerate(training_df.iterrows()):
        home_name  = str(row["home_name"])
        away_name  = str(row["away_name"])
        score_diff = float(row["score_diff"])
        home_won   = float(row["home_team_won"])

        # Row A — home perspective
        X[2 * i]      = build_matchup_vector(home_name, away_name, lookup, numeric_cols, is_home=1.0)
        y_diff[2 * i] = score_diff
        y_bin[2 * i]  = home_won

        # Row B — away perspective (vectors swapped, diff negated, outcome flipped)
        X[2 * i + 1]      = build_matchup_vector(away_name, home_name, lookup, numeric_cols, is_home=0.0)
        y_diff[2 * i + 1] = -score_diff
        y_bin[2 * i + 1]  = 1.0 - home_won

    feature_names = (
        [f"home_{c}" for c in numeric_cols]
        + [f"away_{c}" for c in numeric_cols]
        + [f"diff_{c}" for c in numeric_cols]
        + ["is_home"]
    )
    is_home_idx = feature_names.index("is_home")

    log.info(
        "Matchup vectors     : X=%s  y_diff=%s  y_binary=%s  is_home_idx=%d",
        X.shape, y_diff.shape, y_bin.shape, is_home_idx,
    )
    return X, y_diff, y_bin, feature_names, is_home_idx

# =============================================================================
# NN ARCHITECTURE HELPERS
# =============================================================================

def _build_nn_regressor(input_dim: int) -> "keras.Model":
    """Build and compile the neural network regressor (nn_reg_model)."""
    inp = keras.Input(shape=(input_dim,))
    x   = layers.Dense(
        NN_HIDDEN_1, activation="relu",
        kernel_regularizer=regularizers.l2(NN_L2),
    )(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(NN_DROPOUT)(x)
    x   = layers.Dense(
        NN_HIDDEN_2, activation="relu",
        kernel_regularizer=regularizers.l2(NN_L2),
    )(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(NN_DROPOUT / 2)(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE),
        loss="mse",
    )
    return model


def _build_nn_classifier(input_dim: int) -> "keras.Model":
    """Build and compile the neural network classifier (nn_cls_model)."""
    inp = keras.Input(shape=(input_dim,))
    x   = layers.Dense(
        NN_HIDDEN_1, activation="relu",
        kernel_regularizer=regularizers.l2(NN_L2),
    )(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(NN_DROPOUT)(x)
    x   = layers.Dense(
        NN_HIDDEN_2, activation="relu",
        kernel_regularizer=regularizers.l2(NN_L2),
    )(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(NN_DROPOUT / 2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=NN_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model

# =============================================================================
# CELL 10 (ADAPTED) — TRAIN ALL 6 MODELS
# =============================================================================

def train_models(
    X: np.ndarray,
    y_diff: np.ndarray,
    y_binary: np.ndarray,
    is_home_idx: int,
    scaler: Optional[StandardScaler] = None,
) -> dict:
    """
    Fit all 6 models on the training arrays. Returns a dict of fitted objects
    and the fitted StandardScaler.

    Business rule §9 (is_home zeroing — must never be removed):
        X_train[:, is_home_idx] = 0.0 before fitting any model.
    The is_home flag is only valid at prediction time (set to 1.0 for the
    home-team perspective). Training on 1.0 would leak home-team identity.
    """
    # ── is_home zeroing (business rule §9) ───────────────────────────────────
    X_train = X.copy()
    X_train[:, is_home_idx] = 0.0

    # ── Scale features ────────────────────────────────────────────────────────
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
    else:
        X_scaled = scaler.transform(X_train)

    input_dim = X_scaled.shape[1]
    y_int     = y_binary.astype(int)
    models    = {"scaler": scaler}

    # ── 1. Ridge Regressor (lr_model) ─────────────────────────────────────────
    log.info("Training lr_model   (Ridge, alpha=%.1f) ...", RIDGE_ALPHA)
    lr_model = Ridge(alpha=RIDGE_ALPHA)
    lr_model.fit(X_scaled, y_diff)
    models["lr_model"] = lr_model
    log.info("  ✓ lr_model")

    # ── 2. XGBoost Regressor (sgd_model — variable name is a documented misnomer) ─
    if XGB_AVAILABLE:
        log.info("Training sgd_model  (XGBoost, n_est=%d) ...", XGB_N_ESTIMATORS)
        eval_size = max(1, int(0.15 * len(X_scaled)))
        sgd_model = xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=XGB_MAX_DEPTH,
            min_child_weight=XGB_MIN_CHILD_WT,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
            reg_lambda=XGB_L2_REG,
            reg_alpha=XGB_L1_REG,
            early_stopping_rounds=XGB_EARLY_STOPPING,
            eval_metric="mae",
            random_state=42,
            verbosity=0,
        )
        sgd_model.fit(
            X_scaled[:-eval_size], y_diff[:-eval_size],
            eval_set=[(X_scaled[-eval_size:], y_diff[-eval_size:])],
            verbose=False,
        )
    else:
        log.info("Training sgd_model  (GradientBoosting fallback, n_est=%d) ...", GB_N_ESTIMATORS)
        sgd_model = GradientBoostingRegressor(
            n_estimators=GB_N_ESTIMATORS,
            learning_rate=GB_LEARNING_RATE,
            max_depth=GB_MAX_DEPTH,
            min_samples_leaf=GB_MIN_SAMPLES,
            subsample=GB_SUBSAMPLE,
            random_state=42,
        )
        sgd_model.fit(X_scaled, y_diff)
    models["sgd_model"] = sgd_model
    log.info("  ✓ sgd_model")

    # ── 3. Logistic Regression Classifier ─────────────────────────────────────
    log.info("Training logistic_model (LogisticRegression, C=%.2f) ...", LOGISTIC_C)
    logistic_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=LOGISTIC_C,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    logistic_model.fit(X_scaled, y_int)
    models["logistic_model"] = logistic_model
    log.info("  ✓ logistic_model")

    # ── 4. Gaussian Naive Bayes Classifier (bayes_model) — NEW ────────────────
    # GaussianNB models each feature as a Gaussian conditioned on the class label.
    # var_smoothing adds a fraction of the largest per-feature variance to all
    # variances, preventing zero-variance numerical instability.
    log.info("Training bayes_model (GaussianNB, var_smoothing=%.0e) ...", BAYES_VAR_SMOOTH)
    bayes_model = GaussianNB(var_smoothing=BAYES_VAR_SMOOTH)
    bayes_model.fit(X_scaled, y_int)
    models["bayes_model"] = bayes_model
    log.info("  ✓ bayes_model")

    # ── 5. Neural Network Regressor (nn_reg_model) ────────────────────────────
    if TF_AVAILABLE:
        log.info(
            "Training nn_reg_model (NN Regressor, hidden=%d/%d, patience=%d) ...",
            NN_HIDDEN_1, NN_HIDDEN_2, NN_PATIENCE,
        )
        nn_reg_model = _build_nn_regressor(input_dim)
        nn_reg_model.fit(
            X_scaled, y_diff,
            validation_split=0.15,
            epochs=NN_MAX_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=NN_PATIENCE,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )
        models["nn_reg_model"] = nn_reg_model
        log.info("  ✓ nn_reg_model")
    else:
        log.warning("TensorFlow not available — nn_reg_model skipped (install tensorflow)")
        models["nn_reg_model"] = None

    # ── 6. Neural Network Classifier (nn_cls_model) ───────────────────────────
    if TF_AVAILABLE:
        log.info(
            "Training nn_cls_model (NN Classifier, hidden=%d/%d, patience=%d) ...",
            NN_HIDDEN_1, NN_HIDDEN_2, NN_PATIENCE,
        )
        nn_cls_model = _build_nn_classifier(input_dim)
        nn_cls_model.fit(
            X_scaled, y_binary,
            validation_split=0.15,
            epochs=NN_MAX_EPOCHS,
            batch_size=NN_BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=NN_PATIENCE,
                    restore_best_weights=True,
                )
            ],
            verbose=0,
        )
        models["nn_cls_model"] = nn_cls_model
        log.info("  ✓ nn_cls_model")
    else:
        log.warning("TensorFlow not available — nn_cls_model skipped (install tensorflow)")
        models["nn_cls_model"] = None

    return models

# =============================================================================
# CELL 10B (ADAPTED) — MODEL PERFORMANCE SUMMARY
# =============================================================================

def evaluate_models(
    models: dict,
    X: np.ndarray,
    y_diff: np.ndarray,
    y_binary: np.ndarray,
    is_home_idx: int,
    training_df: pd.DataFrame,
) -> dict:
    """
    Compute time-based 80/20 holdout performance metrics for all trained models.

    Games are sorted by date; the first 80% form the train split, the last 20%
    the test split. Sklearn models are cloned and re-fit on the train portion so
    test metrics are true out-of-sample. NNs use the already-fitted weights on
    the test rows (re-training NNs twice would be prohibitively slow).

    Returns a dict:
        "_split"       : { split_date, n_train_games, n_test_games }
        "lr_model"     : { test_mae, test_rmse, split_date, n_train_games, n_test_games }
        "sgd_model"    : { test_mae, test_rmse, ... }
        "nn_reg_model" : { test_mae, test_rmse, ... }   (no re-fit)
        "logistic_model": { test_acc, test_brier, test_logloss, test_auc, ... }
        "bayes_model"  : { test_acc, test_brier, test_logloss, test_auc, ... }
        "nn_cls_model" : { test_acc, test_brier, test_logloss, test_auc, ... }  (no re-fit)
    """
    X_eval = X.copy()
    X_eval[:, is_home_idx] = 0.0
    X_scaled = models["scaler"].transform(X_eval)
    y_int    = y_binary.astype(int)

    # ── Time-based 80/20 split ────────────────────────────────────────────────
    sorted_game_idx  = np.argsort(training_df["date"].values)
    n_games          = len(training_df)
    n_train_games    = int(0.80 * n_games)
    n_test_games     = n_games - n_train_games
    train_game_idx   = sorted_game_idx[:n_train_games]
    test_game_idx    = sorted_game_idx[n_train_games:]
    split_date       = str(training_df.iloc[sorted_game_idx[n_train_games]]["date"])

    # Map game indices to mirrored X rows
    train_rows = np.concatenate([[2 * i, 2 * i + 1] for i in train_game_idx])
    test_rows  = np.concatenate([[2 * i, 2 * i + 1] for i in test_game_idx])

    X_tr   = X_scaled[train_rows];  X_te   = X_scaled[test_rows]
    yd_tr  = y_diff[train_rows];    yd_te  = y_diff[test_rows]
    yb_tr  = y_int[train_rows];     yb_te  = y_int[test_rows]

    log.info(
        "  Time-based split    : train=%d games / test=%d games  (split_date=%s)",
        n_train_games, n_test_games, split_date,
    )

    split_meta = {
        "split_date":    split_date,
        "n_train_games": n_train_games,
        "n_test_games":  n_test_games,
    }
    results: dict = {"_split": split_meta}

    # ── Regressors (sklearn — clone + re-fit) ─────────────────────────────────
    for name in ("lr_model", "sgd_model"):
        tmp = clone(models[name])
        tmp.fit(X_tr, yd_tr)
        pred_te   = tmp.predict(X_te)
        test_mae  = float(mean_absolute_error(yd_te, pred_te))
        test_rmse = float(np.sqrt(mean_squared_error(yd_te, pred_te)))
        results[name] = {**split_meta, "test_mae": round(test_mae, 4), "test_rmse": round(test_rmse, 4)}
        log.info("  %-22s  test_MAE=%.3f  test_RMSE=%.3f", name, test_mae, test_rmse)

    # ── NN Regressor (no re-fit — evaluate pre-trained weights on test rows) ──
    if models["nn_reg_model"] is not None:
        pred_te   = models["nn_reg_model"].predict(X_te, verbose=0).flatten()
        test_mae  = float(mean_absolute_error(yd_te, pred_te))
        test_rmse = float(np.sqrt(mean_squared_error(yd_te, pred_te)))
        results["nn_reg_model"] = {**split_meta, "test_mae": round(test_mae, 4), "test_rmse": round(test_rmse, 4)}
        log.info("  %-22s  test_MAE=%.3f  test_RMSE=%.3f  (no re-fit)", "nn_reg_model", test_mae)

    # ── Classifiers (sklearn — clone + re-fit) ────────────────────────────────
    for name in ("logistic_model", "bayes_model"):
        tmp          = clone(models[name])
        tmp.fit(X_tr, yb_tr)
        pred_prob_te = tmp.predict_proba(X_te)[:, 1]
        pred_cls_te  = (pred_prob_te >= 0.5).astype(int)
        test_acc     = float(accuracy_score(yb_te, pred_cls_te))
        test_brier   = float(brier_score_loss(yb_te, pred_prob_te))
        test_logloss = float(log_loss(yb_te, pred_prob_te))
        test_auc     = float(roc_auc_score(yb_te, pred_prob_te))
        results[name] = {
            **split_meta,
            "test_acc":     round(test_acc, 4),
            "test_brier":   round(test_brier, 4),
            "test_logloss": round(test_logloss, 4),
            "test_auc":     round(test_auc, 4),
        }
        log.info(
            "  %-22s  test_acc=%.3f  Brier=%.3f  LogLoss=%.3f  AUC=%.3f",
            name, test_acc, test_brier, test_logloss, test_auc,
        )

    # ── NN Classifier (no re-fit) ─────────────────────────────────────────────
    if models["nn_cls_model"] is not None:
        pred_prob_te = models["nn_cls_model"].predict(X_te, verbose=0).flatten()
        pred_cls_te  = (pred_prob_te >= 0.5).astype(int)
        test_acc     = float(accuracy_score(yb_te, pred_cls_te))
        test_brier   = float(brier_score_loss(yb_te, pred_prob_te))
        test_logloss = float(log_loss(yb_te, pred_prob_te))
        test_auc     = float(roc_auc_score(yb_te, pred_prob_te))
        results["nn_cls_model"] = {
            **split_meta,
            "test_acc":     round(test_acc, 4),
            "test_brier":   round(test_brier, 4),
            "test_logloss": round(test_logloss, 4),
            "test_auc":     round(test_auc, 4),
        }
        log.info(
            "  %-22s  test_acc=%.3f  Brier=%.3f  AUC=%.3f  (no re-fit)",
            "nn_cls_model", test_acc, test_brier, test_auc,
        )

    return results

# =============================================================================
# CELL 13 (ADAPTED) — HOME BIAS AUDIT
# =============================================================================

def home_bias_audit(
    models: dict,
    X: np.ndarray,
    y_binary: np.ndarray,
    is_home_idx: int,
) -> tuple[float, str]:
    """
    Compute the fraction of training games where the final ensemble predicts
    a home-team win. Uses home-perspective rows (even-indexed after mirroring).

    Thresholds (§5.4 Requirements):
        > 72%  🔴 OVERFIT HOME  — stop; increase RIDGE_ALPHA, GB_MIN_SAMPLES,
                                   NN_L2, NN_DROPOUT; set FORCE_RETRAIN=True
        62–72% 🟡 CAUTION       — monitor; mild regularization increase
        50–62% 🟢 HEALTHY       — no action required
        < 50%  🟡 UNDERVALUING  — decrease regularization

    Returns:
        home_win_rate : fraction predicted home-team wins
        status        : human-readable audit result string
    """
    # Home-perspective rows (row A from mirroring; is_home=1.0 for prediction)
    X_home = X[::2].copy()
    X_home[:, is_home_idx] = 1.0
    X_scaled = models["scaler"].transform(X_home)

    # Regression ensemble → implied win probability
    lr_diff  = models["lr_model"].predict(X_scaled)
    sgd_diff = models["sgd_model"].predict(X_scaled)

    if models["nn_reg_model"] is not None:
        nn_diff  = models["nn_reg_model"].predict(X_scaled, verbose=0).flatten()
        reg_diff = REG_W_RIDGE * lr_diff + REG_W_GB * sgd_diff + REG_W_NN * nn_diff
    else:
        # Redistribute NN weight proportionally between Ridge and GB
        _ridge_w = REG_W_RIDGE / (REG_W_RIDGE + REG_W_GB)
        _gb_w    = REG_W_GB    / (REG_W_RIDGE + REG_W_GB)
        reg_diff = _ridge_w * lr_diff + _gb_w * sgd_diff

    reg_prob = 1.0 / (1.0 + np.exp(-reg_diff / 10.0))  # logistic transform

    # Classification ensemble
    log_prob   = models["logistic_model"].predict_proba(X_scaled)[:, 1]
    bayes_prob = models["bayes_model"].predict_proba(X_scaled)[:, 1]

    if models["nn_cls_model"] is not None:
        nn_prob  = models["nn_cls_model"].predict(X_scaled, verbose=0).flatten()
        cls_prob = CLS_W_LOGISTIC * log_prob + CLS_W_BAYES * bayes_prob + CLS_W_NN * nn_prob
    else:
        # Redistribute NN weight proportionally between Logistic and Bayes
        _total = CLS_W_LOGISTIC + CLS_W_BAYES
        cls_prob = (
            (CLS_W_LOGISTIC / _total) * log_prob
            + (CLS_W_BAYES / _total) * bayes_prob
        )

    final_prob    = FINAL_W_REG * reg_prob + FINAL_W_CLS * cls_prob
    home_win_rate = float((final_prob >= 0.5).mean())

    if home_win_rate > HOME_BIAS_RED:
        status = (
            f"🔴 OVERFIT HOME ({home_win_rate:.1%}) — "
            "increase RIDGE_ALPHA / GB_MIN_SAMPLES / NN_L2 / NN_DROPOUT; "
            "set FORCE_RETRAIN = True and retrain"
        )
    elif home_win_rate > HOME_BIAS_YELLOW:
        status = f"🟡 CAUTION ({home_win_rate:.1%}) — monitor; consider mild regularization increase"
    elif home_win_rate >= HOME_BIAS_LOW:
        status = f"🟢 HEALTHY ({home_win_rate:.1%}) — no action required"
    else:
        status = f"🟡 UNDERVALUING HOME ({home_win_rate:.1%}) — decrease regularization"

    log.info("Home bias audit     : %s", status)
    return home_win_rate, status

# =============================================================================
# MODEL CACHE — SAVE / LOAD
# =============================================================================

def save_models(
    models: dict,
    feature_names: list[str],
    numeric_cols: list[str],
    is_home_idx: int,
    validation_metrics: Optional[dict] = None,
) -> None:
    """Serialize all fitted models and metadata to CACHE_DIR."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    joblib.dump(models["scaler"],         os.path.join(CACHE_DIR, "scaler.joblib"))
    joblib.dump(models["lr_model"],       os.path.join(CACHE_DIR, "ridge_model.joblib"))
    joblib.dump(models["sgd_model"],      os.path.join(CACHE_DIR, "gb_model.joblib"))
    joblib.dump(models["logistic_model"], os.path.join(CACHE_DIR, "logistic_model.joblib"))
    joblib.dump(models["bayes_model"],    os.path.join(CACHE_DIR, "bayes_model.joblib"))

    if models["nn_reg_model"] is not None:
        models["nn_reg_model"].save(os.path.join(CACHE_DIR, "nn_regressor.keras"))
    if models["nn_cls_model"] is not None:
        models["nn_cls_model"].save(os.path.join(CACHE_DIR, "nn_classifier.keras"))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "as_of_date":     AS_OF_DATE,
        "feature_names":  feature_names,
        "numeric_cols":   numeric_cols,
        "n_features":     len(feature_names),
        "is_home_idx":    is_home_idx,
        "ensemble_weights": {
            "reg_w_ridge":    REG_W_RIDGE,
            "reg_w_gb":       REG_W_GB,
            "reg_w_nn":       REG_W_NN,
            "cls_w_logistic": CLS_W_LOGISTIC,
            "cls_w_bayes":    CLS_W_BAYES,
            "cls_w_nn":       CLS_W_NN,
            "final_w_reg":    FINAL_W_REG,
            "final_w_cls":    FINAL_W_CLS,
        },
        "validation_metrics": validation_metrics or {},
    }
    joblib.dump(metadata, os.path.join(CACHE_DIR, "metadata.joblib"))
    log.info("Models saved        : %s", CACHE_DIR)


def load_models_from_cache(cache_dir: str) -> Optional[tuple[dict, dict]]:
    """
    Attempt to load all models from cache_dir.

    Returns (models_dict, metadata_dict) if all required files exist and the
    schema version matches, otherwise returns None to trigger retraining.
    """
    required_files = [
        "scaler.joblib", "ridge_model.joblib", "gb_model.joblib",
        "logistic_model.joblib", "bayes_model.joblib", "metadata.joblib",
    ]
    for fname in required_files:
        if not os.path.isfile(os.path.join(cache_dir, fname)):
            log.info("Cache miss          : %s not found — will retrain", fname)
            return None

    metadata = joblib.load(os.path.join(cache_dir, "metadata.joblib"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        log.warning(
            "Schema version mismatch: cached=%s  current=%s — retraining",
            metadata.get("schema_version"), SCHEMA_VERSION,
        )
        return None

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
        "Models loaded from cache  (schema=%s  as_of=%s)",
        metadata["schema_version"], metadata.get("as_of_date", "?"),
    )
    return models, metadata

# =============================================================================
# SUMMARY PRINT
# =============================================================================

def print_summary(
    models: dict,
    metrics: dict,
    home_win_rate: float,
    audit_status: str,
    X: np.ndarray,
    y_binary: np.ndarray,
    feature_names: list[str],
    from_cache: bool,
) -> None:
    """Print a human-readable training summary to stdout."""
    print()
    print("=" * 68)
    print("NCAAB MODEL TRAINING SUMMARY")
    print("=" * 68)
    print(f"  Schema version    : {SCHEMA_VERSION}")
    print(f"  As-of date        : {AS_OF_DATE}")
    print(f"  Source            : {'loaded from cache' if from_cache else 'freshly trained'}")
    print(f"  TensorFlow        : {'available' if TF_AVAILABLE else 'NOT available — NNs skipped'}")
    print(f"  Training rows     : {len(X):,}  ({len(X) // 2:,} games × 2 mirrored rows)")
    print(f"  Feature dims      : {len(feature_names)}")
    print(f"  Class balance     : {y_binary.mean():.1%} positive (home win)")
    print()

    print("  MODEL PERFORMANCE  (time-based holdout: 80% train / 20% test)")
    split_info = metrics.get("_split", {})
    if split_info:
        print(f"  Split date: {split_info.get('split_date','?')}  "
              f"train={split_info.get('n_train_games','?')} games  "
              f"test={split_info.get('n_test_games','?')} games")
    print(f"  {'Model':<24}  {'Type':<12}  {'Metric 1':<22}  {'Metric 2':<22}  {'Metric 3'}")
    print(f"  {'-'*24}  {'-'*12}  {'-'*22}  {'-'*22}  {'-'*16}")

    for name in ("lr_model", "sgd_model", "nn_reg_model"):
        if name not in metrics:
            continue
        m = metrics[name]
        print(f"  {name:<24}  {'Regressor':<12}  "
              f"test_MAE={m['test_mae']:.3f}             "
              f"test_RMSE={m['test_rmse']:.3f}")

    for name in ("logistic_model", "bayes_model", "nn_cls_model"):
        if name not in metrics:
            continue
        m = metrics[name]
        print(
            f"  {name:<24}  {'Classifier':<12}  "
            f"test_acc={m['test_acc']:.3f}              "
            f"Brier={m['test_brier']:.3f}                "
            f"AUC={m['test_auc']:.3f}"
        )

    print()
    print("  ENSEMBLE WEIGHTS")
    print(f"    Regression : {REG_W_RIDGE:.0%} Ridge  + {REG_W_GB:.0%} GB  + {REG_W_NN:.0%} NN Reg")
    print(f"    Classif.   : {CLS_W_LOGISTIC:.0%} Logist + {CLS_W_BAYES:.0%} Bayes + {CLS_W_NN:.0%} NN Cls")
    print(f"    Final      : {FINAL_W_REG:.0%} reg_implied_prob  +  {FINAL_W_CLS:.0%} cls_ensemble")
    print()
    print("  HOME BIAS AUDIT")
    print(f"    {audit_status}")
    print()
    print(f"  Cache directory   : {CACHE_DIR}")
    print("=" * 68)

# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    log.info("=" * 68)
    log.info("NCAAB Model Training")
    log.info("As-of: %s  |  Schema: %s", AS_OF_DATE, SCHEMA_VERSION)
    log.info("USE_CACHE=%s  FORCE_RETRAIN=%s  TF=%s", USE_CACHE, FORCE_RETRAIN, TF_AVAILABLE)
    log.info("=" * 68)

    # ── Step 1: Load training data ─────────────────────────────────────────────
    log.info("Step 1/5 — Loading training data")
    training_df             = load_training_df(TRAINING_CSV_PATH)
    lookup, numeric_cols    = build_team_lookup(training_df)

    # ── Step 2: Build matchup vectors ──────────────────────────────────────────
    log.info("Step 2/5 — Building matchup vectors (with mirroring)")
    X, y_diff, y_binary, feature_names, is_home_idx = build_training_arrays(
        training_df, lookup, numeric_cols,
    )

    # ── Step 3: Train or load models ───────────────────────────────────────────
    log.info("Step 3/5 — Training / loading models")
    from_cache = False

    needs_save = False
    if USE_CACHE and not FORCE_RETRAIN:
        cached = load_models_from_cache(CACHE_DIR)
        if cached is not None:
            models, metadata = cached
            # Validate cached feature dimension against current data
            if metadata["n_features"] != len(feature_names):
                log.warning(
                    "Feature dim mismatch: cached=%d  current=%d — retraining",
                    metadata["n_features"], len(feature_names),
                )
                models = train_models(X, y_diff, y_binary, is_home_idx)
                needs_save = True
            else:
                from_cache = True
        else:
            models = train_models(X, y_diff, y_binary, is_home_idx)
            needs_save = True
    else:
        models = train_models(X, y_diff, y_binary, is_home_idx)
        if USE_CACHE:
            needs_save = True

    # ── Step 4: Evaluate model performance ────────────────────────────────────
    log.info("Step 4/5 — Evaluating model performance (time-based 80/20 holdout)")
    metrics = evaluate_models(models, X, y_diff, y_binary, is_home_idx, training_df)

    # Save models after evaluation so validation_metrics are included in metadata
    if needs_save:
        save_models(models, feature_names, numeric_cols, is_home_idx, validation_metrics=metrics)

    # ── Step 5: Home bias audit ────────────────────────────────────────────────
    log.info("Step 5/5 — Running home bias audit")
    home_win_rate, audit_status = home_bias_audit(models, X, y_binary, is_home_idx)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary(
        models, metrics, home_win_rate, audit_status,
        X, y_binary, feature_names, from_cache,
    )

    # ── Audit gate (business rule §5 / §6 of Requirements) ────────────────────
    if home_win_rate > HOME_BIAS_RED:
        log.error(
            "Home bias audit FAILED (%.1f%% > %.0f%%). "
            "Adjust hyperparameters and set FORCE_RETRAIN = True before running predictions.",
            home_win_rate * 100, HOME_BIAS_RED * 100,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
