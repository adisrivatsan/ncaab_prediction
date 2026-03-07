# CLAUDE.md — NCAAB Prediction System Context

---

## 1. What This Project Does

This system produces daily betting recommendations for NCAA Men's Division I Basketball. It scrapes CBS Sports for completed game scores (training data) and Google News RSS + ESPN for team-level news (features), then trains a 6-model ensemble to predict score differentials and win probabilities. Each run outputs a win probability, predicted spread, and confidence tier (HIGH/MED/LOW) for every game. Predictions are displayed on a Vercel-hosted static website (`website/index.html`) that reads a JSON file (`predictions_latest.json`) generated daily by a GitHub Actions cron. Scope is strictly Men's D-I NCAAB; women's basketball is filtered out at every data collection step.

---

## 2. Repository Layout

```
NCAAB Prediction/
├── Training Data NCAAB (1).ipynb        ← Data collection (IMPLEMENTED — notebook)
├── Copy of ncaab_predictor.ipynb        ← Model training + predictions (IMPLEMENTED — notebook)
├── Python scripts/                      ← Standalone scripts (COMPLETE pipeline)
│   ├── cbs_scraper.py                   ← CBS game scraper (COMPLETE ✅)
│   ├── sentiment_features.py            ← News sentiment feature builder (COMPLETE ✅)
│   ├── efficiency_metrics.py            ← Efficiency metrics collector (COMPLETE ✅)
│   ├── kenpom_ratings.py                ← KenPom-style ratings via Bart Torvik (COMPLETE ✅)
│   ├── feature_assembly.py              ← Joins all sources → training_df.csv (COMPLETE ✅)
│   ├── model_training.py                ← Trains 6 models + home bias audit (COMPLETE ✅)
│   ├── predict_today.py                 ← Fetches CBS + predicts today's games (COMPLETE ✅)
│   ├── test_feature_assembly.py         ← Test harness for feature assembly (COMPLETE ✅)
│   ├── cbs_games.csv                    ← Output: 709 games, Feb 16–Mar 2 2026
│   ├── sentiment_features.csv           ← Output: 365 teams × 31 features
│   ├── efficiency_metrics.csv           ← Output: 365 teams × 4 efficiency metrics
│   ├── kenpom_ratings.csv               ← Output: 365 teams × 4 T-Rank ratings (regenerated Mar 4 ✅)
│   ├── training_df.csv                  ← Output: 709 rows × 90 cols (all sources joined)
│   ├── predictions_latest.json          ← Output: daily predictions export (✅ --export-json flag added)
│   ├── team_name_mapping.csv            ← Efficiency source→CBS name map
│   ├── kenpom_name_mapping.csv          ← T-Rank source→CBS name map
│   └── model_cache/                     ← Trained model files (8 files, schema v1.0)
│       ├── scaler.joblib
│       ├── ridge_model.joblib
│       ├── gb_model.joblib
│       ├── logistic_model.joblib
│       ├── bayes_model.joblib           ← GaussianNB (NEW — standalone only)
│       ├── nn_regressor.keras
│       ├── nn_classifier.keras
│       └── metadata.joblib              ← feature_names, numeric_cols, ensemble_weights
├── website/                             ← Vercel static frontend
│   ├── index.html                       ← Dashboard layout (Option C) ✅ COMPLETE (with legend, optimizer, pick analysis)
│   ├── predictions_latest.json          ← Real predictions data (written by predict_today.py --export-json ✅)
│   └── vercel.json                      ← Vercel config ✅ COMPLETE
├── .github/
│   └── workflows/
│       └── daily_predictions.yml        ← GitHub Actions cron ✅ COMPLETE (runs 3 PM EST daily)
├── Agents/
│   ├── CBS_games.md                     ← CBS scraper agent spec (IMPLEMENTED)
│   ├── sentiment.md                     ← News sentiment agent spec (IMPLEMENTED)
│   ├── KenPomRatings.md                 ← KenPom harvester spec (SPEC ONLY — not in notebooks)
│   └── Efficiency.md                    ← Efficiency metrics spec (SPEC ONLY — not in notebooks)
├── NCAAB_Requirements.md                ← Full system requirements
└── CLAUDE.md                            ← This file
```

No `src/`, no `requirements.txt`, no test suite. Runnable code lives in both notebook cells and standalone Python scripts. Data and models persist on Google Drive (notebooks) or locally in `Python scripts/` (standalone scripts).

---

## 3. Architecture and Data Flow

### Stage 1 — CBS Game Scraping (`Training Data NCAAB (1).ipynb`, Cell 2)

Function: `get_games_for_date_cbs(game_date)`

- URL: `https://www.cbssports.com/college-basketball/scoreboard/FBS/{YYYYMMDD}/`
- Finds `div.single-score-card` elements; reads `data-abbrev` for `AWAY@HOME` identity
- **Key quirk:** static HTML renders both team rows with class `tiedGame`; row order is determined by `data-abbrev`, not by CSS class
- Only cards where `div.game-status` text contains "final" (case-insensitive) are kept
- Output: list of dicts → `day_games_df` with 12 columns (see §4.1 of Requirements)
- Data shape at boundary: ~50 games/day × 12 columns

### Stage 2 — Feature Extraction (Cell 3)

Function: `extract_team_features(team_name)`

Calls 4 fetch functions:
1. `fetch_google_news_articles()` — 20 Google News RSS items; full body fetched for top 3
2. `fetch_espn_news_articles()` — 15 ESPN search results
3. `fetch_espn_targeted()` (injury suffix) — 10 ESPN results
4. `fetch_espn_targeted()` (lineup suffix) — 10 ESPN results

Returns a dict of 27 named feature scalars (groups A–F). All values are floats; see §4.2 of Requirements for full list.

### Stage 3 — Main Loop with Team Cache (Cell 4)

Iterates over `DATE_RANGE`. For each date:
- Step A: calls `get_games_for_date_cbs()` → appends to `all_games_df`
- Step B: for each unique team name in that day's games, calls `extract_team_features()` if not already in `processed_teams`

`processed_teams`: `dict[team_name → feature_dict]` — in-memory cache across the date loop. Join key is the raw CBS team name string; no normalization applied.

### Stage 4 — Training DataFrame Assembly (Cell 5)

```python
master_games_df    # shape: (691, 11)  — all games across date range
master_features_df # shape: (365, 32)  — one row per unique team (31 features + team_name)
training_df        # shape: (691, 73)  — 11 game cols + 31 home_* + 31 away_*
```

Merge logic:
```python
home_feats = master_features_df.add_prefix('home_').rename(columns={'home_team_name': 'home_name'})
away_feats = master_features_df.add_prefix('away_').rename(columns={'away_team_name': 'away_name'})
training_df = master_games_df.merge(home_feats, on='home_name', how='left')
training_df = training_df.merge(away_feats, on='away_name', how='left')
```

Rows with null `home_team_won` are dropped before training.

### Stage 5 — Today's Games (`Copy of ncaab_predictor.ipynb`, Cell 3)

Function: `get_todays_games()` — same CBS parsing logic as Stage 1, but **no Final-only filter**. All scheduled games (including live and upcoming) appear so every game has a prediction.

### Stage 6 — Matchup Vector Construction (Cell 9)

Function: `build_matchup_vector(home_name, away_name, lookup, cols)`

```python
return np.concatenate([home_vec, away_vec, diff_vec, is_home])
# shape: 27 + 27 + 27 + 1 = 82 dimensions
```

- `is_home` = **0.0 during training** (mirrored rows eliminate home bias), **1.0 during prediction**
- `is_home_idx` = `feature_names.index("is_home")` — used to zero out during training

### Stage 7 — Model Training (Cell 10 / `model_training.py`)

**Notebook (5 models):** `X_train` shape: games × 82, targets: `y_train` (score_diff), `y_train_binary` (home_team_won)

**Standalone (6 models):** `X_train` shape: (games×2) × 118 (mirrored rows; 39 features × 3 + is_home)

```
lr_model        = Ridge(alpha=RIDGE_ALPHA)
sgd_model       = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, ...)
nn_reg_model    = keras [Dense(128,relu) → BatchNorm → Dropout → Dense(64,relu) → BatchNorm → Dropout → Dense(1)]
logistic_model  = LogisticRegression(penalty='elasticnet', C=LOGISTIC_C, class_weight='balanced')
bayes_model     = GaussianNB(var_smoothing=1e-9)                    ← standalone only
nn_cls_model    = keras [Dense(128,relu) → BatchNorm → Dropout → Dense(64,relu) → BatchNorm → Dropout → Dense(1,sigmoid)]
```

EarlyStopping on val_loss (patience=20) for both NNs. Notebook models saved to `/content/drive/MyDrive/ncaab_model_cache/`. Standalone models saved to `Python scripts/model_cache/`.

### Stage 8 — Prediction and Output (Cells 11–14)

Function: `predict_game(game, today_team_lookup, numeric_cols)`

Three output tables:
1. Score differential table (regression model outputs)
2. Win probability table (classification + final ensemble)
3. Confidence + moneyline summary

Two betting outputs:
1. Top 3 straight picks (sorted by confidence → EV → win prob)
2. 2- and 3-leg parlay combinations

---

## 4. Coding Conventions

**Variable naming — do not rename these without reading §9 (Limitations) first:**
- `lr_model` = Ridge regressor (not logistic, despite the name)
- `sgd_model` = GradientBoostingRegressor (not SGD — known misnomer, see Limitation 6)
- `nn_reg_model` = NN regressor
- `logistic_model` = LogisticRegression classifier
- `bayes_model` = GaussianNB Bayesian classifier (standalone only — not in notebooks)
- `nn_cls_model` = NN classifier
- `master_games_df`, `master_features_df`, `training_df` = canonical DataFrame names in training notebook
- `processed_teams` = the in-memory team feature cache
- `today_team_lookup` = team feature dict used at prediction time (notebook); standalone uses `lookup` dict loaded from CSVs

**Cell header format:**
```python
# =============================================================================
# CELL N — Description
# =============================================================================
```
All cells follow this pattern. Do not reformat or remove headers.

**Schema stability:** The feature set is frozen per pipeline:
- **Notebook:** 27-feature set. Adding any column requires updating both notebooks, deleting `/content/drive/MyDrive/ncaab_model_cache/`, and incrementing the schema version.
- **Standalone:** 39-feature set (31 sentiment + 4 efficiency + 4 kenpom). Adding any column requires deleting `Python scripts/model_cache/`, incrementing `SCHEMA_VERSION` in `model_training.py`, and retraining.

**Team name handling:** Team names flow directly from CBS HTML text to all join keys with no normalization. Do not add normalization logic without also building a full alias table. A partial normalization will silently break joins.

**Numeric type enforcement:** All feature columns must pass through:
```python
pd.to_numeric(df[col], errors='coerce').fillna(0.0)
```
before training or vector construction. This is the only place where nulls are filled with 0.

**Error handling philosophy:** Prefer explicit failure with a logged message over silent fallback. The one sanctioned silent fallback is zeroing out the feature vector for a team that throws an exception in `extract_team_features()` — this keeps the team in the matrix at neutral values rather than dropping the game entirely.

---

## 5. Business Rules for Agents

These are hard constraints; do not soften them for convenience:

1. **"Final means final."** The status filter (`'final' in status_raw`) in `get_games_for_date_cbs()` must never be relaxed or bypassed. Partial/live games produce incorrect training labels.

2. **"One row per game."** The natural key is `data-abbrev`. No deduplication heuristic is permitted. If two cards share the same abbrev, log it and skip the duplicate.

3. **KenPom must use user-provided CSV.** Never attempt automated login, redirect-following past a login page, or CAPTCHA solving. If KenPom data is needed, output the CSV schema template and ingestion instructions and wait.

4. **Women's filter in all article fetch functions.** `is_womens_article(title, full_text)` must be called on every article before it enters any feature computation. This applies to Google News, ESPN general, ESPN injury, and ESPN lineup fetches.

5. **Home bias audit must pass before predictions run.** If the audit shows > 72% predicted home win rate, stop and instruct the user to increase regularization hyperparameters (`RIDGE_ALPHA`, `GB_MIN_SAMPLES`, `NN_L2`, `NN_DROPOUT`) and retrain.

6. **`is_home` zeroing must never be removed.** `X_train[:, is_home_idx] = 0.0` before fitting any model. The flag is legitimate only at prediction time.

---

## 6. Current Limitations (Quick Reference)

| # | Limitation | Pipeline | Triage |
|---|---|---|---|
| 1 | 14-day training window — poor generalization | Both | Expand to full season |
| 2 | KenPom / Efficiency not in notebooks — 8 features missing from notebook models | Notebook only | Port to notebooks; bump schema version; clear notebook cache |
| 3 | bayes_model (GaussianNB) not in notebook — ensemble diverges between pipelines | Notebook only | Port bayes_model to Cell 10; update classification ensemble weights |
| 4 | No out-of-sample validation — CV folds share teams | Both | Use time-based splits |
| 5 | Manual Vegas moneylines — spread analysis returns N/A | Both | Integrate odds API |
| 6 | `sgd_model` is actually GradientBoosting — confusing name | Both | Rename; clear cache |
| 7 | No women's filter in training notebook — possible label corruption | Notebook only | Port filter to training Cell 3 |
| 8 | Team name brittleness — no alias table | Both | Build alias table; add normalization step |
| 9 | kenpom_ratings.csv stale — adj_em/adj_d swap fix not reflected in CSV | Standalone | Re-run `kenpom_ratings.py` to regenerate |

---

## 7. How to Run

### A. Collect Training Data (`Training Data NCAAB (1).ipynb`)

1. Set `START_DATE` and `END_DATE` in Cell 1
2. Run Cell 1 (installs + imports)
3. Run Cell 4 (main loop — scrapes games + fetches features for all teams)
   - `processed_teams` cache avoids re-fetching teams seen on earlier dates
4. Run Cell 5 (assembles `master_games_df`, `master_features_df`, `training_df`)
5. Run Cell 6 (saves CSVs to Google Drive)

Re-scrape games without re-fetching features: run Cell 4b instead of Cell 4.

### B. Run Daily Predictions (`Copy of ncaab_predictor.ipynb`)

1. Ensure training CSVs exist on Drive from step A
2. Run Cells 1–3 (setup + CBS today's games scraper)
3. Run Cell 4 (women's filter + keyword dictionaries)
4. Run Cells 5–7 (feature extractor + build feature matrix for today's teams)
5. Run Cells 8–9 (load training data + build matchup vectors)
6. Run Cell 10 (train models or load from cache)
7. Run Cell 10B (model performance summary + overfitting check)
8. Run Cell 11 (prediction functions)
9. Run Cell 12 (run predictions for today's games)
10. Run Cell 13 (home bias audit — must pass before reviewing picks)
11. Run Cell 14 (betting optimizer — top 3 picks + parlays)

### C. Run Standalone Pipeline (local Python 3.9)

```bash
# One-time setup
pip install feedparser vaderSentiment joblib scikit-learn numpy pandas tensorflow tabulate requests beautifulsoup4 lxml

# Step 1: Scrape training games
python "Python scripts/cbs_scraper.py"

# Step 2: Collect features (run independently; order doesn't matter)
python "Python scripts/sentiment_features.py"
python "Python scripts/efficiency_metrics.py"
python "Python scripts/kenpom_ratings.py"

# Step 3: Assemble training_df.csv
python "Python scripts/feature_assembly.py"

# Step 4: Train all 6 models (runs home bias audit; exits with code 1 if 🔴)
python "Python scripts/model_training.py"

# Step 5: Predict today's games
python "Python scripts/predict_today.py"
```

Re-train from scratch: set `FORCE_RETRAIN = True` in `model_training.py`, then re-run Step 4.

### D. Update Hyperparameters After Audit Failure

**Notebook:** If Cell 13 shows 🔴 (> 72% home win rate):
1. In Cell 10, increase: `RIDGE_ALPHA`, `NN_L2`, `NN_DROPOUT`; decrease `GB_MAX_DEPTH`, `GB_SUBSAMPLE`
2. Set `FORCE_RETRAIN = True`
3. Re-run Cell 10 and Cell 13 to confirm audit passes

**Standalone:** If `model_training.py` exits with code 1 (🔴 audit):
1. In `model_training.py`, increase: `RIDGE_ALPHA`, `GB_MIN_SAMPLES`, `NN_L2`, `NN_DROPOUT`
2. Set `FORCE_RETRAIN = True`
3. Re-run `model_training.py` and confirm 🟢 before running `predict_today.py`

---

## 8. Extending the System

### Adding KenPom Features

1. Obtain `kenpom_export.csv` from kenpom.com (manual download required — no scraping)
2. Load and map team names to CBS team name strings (requires alias table)
3. Join to `master_features_df` on the mapped team name
4. Add KenPom columns to `extract_team_features()` return dict (or as a separate merge step)
5. Bump schema version; delete model cache; retrain both notebooks

Validate: `adj_em` range is typically −30 to +40; `adj_o` and `adj_d` are typically 80–130.

### Adding Efficiency Metrics (eFG%, TOV%, ORB%, FTR)

1. Implement data collection for the four metrics using documented formulas (see Requirements §3.3)
2. Validate ranges: all four metrics should be in [0, 1] after formula application
3. Add 4 columns to the team feature schema; update `extract_team_features()` return dict
4. Bump schema version; delete model cache
5. Retrain both notebooks and confirm matchup vector dimension updates from 82 to 82 + 4×3 = 94

---

## 9. Standalone Python Scripts — Implementation Progress

This section tracks the status of the standalone Python scripts in `Python scripts/`. These are independent of the notebooks and form a complete local execution pipeline.

### All Scripts / Components Status (Mar 7 2026)

| Script / Component | Status | Output | Notes |
|---|---|---|---|
| `cbs_scraper.py` | ✅ **COMPLETE** | `cbs_games.csv` at project root | Dynamic 3-week window ending yesterday (updated Mar 7) |
| `sentiment_features.py` | ✅ **COMPLETE** | `sentiment_features.csv` (365 teams × 31 features) | Hardcoded paths fixed Mar 7; uses `__file__`-based paths |
| `efficiency_metrics.py` | ✅ **COMPLETE** | `efficiency_metrics.csv` (365 rows × 4 metrics) | Hardcoded paths fixed Mar 7; uses `__file__`-based paths |
| `kenpom_ratings.py` | ✅ **COMPLETE** | `kenpom_ratings.csv` (365 rows) | adj_em/adj_d bug fixed + CSV regenerated Mar 4 ✅ |
| `feature_assembly.py` | ✅ **COMPLETE** | `training_df.csv` (709 rows × 90 cols) | 0 NaN; 61.1% home win rate |
| `test_feature_assembly.py` | ✅ **COMPLETE** | console output | Test harness for feature shapes and null checks |
| `model_training.py` | ✅ **COMPLETE** | `model_cache/` (8 files) | 6 models; X=(1418,118); home bias 🟢 57.1% |
| `predict_today.py` | ✅ **COMPLETE** | console (4 tables) + `predictions_latest.json` | `--export-json` flag; enriches JSON with feature_drivers, articles, top_picks |
| `website/index.html` | ✅ **COMPLETE** | Deployed on Vercel | Confidence legend; Betting Optimizer; Pick Analysis with feature drivers + article links |
| `website/predictions_latest.json` | ✅ **AUTO-UPDATED** | Daily via GitHub Actions | Written by workflow cron; Vercel auto-deploys on push |
| `website/vercel.json` | ✅ **COMPLETE** | Vercel deploy config | Cache-Control + SPA rewrite (added Mar 7) |
| `.github/workflows/daily_predictions.yml` | ✅ **COMPLETE** | Daily cron at 3 PM EST | Full pipeline: scrape → sentiment → efficiency → kenpom → assemble → train → predict → commit (added Mar 7) |

### Known Bugs

| Script | Bug | Status |
|---|---|---|
| `kenpom_ratings.py` | adj_em/adj_d columns swapped in existing CSV | **FIXED + CSV regenerated Mar 4** ✅ |
| `efficiency_metrics.py` | Barttorvik JSON W-L parse fails (`int("19-12")` raises ValueError) | **Non-blocking** — Sports Reference fallback delivers correct data |
| `feature_assembly.py` | Fuzzy name matching creates duplicate cbs_name rows (25 efficiency, 36 kenpom) | **FIXED** — deduplication by averaging per cbs_name |
| `sentiment_features.py` | Hardcoded absolute path `/Users/adityasrivatsan/...` broke GitHub Actions runner | **FIXED Mar 7** — replaced with `__file__`-based paths ✅ |
| `efficiency_metrics.py` | Hardcoded absolute path `/Users/adityasrivatsan/...` broke GitHub Actions runner | **FIXED Mar 7** — replaced with `__file__`-based paths ✅ |

### Verified Dimensions (Mar 3 2026)

```
feature_assembly.py output:
  training_df        : (709, 90)   — 12 game cols + 78 feature cols
  master_features_df : (365, 40)   — 365 teams × 39 features + team_name

model_training.py matchup vectors:
  X                  : (1418, 118) — 709 games × 2 mirrored rows × (39×3+1)
  y_diff             : (1418,)     — score differential
  y_binary           : (1418,)     — home_team_won
  is_home_idx        : 117
  Home bias audit    : 🟢 57.1%

predict_today.py (Mar 4 2026):
  CBS cards scraped  : 40
  Teams in lookup    : 80/80 (100% coverage)
  HIGH conf picks    : 0
  MED conf picks     : 9
  top_picks          : 3 (E. Kentucky HOME, Purdue AWAY, Wisconsin HOME)
  feature_drivers    : computed for all 9 MED picks (top 5 per pick, Ridge+GB+Logistic)
  articles           : fetched for 6 top-pick teams via Google News RSS
```

### Key Differences: Scripts vs Notebooks

| Aspect | Notebooks | Standalone Scripts |
|---|---|---|
| Runtime | Google Colab | Local Python 3.9 |
| Model cache | `/content/drive/MyDrive/ncaab_model_cache/` | `Python scripts/model_cache/` |
| Data output | Google Drive CSVs | `Python scripts/` directory; `cbs_games.csv` at project root |
| Models | 5 (no bayes_model) | 6 (adds GaussianNB `bayes_model`) |
| Features per team | 27 (sentiment only, notebook Cell 3) | 39 (sentiment + efficiency + kenpom) |
| Matchup vector dim | 82 (27×3+1) | 118 (39×3+1) |
| Classification ensemble | 40% Logistic + 60% NN Cls | 35% Logistic + 20% Bayes + 45% NN Cls |
| Python version | 3.x (Colab default) | 3.9 (local) — uses `from __future__ import annotations` |
| KenPom data | Not integrated | `kenpom_ratings.py` uses Bart Torvik (free, no login) |
| Required packages | Managed by Colab | `pip install feedparser vaderSentiment joblib scikit-learn tensorflow tabulate` |
