# Architecture

**Analysis Date:** 2026-03-17

## Pattern Overview

**Overall:** Multi-stage data pipeline with model ensemble prediction system

**Key Characteristics:**
- Linear data transformation pipeline: raw HTML → scraped games → features → training matrix → models → predictions
- 6-model ensemble with mirrored training rows to eliminate home-court bias
- Dual implementation: Jupyter notebooks (Colab with Google Drive) and standalone Python scripts (local/CI/CD)
- Two-phase prediction: on-demand daily predictions and March Madness bracket generation
- Static website frontend (Vercel) reading JSON predictions from GitHub repo
- Scheduled daily training every other day; predictions run every day

## Layers

**Scraping Layer:**
- Purpose: Collect raw game data and team news from external sources
- Location: `Python scripts/cbs_scraper.py`, `Python scripts/sentiment_features.py`, `Python scripts/efficiency_metrics.py`, `Python scripts/kenpom_ratings.py`, `Python scripts/odds_features.py`
- Contains: CBS Sports HTML parsing, news API fetches (Google News RSS, ESPN), financial data collection (The Odds API), Bart Torvik ratings parsing
- Depends on: External HTTP sources (CBS Sports, Google News, ESPN, Bart Torvik, The Odds API)
- Used by: Feature assembly layer

**Feature Engineering Layer:**
- Purpose: Transform raw scraped data into normalized numeric feature vectors
- Location: `Python scripts/feature_assembly.py`, `Python scripts/sentiment_features.py` (sentiment computation)
- Contains: VADER sentiment analysis, numeric scaling, team name mapping/deduplication, null-fill logic (0.0), cross-source feature joining
- Depends on: Scraping layer outputs (CSVs); mapping tables (`team_name_mapping.csv`, `kenpom_name_mapping.csv`, `odds_name_mapping.csv`)
- Used by: Training layer

**Training Layer:**
- Purpose: Build mirrored training matrix and train 6-model ensemble
- Location: `Python scripts/model_training.py`
- Contains: Home/away feature prefixing, game-to-matchup-vector conversion, model instantiation (Ridge, XGBoost/GradientBoosting, LogisticRegression, GaussianNB, 2× Neural Networks), home-bias audit
- Depends on: Feature engineering layer outputs (`training_df.csv`); hyperparameter configuration
- Used by: Prediction layer

**Prediction Layer:**
- Purpose: Generate daily match predictions and confidence assessments
- Location: `Python scripts/predict_today.py`, `Python scripts/march_madness_bracket.py`
- Contains: Today's CBS game scraper, feature vector construction (is_home=1.0), ensemble inference, confidence tier assignment (HIGH/MED/LOW), pick optimization, feature attribution
- Depends on: Training layer (models in cache); Feature engineering CSVs; CBS Sports scraper
- Used by: Website frontend, GitHub Actions workflow

**Frontend Layer:**
- Purpose: Display predictions and betting analysis via interactive dashboard
- Location: `website/index.html`, `website/predictions_latest.json`, `website/vercel.json`
- Contains: HTML/CSS/JS dashboard (stat cards, confidence legend, picks table, optimizer, per-pick feature drivers, article feeds, collapsible tables)
- Depends on: Prediction layer JSON output; Vercel for hosting and SPA routing
- Used by: End users

## Data Flow

**Training Pipeline (Every Other Day — Odd Day-of-Month):**

1. Scraper (`cbs_scraper.py`): CBS Sports → `cbs_games.csv` (709 final games, 3-week rolling window)
2. Sentiment (`sentiment_features.py`): Google News RSS + ESPN → `sentiment_features.csv` (365 teams × 31 features)
3. Efficiency (`efficiency_metrics.py`): Sports Reference → `efficiency_metrics.csv` (365 teams × 4 metrics)
4. KenPom (`kenpom_ratings.py`): Bart Torvik → `kenpom_ratings.csv` (365 teams × 4 ratings)
5. Assembly (`feature_assembly.py`): Join all sources on CBS team name → `training_df.csv` (709 games × 90 cols: 12 game + 39 home + 39 away)
6. Training (`model_training.py`): Load `training_df.csv`, build mirrored rows (1418 total), train 6 models, home-bias audit → `model_cache/` (8 files)

**Prediction Pipeline (Daily):**

1. CBS Scraper (`predict_today.py`): CBS Sports → today's 40±5 games (all statuses, no Final filter)
2. Feature Lookup: Load pre-computed CSVs (sentiment, efficiency, kenpom) → per-team feature dict
3. Odds Fetch (`odds_features.py`): The Odds API (optional) → `odds_snapshot.json`, moneylines, spreads, line-move tracking
4. Matchup Vectors: For each game, build 118-dim vector (39×3 features + is_home flag)
5. Ensemble Inference: Pass vectors through all 6 models
6. Aggregation: Compute weighted ensemble probabilities (55% regression + 45% classification)
7. Confidence Assignment: Label as HIGH/MED/LOW based on probability variance and margin
8. Optimization: Identify top 3 straight picks and 2/3-leg parlay combinations
9. Feature Attribution: For each MED/HIGH pick, compute top 5 drivers (Ridge + GB + Logistic avg)
10. Article Fetch: Google News RSS for top-pick teams
11. Export: Write `predictions_latest.json` to `website/predictions_latest.json` and `Python scripts/predictions_latest.json`

**State Management:**

- **Model Cache:** Persistent joblib/Keras files in `Python scripts/model_cache/` + GitHub Actions cache (on every-other-day runs)
- **Team Features:** Static CSVs regenerated every other day (sentiment, efficiency, kenpom)
- **Odds Snapshot:** `odds_snapshot.json` updated after each prediction run for line-move comparison
- **GitHub Integration:** Predictions committed to repo daily; Vercel auto-deploys on push

## Key Abstractions

**Matchup Vector:**
- Purpose: Unified representation of a game for inference
- Examples: `Python scripts/model_training.py` (build_matchup_vector), `Python scripts/predict_today.py` (vector construction)
- Pattern: home_features(39) + away_features(39) + diff_features(39) + is_home_flag(1) = 118 dimensions
  - Diff features capture home − away for key sentiment indicators
  - is_home is zeroed during training (mirrored rows), set to 1.0 during prediction

**Team Feature Dict:**
- Purpose: Per-team scalar values indexed by feature name
- Examples: Reconstructed from `training_df.csv` in `model_training.py` (build_team_lookup) or loaded from CSVs in `predict_today.py`
- Pattern: `{team_name → {'news_sentiment': 0.45, 'injury_count': 2, 'adj_em': 15.2, ...}}`

**Ensemble Weights:**
- Purpose: Aggregate predictions from 6 independent models into final probability
- Pattern:
  - Regression ensemble (score differential): 30% Ridge + 30% GB + 40% NN → implies home win probability via logistic
  - Classification ensemble (binary home win): 35% Logistic + 20% Bayes + 45% NN Cls
  - Final: 55% regression-implied + 45% classification-ensemble

**Confidence Tier:**
- Purpose: Assess reliability of prediction across ensemble models
- Pattern:
  - HIGH: prob_std < 0.08 AND |final_prob − 0.5| > 0.15
  - MED: prob_std < 0.15 AND |final_prob − 0.5| > 0.08
  - LOW: else

## Entry Points

**GitHub Actions Workflow (`daily_predictions.yml`):**
- Location: `.github/workflows/daily_predictions.yml`
- Triggers: Mon-Fri 3pm EDT, Sat-Sun 10am EDT (UTC cron converted for daylight savings offset)
- Responsibilities:
  - Conditional training (every other day based on day-of-month parity)
  - Step 1: CBS scraper → `cbs_games.csv`
  - Steps 2-5: Feature collectors (sentiment, efficiency, kenpom, assembly) → CSVs
  - Step 6: Model training → `model_cache/`
  - Step 7: Predictions + JSON export → `website/predictions_latest.json`
  - Step 8: March Madness bracket export
  - Step 9: Git commit and push to trigger Vercel deploy

**Local CLI (`predict_today.py`):**
- Location: `Python scripts/predict_today.py`
- Invocation: `python "Python scripts/predict_today.py" --export-json`
- Responsibilities: Same as workflow Step 7 (predictions + JSON export)

**Notebook Entry Points:**
- Training notebook (`Training Data NCAAB (1).ipynb`, Cells 1-6): Scrape games and extract features to Google Drive
- Predictor notebook (`Copy of ncaab_predictor.ipynb`, Cells 1-14): Load training data, train models, predict, optimize picks

**Bracket Generator:**
- Location: `Python scripts/march_madness_bracket.py`
- Invocation: `PYTHONUTF8=1 python "Python scripts/march_madness_bracket.py" --export-json --skip-refresh`
- Responsibilities: Generate March Madness tournament bracket predictions using trained ensemble

## Error Handling

**Strategy:** Explicit failure with logging over silent fallback. One sanctioned fallback: zeroing feature vectors for teams that throw exceptions in feature extraction, keeping the team in the matrix at neutral values.

**Patterns:**

**HTTP Failures (Scraping):**
- Retry logic with exponential backoff ([2s, 5s, 10s]) for CBS, ESPN, Google News fetches
- Log warnings per failed attempt; return empty list if all retries exhausted
- Training scraper (`cbs_scraper.py`) raises `FileNotFoundError` if data not found pre-prediction
- Example: `Python scripts/cbs_scraper.py` lines 43-61 (_fetch_page)

**Missing Data (Feature Assembly):**
- Numeric coercion with 0.0 fill for all feature columns: `pd.to_numeric(df[col], errors='coerce').fillna(0.0)`
- Applied in all loaders: `feature_assembly.py` (lines 105), `predict_today.py` feature building
- Teams missing from mapping → logged with warning, dropped from result set

**Model Training Failures:**
- Home-bias audit: if > 72% home wins predicted, exit with code 1 (🔴 — user must increase regularization)
- Graceful TensorFlow/XGBoost fallback: if TensorFlow unavailable, NN models skipped; if XGBoost unavailable, GradientBoosting used instead
- Example: `model_training.py` lines 63-77 (optional imports)

**Prediction-Time Failures:**
- Missing models in cache → raise `FileNotFoundError` (user must run training)
- Missing team in feature lookup → zero-fill 39-dim vector (neutral assessment)
- CBS scraper returns empty list → 0 predictions for the day (logged)

## Cross-Cutting Concerns

**Logging:** Python `logging` module with ISO timestamps and level labels. All scripts configured in `basicConfig()` to stdout.

**Validation:** All feature columns normalized to float via `pd.to_numeric(..., errors='coerce').fillna(0.0)`. No nulls permitted in training/prediction matrices.

**Authentication:** No API authentication except optional The Odds API key (`ODDS_API_KEY` env var). If missing, predictions run without Vegas moneyline enrichment (graceful no-op).

**Schema Versioning:** `SCHEMA_VERSION` in `model_training.py` (currently "2.0"). Must be bumped and cache cleared when feature columns change. Standalone pipeline has 39 features per team; notebook pipeline has 27 (notebook missing efficiency + kenpom).

**Team Name Stability:** Team names flow directly from CBS HTML text with no normalization. Join keys are raw CBS strings. Mapping tables (`team_name_mapping.csv`, `kenpom_name_mapping.csv`) resolve external source names to CBS names.

**Hyperparameter Control:** All regularization, learning rates, ensemble weights configured at module level in `model_training.py` (lines 94-147). Home-bias thresholds defined for audit pass/caution/fail (lines 129-132).

---

*Architecture analysis: 2026-03-17*
