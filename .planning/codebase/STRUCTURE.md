# Codebase Structure

**Analysis Date:** 2026-03-17

## Directory Layout

```
NCAAB Prediction/
├── Training Data NCAAB (1).ipynb              # Notebook: game scraping + feature extraction (Colab)
├── Copy of ncaab_predictor.ipynb              # Notebook: model training + daily predictions (Colab)
├── Python scripts/                             # Standalone data pipeline (local + CI/CD)
│   ├── cbs_scraper.py                         # CBS Sports final games scraper
│   ├── sentiment_features.py                  # News sentiment feature extractor
│   ├── efficiency_metrics.py                  # Team efficiency metrics (Sports Ref)
│   ├── kenpom_ratings.py                      # Bart Torvik T-Rank ratings
│   ├── odds_features.py                       # The Odds API moneyline/spread fetch
│   ├── feature_assembly.py                    # Join all sources → training_df.csv
│   ├── model_training.py                      # Train 6-model ensemble + home-bias audit
│   ├── predict_today.py                       # Daily predictions + JSON export
│   ├── march_madness_bracket.py               # March Madness bracket predictions
│   ├── test_feature_assembly.py               # Test harness (feature shape validation)
│   ├── model_cache/                           # Persistent trained model files (8 files)
│   │   ├── scaler.joblib                      # StandardScaler for feature normalization
│   │   ├── ridge_model.joblib                 # Ridge regressor
│   │   ├── gb_model.joblib                    # XGBoost or GradientBoosting regressor
│   │   ├── logistic_model.joblib              # LogisticRegression classifier
│   │   ├── bayes_model.joblib                 # GaussianNB classifier
│   │   ├── nn_regressor.keras                 # Keras NN regressor
│   │   ├── nn_classifier.keras                # Keras NN classifier
│   │   └── metadata.joblib                    # Feature names, numeric cols, ensemble weights
│   ├── cbs_games.csv                          # Scraper output: ~700 final games
│   ├── sentiment_features.csv                 # Sentiment output: 365 teams × 31 features
│   ├── efficiency_metrics.csv                 # Efficiency output: 365 teams × 4 metrics
│   ├── kenpom_ratings.csv                     # KenPom output: 365 teams × 4 ratings
│   ├── training_df.csv                        # Assembly output: 709 games × 90 columns
│   ├── team_name_mapping.csv                  # Sports Ref → CBS team name map
│   ├── kenpom_name_mapping.csv                # Bart Torvik → CBS team name map
│   ├── odds_name_mapping.csv                  # The Odds API → CBS team name map
│   ├── odds_snapshot.json                     # Previous-run odds cache (line-move tracking)
│   ├── predictions_latest.json                # Daily predictions JSON (backup copy)
│   └── bracket_predictions_2026.json          # March Madness bracket export
├── website/                                    # Vercel frontend (static)
│   ├── index.html                             # Interactive dashboard (440+ lines)
│   ├── predictions_latest.json                # Live predictions (read by index.html)
│   ├── bracket_predictions_2026.json          # Live bracket data
│   └── vercel.json                            # Vercel config (SPA routing, cache headers)
├── .github/
│   └── workflows/
│       └── daily_predictions.yml              # GitHub Actions cron workflow
├── Agents/                                     # Agent specification documents
│   ├── CBS_games.md                           # CBS scraper spec
│   ├── sentiment.md                           # Sentiment extractor spec
│   ├── KenPomRatings.md                       # KenPom harvester spec
│   └── Efficiency.md                          # Efficiency metrics spec
├── .planning/
│   └── codebase/                              # GSD codebase docs (generated)
├── CSV Tests/                                  # Validation outputs (legacy)
├── memory/
│   └── MEMORY.md                              # User session memory
├── CLAUDE.md                                  # Complete system documentation
├── NCAAB_Requirements.md                      # Full system requirements spec
├── purrfect-drifting-deer.md                  # Active plan (ML system revisions)
└── bracket_2026.md                            # March Madness bracket analysis
```

## Directory Purposes

**Python scripts/:**
- Purpose: Standalone Python data pipeline (independent of notebooks)
- Contains: Feature collectors, feature assembler, model trainer, predictor, bracket generator
- Key files: All .py scripts and generated CSVs/JSONs
- Committed to repo: Yes (.py scripts, name mappings)
- Generated (git-ignored): CSVs, model cache, odds snapshot, predictions JSON

**website/:**
- Purpose: Vercel-hosted static frontend for predictions dashboard
- Contains: HTML/CSS/JS dashboard, predictions data, Vercel config
- Key files: `index.html` (440+ lines), `predictions_latest.json`, `vercel.json`
- Committed to repo: Yes (all files)
- Deployed: Yes (Vercel auto-deploys on git push)

**.github/workflows/:**
- Purpose: CI/CD automation (daily prediction cron)
- Contains: YAML workflow definition
- Key files: `daily_predictions.yml` (9 steps, 127 lines)
- Committed to repo: Yes
- Triggered: Mon-Fri 3pm EDT, Sat-Sun 10am EDT (UTC cron)

**model_cache/:**
- Purpose: Persistent trained model storage (persists across runs)
- Contains: 8 joblib/Keras files (scaler, 6 models, metadata)
- Key files: `metadata.joblib` (feature names, ensemble weights — critical)
- Committed to repo: No (git-ignored, cached via GitHub Actions)
- Lifecycle: Overwritten every other day (training day)

## Key File Locations

**Entry Points:**

| File | Type | Purpose |
|------|------|---------|
| `Python scripts/cbs_scraper.py` | Script | Scrape CBS Sports final games (3-week rolling window) |
| `Python scripts/sentiment_features.py` | Script | Extract news sentiment features for all teams |
| `Python scripts/efficiency_metrics.py` | Script | Fetch team efficiency metrics from Sports Reference |
| `Python scripts/kenpom_ratings.py` | Script | Scrape Bart Torvik T-Rank ratings |
| `Python scripts/feature_assembly.py` | Script | Join all feature sources into training_df.csv |
| `Python scripts/model_training.py` | Script | Train 6-model ensemble and audit home bias |
| `Python scripts/predict_today.py` | Script | Generate daily predictions and export JSON |
| `Python scripts/march_madness_bracket.py` | Script | Generate March Madness bracket predictions |
| `.github/workflows/daily_predictions.yml` | Workflow | Orchestrate all 9 steps (training every other day) |
| `Training Data NCAAB (1).ipynb` | Notebook | Google Colab: Cells 1-6 (scrape games + extract features) |
| `Copy of ncaab_predictor.ipynb` | Notebook | Google Colab: Cells 1-14 (train models + predict) |
| `website/index.html` | Frontend | Interactive dashboard (reads predictions_latest.json) |

**Configuration:**

| File | Type | Purpose |
|------|------|---------|
| `model_training.py` | Python | Lines 82-147: Hyperparameters, ensemble weights, home-bias thresholds, schema version |
| `predict_today.py` | Python | Lines 70-108: Configuration (paths, API URLs, confidence thresholds) |
| `cbs_scraper.py` | Python | Lines 17-37: Scraper config (date window, timeout, retry backoff, headers) |
| `sentiment_features.py` | Python | Lines 25-100: Keywords (injury, lineup, win, loss, momentum, etc.), rate limits |
| `vercel.json` | JSON | Cache headers, SPA rewrite rule, error handlers |

**Data Persistence:**

| File | Source | Updated | Purpose |
|------|--------|---------|---------|
| `Python scripts/cbs_games.csv` | `cbs_scraper.py` | Every other day (training) | ~700 final games for training |
| `Python scripts/sentiment_features.csv` | `sentiment_features.py` | Every other day (training) | 365 teams × 31 features |
| `Python scripts/efficiency_metrics.csv` | `efficiency_metrics.py` | Every other day (training) | 365 teams × 4 metrics |
| `Python scripts/kenpom_ratings.csv` | `kenpom_ratings.py` | Every other day (training) | 365 teams × 4 ratings |
| `Python scripts/training_df.csv` | `feature_assembly.py` | Every other day (training) | 709 games × 90 cols (joined features) |
| `Python scripts/model_cache/` | `model_training.py` | Every other day (training) | 8 model files (scaler, models, metadata) |
| `Python scripts/odds_snapshot.json` | `predict_today.py` | Daily (prediction) | Previous moneylines for line-move tracking |
| `website/predictions_latest.json` | `predict_today.py` | Daily (prediction) | Live predictions (read by frontend) |

## Naming Conventions

**Files:**

| Pattern | Example | Usage |
|---------|---------|-------|
| Snake case `.py` | `cbs_scraper.py`, `sentiment_features.py` | All Python scripts |
| Descriptive names | `feature_assembly.py`, `model_training.py` | Scripts that do one main task |
| CSV output | `cbs_games.csv`, `training_df.csv` | Training data; naming matches variable names |
| Model cache | `ridge_model.joblib`, `nn_classifier.keras` | Model files; type indicates serialization format |
| Mapping CSVs | `team_name_mapping.csv`, `odds_name_mapping.csv` | Name resolution files (source → CBS name) |
| Notebook files | `Training Data NCAAB (1).ipynb` | Spaces permitted in Colab; numbers for versioning |

**Directories:**

| Pattern | Example | Usage |
|---------|---------|-------|
| Snake case | `python scripts`, `csv_tests`, `model_cache` | Directory names (except legacy "CSV Tests") |
| Plural for data | `Python scripts`, `.planning`, `Agents` | Multi-file collections |
| No underscores in hidden | `.github`, `.planning`, `.claude` | Dotfiles (standard conventions) |

## Where to Add New Code

**New Data Source (Feature):**
1. Create `Python scripts/{source}_features.py` following the pattern:
   - Top docstring: input/output CSV paths, output schema
   - `_SCRIPT_DIR` and `_PROJECT_DIR` path setup (lines 26-27 pattern)
   - Logging configuration
   - Core fetch/compute function
   - Main entry point calling fetch/compute, writing CSV with logging
2. Add mapping CSV if external team names differ from CBS (e.g., `{source}_name_mapping.csv`)
3. Update `feature_assembly.py` to load new CSV and join to `training_df`
4. Add script to `.github/workflows/daily_predictions.yml` as conditional step
5. Bump `SCHEMA_VERSION` in `model_training.py` and clear `model_cache/`

**New Model Type:**
1. Add model instantiation in `model_training.py` (Cell 2 section, lines 200+)
2. Add to training loop (Cell 4, around line 500+)
3. Update ensemble weights (lines 140-147) — ensure new weights sum to 1.0
4. Save model to `model_cache/{model_name}.joblib` or `.keras`
5. Update `metadata.joblib` with new weights
6. Update `predict_today.py` to load and infer with new model
7. Bump `SCHEMA_VERSION` and clear cache

**New Dashboard Feature (Frontend):**
1. Edit `website/index.html`
2. Add HTML structure in appropriate section (see comments: Header, Stat cards, Legend, Picks table, Optimizer, Pick Analysis, All Data)
3. Add CSS to `<style>` block (follow existing utility patterns: flex, grid, colors from palette)
4. Add JavaScript function in `<script>` block
5. Reference `data.predictions`, `data.summary`, `data.top_picks` from `predictions_latest.json`
6. Commit and push; Vercel auto-deploys

**New Prediction Output (JSON):**
1. In `predict_today.py`, add field to final dict before JSON export (around line 1200+)
2. Follow existing JSON schema structure (nested dicts, lists of dicts per prediction)
3. Use datetime ISO format for timestamps
4. Write to `website/predictions_latest.json` via existing export logic
5. Update `website/index.html` to consume new field

## Special Directories

**model_cache/:**
- Purpose: Persistent trained model storage
- Generated: Yes (by `model_training.py`)
- Committed: No (git-ignored; cached via GitHub Actions)
- Lifecycle: Recreated every other day (training day); persists across daily prediction runs

**Python scripts/CSV outputs:**
- Purpose: Training data and intermediate results
- Generated: Yes (by feature collectors and assembler)
- Committed: Partially
  - `.gitignore` excludes: `*.csv` (all generated CSVs)
  - `.gitignore` excludes: `*.json` (all generated predictions)
  - Mapping CSVs (`team_name_mapping.csv`, etc.) committed manually if stable
- Lifecycle: Regenerated every training run; cached via GitHub Actions

**website/predictions_latest.json:**
- Purpose: Live predictions data (read by frontend)
- Generated: Yes (by `predict_today.py --export-json`)
- Committed: Yes (pushed daily by workflow)
- Lifecycle: Updated daily; Vercel auto-deploys on push

**memory/:**
- Purpose: User session memory for GSD orchestrator
- Contains: `MEMORY.md` (persistent context across conversations)
- Generated: User-maintained (not auto-generated)
- Lifecycle: Manually updated between sessions

**.planning/codebase/:**
- Purpose: Generated codebase documentation (GSD mapping)
- Contains: `ARCHITECTURE.md`, `STRUCTURE.md`, `CONVENTIONS.md`, `TESTING.md`, `CONCERNS.md`, `STACK.md`, `INTEGRATIONS.md`
- Generated: Yes (by GSD agent on `/gsd:map-codebase`)
- Lifecycle: Regenerated when codebase structure changes

## Testing & Validation

**Test File:**
- Location: `Python scripts/test_feature_assembly.py`
- Purpose: Validate feature assembly pipeline (shape, nulls, data types)
- Run: `python "Python scripts/test_feature_assembly.py"`
- Coverage: Feature column counts, null checks, numeric type enforcement

**Home Bias Audit:**
- Location: `model_training.py` (lines 600+)
- Purpose: Verify models don't overpredict home wins (business rule §5.4)
- Threshold: > 72% → 🔴 fail (exit code 1); 62-72% → 🟡 caution; < 50% → 🟡 undervalues home
- Action on fail: Increase regularization (RIDGE_ALPHA, GB_MIN_SAMPLES, NN_L2, NN_DROPOUT) and retrain

**CSV Validation (Manual):**
- Check: `cbs_games.csv` has 709 rows and 12 columns
- Check: `sentiment_features.csv` has 365 rows and 32 columns (31 features + team_name)
- Check: `training_df.csv` has 0 NaN values and 90 columns
- Example validation: `feature_assembly.py` loads and enforces numeric types (lines 71-111)

---

*Structure analysis: 2026-03-17*
