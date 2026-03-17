# Codebase Concerns

**Analysis Date:** 2026-03-17

## Tech Debt

**21-day rolling training window — poor generalization:**
- Issue: Models train only on the most recent 21 days of games (~700 games). This window is too narrow for robust patterns; teams appearing 1-2 times may cause overfitting to schedule artifacts rather than true team strength.
- Files: `cbs_scraper.py` (line 20–21), `CLAUDE.md` (§6 Limitation 1)
- Impact: Predictions may regress significantly when the season changes phase (e.g., mid-season to tournament prep). Model performance is not stable across the full season.
- Fix approach: Expand training window to full regular season (Nov–Mar, ~1500+ games). Requires retraining all models; model cache schema must remain stable.

**No out-of-sample validation — CV folds share teams:**
- Issue: `model_training.py` uses a naive time-based 80/20 split (first 80% of games train, last 20% test). This violates the assumption that training and test folds are independent: teams appearing in both splits cause information leakage.
- Files: `model_training.py` (lines 581–591), NCAAB_Requirements.md (§6 Limitation 4)
- Impact: Validation metrics (`test_mae`, `test_rmse`, `test_acc`, `test_auc`) are artificially optimistic. True out-of-sample performance is likely 5–15% worse than reported.
- Fix approach: Implement time-based or team-stratified cross-validation. Ensure each team appears only in train OR test, not both. Requires refactoring `evaluate_models()`.

**Dual pipeline feature divergence (27 vs 39 features):**
- Issue: The notebook training pipeline uses 27 features (sentiment only); the standalone pipeline uses 39 features (sentiment + efficiency + kenpom). Models trained in each pipeline have different input dimensions (82 vs 118). If one pipeline is updated, the other becomes stale.
- Files: `Training Data NCAAB (1).ipynb` (Cell 3), `Copy of ncaab_predictor.ipynb` (Cells 5–7), `model_training.py` (line 14), CLAUDE.md (§9 Key Differences)
- Impact: Predictions from the notebook and standalone scripts diverge. Betting decisions made on one pipeline may not align with the other. If notebooks are run to generate new predictions, they will differ from GitHub Actions (which uses standalone).
- Fix approach: Port efficiency_metrics and kenpom_ratings to the notebooks. Bump notebook schema version; clear notebook model cache; retrain and validate. Consolidate to single 39-feature pipeline.

**Inconsistent model ensemble across pipelines:**
- Issue: Classification ensemble weights differ: Notebook uses 40% Logistic + 60% NN Cls; Standalone uses 35% Logistic + 20% Bayes + 45% NN Cls. The bayes_model (GaussianNB) is only in the standalone pipeline.
- Files: `model_training.py` (lines 140–143), CLAUDE.md (§9 Key Differences, Limitation 3)
- Impact: Two systems making predictions on the same matchup will produce different probabilities. The GaussianNB model in the standalone pipeline adds ~20% weight to a model that doesn't exist in notebooks.
- Fix approach: Port bayes_model to the training notebook (Cell 10). Update notebook classification ensemble to 35% Logistic + 20% Bayes + 45% NN Cls. Clear notebook model cache and retrain.

---

## Known Bugs

**Barttorvik JSON parser fails gracefully but logs no error:**
- Symptoms: `efficiency_metrics.py` attempts to parse W-L record from Bart Torvik JSON as integers, but Torvik returns W-L as a string `"19-12"`. The parser calls `int("19-12")`, which raises `ValueError`. Code catches the exception silently and falls back to Sports Reference.
- Files: `efficiency_metrics.py` (lines 150–170, approx — W-L parsing in try/except block)
- Trigger: Any run of `efficiency_metrics.py` when Torvik API is available. Happens on every daily run via GitHub Actions.
- Workaround: Sports Reference fallback delivers correct data, so no user-visible impact. However, silent failures mask the real issue.
- Fix approach: Parse W-L as `record.split("-")` instead of `int(record)`. Log the fallback decision explicitly.

**Duplicate fuzzy name matching creates averaged feature rows:**
- Symptoms: `efficiency_metrics.py` and `kenpom_ratings.py` perform fuzzy team name matching. Multiple source team names can match to the same CBS name (e.g., "Virginia" and "Virginia Commonwealth" both fuzzy-match to "Virginia" if the matcher is loose). This creates duplicate rows that are then deduplicated by averaging.
- Files: `feature_assembly.py` (lines 151–162 for efficiency, lines 190–201 for kenpom)
- Trigger: Any team with a short or generic name that appears in multiple source datasets. Affects ~25 efficiency rows and ~36 kenpom rows (fixed by averaging in current version).
- Workaround: Current code detects and fixes by averaging per cbs_name (already implemented in `load_efficiency()` and `load_kenpom()`). Warning logged.
- Fix approach: Improve fuzzy matching thresholds or use exact/alias table matching instead. Validate match quality before aggregation.

**CBS scraper HTML parsing brittle to page structure changes:**
- Symptoms: Scraper relies on exact CSS selectors (`div.single-score-card`, `div.game-status`). If CBS redesigns the scoreboard page, selectors will break silently.
- Files: `cbs_scraper.py` (lines 82–84), `get_todays_games()` in `predict_today.py` (similar logic)
- Trigger: Any CBS website redesign. Last redesign was early 2025; next is unpredictable.
- Workaround: None — scraper fails silently and returns empty game list.
- Fix approach: Add validation to confirm expected HTML structure. If selectors return 0 cards but the page loaded, log an error and exit with non-zero code. Monitor page structure monthly.

---

## Security Considerations

**Secrets potentially committed to git history:**
- Risk: `.env` file is in `.gitignore` but the GitHub Actions workflow injects `ODDS_API_KEY` as a secret. If the key were ever logged to stdout, it would be captured in GitHub Actions logs. Previous Python code may have logged full feature values or URLs with embedded auth.
- Files: `.github/workflows/daily_predictions.yml` (Step 7), `odds_features.py` (API calls at prediction time)
- Current mitigation: `ODDS_API_KEY` is never logged; it's only used in the request header. Logs are not persisted in the repo.
- Recommendations: (1) Add a pre-commit hook to scan for common secret patterns. (2) Rotate `ODDS_API_KEY` every 90 days. (3) Monitor GitHub Actions logs for accidental stdout leaks.

**No input validation on team names:**
- Risk: Team names flow directly from CBS HTML text (via BeautifulSoup) to DataFrames and join keys, with no sanitization. If a team name contains special characters or unicode, subsequent joins may fail silently.
- Files: `cbs_scraper.py` (lines 106–130, name extraction), `feature_assembly.py` (merge logic on raw team_name strings), `model_training.py` (build_team_lookup uses row["home_name"] without validation)
- Current mitigation: pandas `.merge()` is strict about types; unicode/encoding issues would surface as NaN joins.
- Recommendations: (1) Add explicit `.str.strip()` to team names before any join. (2) Add assertion that all game rows have non-null team names after scraping. (3) Log first 5 team names from each scrape to validate format.

**Network requests lack timeout uniformity:**
- Risk: `sentiment_features.py` uses `REQUEST_TIMEOUT = 11` seconds (line 32); `cbs_scraper.py` uses `REQUEST_TIMEOUT = 20` seconds (line 24). `predict_today.py` also uses 20. If a service hangs, the system may wait inconsistently.
- Files: `sentiment_features.py` (line 32), `cbs_scraper.py` (line 24), `predict_today.py` (line 86), `odds_features.py` (implicit via requests default)
- Current mitigation: None — inconsistent but all finite.
- Recommendations: (1) Define a single global `REQUEST_TIMEOUT` constant shared across all scripts. (2) Document why sentiment needs a longer timeout. (3) Add a backoff ceiling (max 30 seconds total wait before giving up).

---

## Performance Bottlenecks

**Sentiment feature extraction is the slowest pipeline step:**
- Problem: `sentiment_features.py` fetches articles from Google News + ESPN for 365 teams, with a 0.4-second rate-limit sleep between teams. This takes ~150 seconds (2.5 minutes) just for sleeps, plus network latency. For 365 teams × 4 fetch calls = 1,460 HTTP requests.
- Files: `sentiment_features.py` (lines 200–250, the main loop), line 31 (`RATE_LIMIT_SLEEP = 0.4`)
- Cause: Sequential fetching with no parallelization. Network I/O is not CPU-bound; async or threading would help.
- Improvement path: (1) Use `asyncio` or `ThreadPoolExecutor` to fetch 5–10 teams in parallel. (2) Cache articles in a local database (SQLite) so repeated runs don't re-fetch. (3) Reduce rate-limit sleep to 0.2 seconds if the APIs allow. Estimated speedup: 3–5x.

**Training DataFrame merge is O(n²) in worst case:**
- Problem: `feature_assembly.py` performs two sequential left-merges: games + home_features, then + away_features. If team name strings are not pre-sorted, pandas may use a slow merge algorithm.
- Files: `feature_assembly.py` (lines 286–287), `assemble_training_df()`
- Cause: No `.sort_values()` before merge; pandas defaults to hash-join if indices aren't sorted.
- Improvement path: Pre-sort both games_df and features_df by team_name before merge. Estimated speedup: 2x for 700 games × 365 teams. Low priority (merge is already <1 second).

**Model training NNs use early stopping with patience=20, but may train for full 300 epochs:**
- Problem: `model_training.py` sets `NN_MAX_EPOCHS = 300` and `NN_PATIENCE = 20` (line 126–127). If validation loss doesn't improve, the NN trains for 300 epochs. Early stopping patience of 20 means 20 epochs of no improvement before stopping. On a large dataset (1418 rows), this can take 2–3 minutes.
- Files: `model_training.py` (lines 126–127, 504–507)
- Cause: Hyperparameter choices. NN_MAX_EPOCHS is a hard ceiling; patience is only a soft target.
- Improvement path: (1) Reduce `NN_MAX_EPOCHS` to 150. (2) Increase `NN_PATIENCE` to 30 or 40 to allow more exploration but cap total epochs. (3) Monitor validation loss and adjust learning rate adaptively. Estimated speedup: 2x.

---

## Fragile Areas

**Team name brittle matching — no alias table:**
- Files: `cbs_scraper.py` (CBS team name extraction), `feature_assembly.py` (join on raw team_name strings), `odds_features.py` (6-step name matching with fallbacks)
- Why fragile: Team names must match exactly between CBS, sentiment features, efficiency metrics, and kenpom_ratings. If any source uses a variant name (e.g., "UConn" vs "Connecticut" vs "University of Connecticut"), the join breaks silently and features become 0.0. Already observed in fuzzy matching bugs (see Known Bugs).
- Safe modification: (1) Build a single canonical team name mapping table (CSV with columns: cbs_name, sentiment_name, efficiency_name, kenpom_name). (2) Load and apply this mapping in `feature_assembly.py` before any join. (3) Test that 365 teams map to 365 unique CBS names (no collisions).
- Test coverage: `test_feature_assembly.py` checks for nulls but not for zero-filled features due to mismatched names. Add a check: "if (df['home_sent_overall'] == 0.0).sum() > 5, warn('possible name mismatch')"

**Model cache invalidation via schema version mismatch:**
- Files: `model_training.py` (lines 819–824, schema version check in `load_models_from_cache()`)
- Why fragile: If a developer updates feature schema (e.g., adds a new sentiment feature) but forgets to increment `SCHEMA_VERSION` in `model_training.py`, the old cache is loaded silently with mismatched dimensions. This causes a crash in `predict_today.py` when building matchup vectors (line 312 in `model_training.py`: `np.concatenate()` will fail if shapes don't match).
- Safe modification: (1) Add a unit test that trains a model, saves it, then loads it and verifies shapes match. (2) In `predict_today.py`, add an explicit dimension check before using cached models: assert that `len(feature_names) == cached_feature_count`. (3) Document the schema version increment requirement in CLAUDE.md §4 Coding Conventions.
- Test coverage: None currently. Add a new test: `test_model_cache_schema_mismatch()`.

**GitHub Actions workflow conditional training may miss schema version updates:**
- Files: `.github/workflows/daily_predictions.yml` (Steps 1–6 are conditional on `is_training_day`)
- Why fragile: Training runs only every other day (odd day-of-month). If someone updates the feature schema on an even day, the workflow won't retrain. The next odd day will load the stale cache. Predictions run with mismatched feature dimensions.
- Safe modification: (1) Always run model_training.py (remove the conditional). It will load from cache if schema matches, train only if needed. (2) Or, add a cache invalidation trigger: if `feature_assembly.py` output size changes, force retraining in the next workflow. (3) Document this risk in the workflow file.
- Test coverage: No CI test for this. Add a GitHub Actions test: "verify that schema version is incremented when feature count changes".

---

## Scaling Limits

**API rate limits not monitored:**
- Current capacity: Google News RSS returns 20 items; ESPN API has no official rate limit but is likely ~60 req/min. The Odds API logs `x-requests-remaining` but doesn't enforce a ceiling in the code.
- Limit: If Sentiment extraction tries to fetch >100 teams/min, Google News or ESPN will return 429 Too Many Requests or empty results. Silent failure (empty article lists) leads to zero-filled feature vectors.
- Scaling path: (1) Add explicit rate-limit header parsing in `odds_features.py` (already done for logging; add logic to sleep if remaining quota is low). (2) Cache article fetches by team + date so repeated runs don't re-request. (3) Batch requests using asyncio instead of sequential fetches.

**Memory usage for large feature matrices:**
- Current capacity: training_df is (709, 90) = ~65 KB in CSV; 9 MB in memory as float64. Model cache is ~50 MB (TensorFlow NNs). Prediction on 40 games takes <1 second.
- Limit: If training window expands to full season (1500+ games) and feature count increases further, matchup vector X grows to (3000, 150+) = ~7.2 MB. NNs will need to re-initialize for larger input dimensions. TensorFlow model files may exceed GitHub Actions artifact size limits (5 GB per workflow).
- Scaling path: (1) Use float32 instead of float64 everywhere to cut memory by half. (2) Compress model cache with gzip (TensorFlow .keras files compress well). (3) Archive old model caches outside the repo (Google Drive or S3).

---

## Dependencies at Risk

**TensorFlow optional but critical for predictions:**
- Risk: Both NNs (nn_reg_model, nn_cls_model) are optional — code skips them gracefully if TensorFlow is not installed. However, the ensemble weights assume both NNs are present (40% + 60% in classification; 30% + 30% + 40% in regression). If NNs are missing, the ensemble redistributes weights naively (see `model_training.py` lines 714–716, 728–733), but re-weighting on the fly introduces bias.
- Impact: Predictions without TensorFlow will differ from predictions with it (may be 5–10% off). If GitHub Actions runner doesn't have TensorFlow, all predictions go out with degraded models.
- Migration plan: (1) Make TensorFlow a hard requirement (add to workflow `pip install`). (2) Or, pre-train 2 NN models on the full training set and ship them in the repo (but this increases repo size). (3) Add a validation check in `predict_today.py`: if either NN is None, exit with an error message.

**XGBoost optional, falls back to GradientBoosting:**
- Risk: XGBoost provides better regularization and early stopping than sklearn's GradientBoostingRegressor. If XGBoost is not installed, `model_training.py` (line 63–66) falls back silently. The fallback hyperparameters are different (`n_estimators=200` vs `300`, `max_depth=4` vs XGB's equivalent).
- Impact: Predictions using GradientBoosting will be slightly different (~5% variance). Model cache metadata doesn't record whether XGBoost was used, so reloading the cache on a system without XGBoost will apply wrong hyperparameters.
- Migration plan: (1) Make XGBoost a required dependency (pip install xgboost). (2) Or, unify hyperparameters between XGBoost and GradientBoosting (e.g., use `n_estimators=200` for both). (3) Store a flag in metadata.joblib recording whether XGBoost was used during training.

**Google Drive dependency (notebooks only):**
- Risk: Training and prediction notebooks store models on Google Drive. If Drive quota is exceeded or access is revoked, notebooks cannot load or save models. No local fallback.
- Impact: Notebooks will fail silently (attempting to load from an inaccessible path).
- Migration plan: (1) Download model cache to local disk before each notebook run (using drive.flush() or manual export). (2) Or, store models in git (as we now do for standalone). (3) Implement a GitHub Actions job that backs up the notebook cache to GCS.

---

## Missing Critical Features

**No ensemble uncertainty quantification:**
- Problem: Confidence tiers (HIGH/MED/LOW) are based on probability spread across models (`prob_std`) and distance from 0.5. But no calibration or confidence intervals. A prediction of 0.75 with confidence=HIGH may be accurate only 60% of the time (i.e., if we bet on all HIGH picks, we lose money).
- Blocks: Cannot reliably rank picks by expected value without knowing prediction reliability.
- Fix: (1) Compute calibration curve on historical holdout data (predicted prob vs actual win %). (2) Compute per-pick uncertainty as the standard deviation of predictions across the 6 models (already computed as `prob_std`, but not stored in JSON). (3) Output a confidence score that accounts for calibration.

**No feature importance attribution for predictions:**
- Problem: `predict_today.py` computes `feature_drivers` (top 5 features by Ridge/GB/Logistic contribution) for top-3 picks, but not for all predictions. Users can't understand why a given matchup has a certain probability.
- Blocks: Betting decisions are unexplainable; risky for sports betting (regulatory issue if deployed commercially).
- Fix: (1) Compute SHAP values for each prediction (via sklearn-like .feature_importance_ for tree models, gradient-based for NNs). (2) Store top 5 drivers for all predictions, not just top 3 picks. (3) Document which features favor home/away explicitly.

**No real-time monitoring of prediction accuracy:**
- Problem: No logic to compare daily predictions against actual game outcomes and track accuracy over time. Users don't know if the system is degrading.
- Blocks: Cannot detect when models become stale or when a change in the system introduces a regression.
- Fix: (1) Add a `postdiction.py` script that loads yesterday's predictions, scrapes yesterday's final scores, and computes accuracy/calibration. (2) Write results to a CSV. (3) Display accuracy trend on the website. (4) Alert if accuracy drops below 50%.

---

## Test Coverage Gaps

**Untested: feature assembly nulls and zeros:**
- What's not tested: `feature_assembly.py` fills missing efficiency and kenpom values with 0.0. This is silent and could hide upstream errors (e.g., if kenpom_ratings.py crashes, feature_assembly.py won't know and will output all-zero features). `test_feature_assembly.py` checks for NaN but not for suspicious zeros.
- Files: `feature_assembly.py` (line 250), `test_feature_assembly.py` (no zero-checking)
- Risk: If kenpom_ratings.py is broken, the downstream training_df will have correct shape but wrong semantics (all home_adj_em = 0.0 instead of true values). Models will train silently and produce bad predictions.
- Priority: **HIGH** — feature corruption is silent and hard to debug.

**Untested: name matching coverage:**
- What's not tested: Does every CBS team from games have a match in sentiment_features? Does every efficiency team map to exactly one CBS name? Are there collisions?
- Files: `feature_assembly.py` (merge logic), no test coverage
- Risk: Unmatched teams become all-zero feature rows, leading to degraded predictions for those matchups.
- Priority: **HIGH** — should be checked on every run.

**Untested: model dimensionality contract:**
- What's not tested: Do cached models have the same feature dimension as the current training data? This is checked in `model_training.py` line 950, but only during training. `predict_today.py` doesn't validate until it tries to build matchup vectors (crash at runtime if mismatch).
- Files: `predict_today.py` (lines 300–350, no pre-check), `model_training.py` (line 950, train-only check)
- Risk: Dimension mismatch causes a cryptic numpy error instead of a clear error message.
- Priority: **MEDIUM** — defensive programming; add assertion at the start of `predict_today.py`.

**Untested: home bias audit failure mode:**
- What's not tested: If home_bias_audit returns > 72%, the script exits with code 1. But GitHub Actions workflow doesn't check for this. The workflow continues to Step 7 (predict_today.py) even if Step 4 (model_training.py) exits with code 1. This is because the workflow has `if: always()` or similar on Step 7.
- Files: `.github/workflows/daily_predictions.yml` (Step 4 exit code is not checked), `model_training.py` (line 992, sys.exit(1))
- Risk: Bad models get deployed to the website and used for betting recommendations.
- Priority: **HIGH** — must add `|| exit 1` after Step 4 or check exit code explicitly.

**Untested: CBS scraper against actual live site:**
- What's not tested: Does the scraper work against the current CBS website? It was last tested Mar 7, 2026. If CBS redesigned, selectors will break silently.
- Files: `cbs_scraper.py` (no unit tests against live site)
- Risk: Silently returns 0 games; downstream data assembly produces empty CSV; models don't retrain; stale predictions deployed.
- Priority: **HIGH** — add a smoke test to GitHub Actions that runs `cbs_scraper.py` daily and asserts n_games > 30.

---

*Concerns audit: 2026-03-17*
