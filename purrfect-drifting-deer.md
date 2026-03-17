# NCAAB ML System Revisions — Implementation Plan

## Context

The project's ML pipeline needs a full refresh ahead of the 2026 March Madness tournament. The current 6-model ensemble uses GradientBoostingRegressor (a weaker tree model), no time-based validation, and the website shows no bracket visualization or model performance stats. This plan: (1) refreshes all training data as of 2026-03-16, (2) replaces GradientBoosting with XGBoost in the regression ensemble, (3) adds proper time-based train/test evaluation, (4) creates a new Jupyter evaluation notebook, (5) re-runs the bracket with fresh models, and (6) updates the website with a bracket tab and model performance tab.

---

## Phase 0 — Refresh Training Data

Run all data collection scripts in order to get fresh CSVs as of today (2026-03-16):

```bash
python "Python scripts/cbs_scraper.py"           # → cbs_games.csv
python "Python scripts/sentiment_features.py"    # → sentiment_features.csv
python "Python scripts/efficiency_metrics.py"    # → efficiency_metrics.csv
python "Python scripts/kenpom_ratings.py"        # → kenpom_ratings.csv
python "Python scripts/feature_assembly.py"      # → training_df.csv
```

No code changes needed — just run the existing scripts. Verify output row counts are >= prior run.

---

## Phase 1 — model_training.py: XGBoost + Time-Based Validation

**File:** `Python scripts/model_training.py`

### 1.1 Add XGBoost import (guarded)
```python
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
```

### 1.2 Add XGBoost hyperparameters in the CONFIGURATION block
```python
XGB_N_ESTIMATORS   = 300
XGB_LEARNING_RATE  = 0.05
XGB_MAX_DEPTH      = 4
XGB_MIN_CHILD_WT   = 3
XGB_SUBSAMPLE      = 0.85
XGB_COLSAMPLE      = 0.85
XGB_L2_REG         = 1.0
XGB_L1_REG         = 0.1
XGB_EARLY_STOPPING = 20
```

### 1.3 Replace GradientBoostingRegressor with XGBoost in train_models()
The variable stays `sgd_model` (documented misnomer — do not rename).

```python
if XGB_AVAILABLE:
    eval_size = max(1, int(0.15 * len(X_scaled)))
    sgd_model = xgb.XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS, learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH, min_child_weight=XGB_MIN_CHILD_WT,
        subsample=XGB_SUBSAMPLE, colsample_bytree=XGB_COLSAMPLE,
        reg_lambda=XGB_L2_REG, reg_alpha=XGB_L1_REG,
        early_stopping_rounds=XGB_EARLY_STOPPING,
        eval_metric="mae", random_state=42, verbosity=0,
    )
    sgd_model.fit(
        X_scaled[:-eval_size], y_diff[:-eval_size],
        eval_set=[(X_scaled[-eval_size:], y_diff[-eval_size:])],
        verbose=False,
    )
else:
    sgd_model = GradientBoostingRegressor(...)  # fallback unchanged
```

XGBoost preserves `feature_importances_` — no changes needed in `predict_today.py`'s `_compute_feature_drivers()`.

### 1.4 Replace KFold CV with time-based 80/20 holdout in evaluate_models()
- **Remove** `cross_val_score` calls (limitation #4 fix)
- Add `training_df: pd.DataFrame` parameter to get the date column
- Sort by date, take first 80% as train, last 20% as holdout test
- Compute per model on the holdout:
  - Regressors: `test_mae`, `test_rmse`
  - Classifiers: `test_acc`, `test_brier` (brier_score_loss), `test_logloss`, `test_auc`
  - Both: `split_date`, `n_train_games`, `n_test_games`
- Use mirrored row indexing: row `2*i` and `2*i+1` map to game `i`
- Return a `dict[model_name → metrics_dict]`

```python
# Imports to add at top
from sklearn.metrics import (
    mean_squared_error, brier_score_loss, log_loss, roc_auc_score
)
```

### 1.5 Store validation metrics in metadata.joblib
In `save_models()`, add to the `metadata` dict:
```python
"validation_metrics": metrics_dict,  # returned by evaluate_models()
```

### 1.6 Update print_summary() to show time-based metrics
Replace KFold columns (cv_mae, cv_acc) with holdout columns: `test_mae`, `test_rmse`, `test_brier`, `test_auc`.

### 1.7 Bump SCHEMA_VERSION
Change `SCHEMA_VERSION = "1.0"` → `SCHEMA_VERSION = "2.0"` to force cache invalidation.

---

## Phase 2 — predict_today.py: Add model_performance to JSON export

**File:** `Python scripts/predict_today.py`

In `export_json()`, add a `"model_performance"` top-level key to the JSON payload:

```python
perf = metadata.get("validation_metrics", {}) if metadata else {}
payload = {
    "generated_at": ...,
    "date": ...,
    "summary": {...},
    "model_performance": {           # NEW
        "schema_version": metadata.get("schema_version") if metadata else None,
        "metrics": perf,
    },
    "top_picks": top_picks,
    "predictions": predictions,
}
```

`metadata` is already available in `export_json()` — no signature change needed.

---

## Phase 3 — daily_predictions.yml: pip install + bracket step + cache key

**File:** `.github/workflows/daily_predictions.yml`

### 3.1 Add xgboost to pip install
```yaml
pip install feedparser vaderSentiment joblib scikit-learn xgboost numpy pandas tensorflow-cpu tabulate requests beautifulsoup4 lxml
```

### 3.2 Add Step 8 — bracket prediction (every run, not just training days)
```yaml
- name: Step 8 — Generate bracket predictions
  run: python "Python scripts/march_madness_bracket.py" --export-json --skip-refresh
```

### 3.3 Extend git commit step to include bracket JSON
```yaml
git add website/predictions_latest.json
git add website/bracket_predictions_2026.json
git diff --staged --quiet || git commit -m "chore: update predictions $(date -u +%Y-%m-%d)"
```

### 3.4 Change cache key from v1 → v2
Both `key:` and `restore-keys:` lines: `ncaab-model-v1-` → `ncaab-model-v2-`

---

## Phase 4 — march_madness_bracket.py: Copy JSON to website/

**File:** `Python scripts/march_madness_bracket.py`

In `export_bracket_json()`, after writing the local file, copy it to `website/`:

```python
import shutil
project_dir  = os.path.dirname(_SCRIPT_DIR)
website_path = os.path.join(project_dir, "website", "bracket_predictions_2026.json")
if os.path.isdir(os.path.dirname(website_path)):
    shutil.copy2(output_path, website_path)
    log.info("Bracket JSON copied to: %s", website_path)
```

No other changes — `models["sgd_model"].predict()` works identically with XGBoost.

---

## Phase 5 — index.html: Tabs + Bracket View + Model Performance

**File:** `website/index.html`

### 5.1 Add tab navigation CSS (after existing styles)
```css
.tab-nav { max-width: 1100px; margin: 0 auto 20px; padding: 0 16px;
  display: flex; gap: 4px; border-bottom: 2px solid #e2e8f0; }
.tab-btn { padding: 10px 18px; font-size: 0.78rem; font-weight: 600;
  color: #718096; background: none; border: none; cursor: pointer;
  border-bottom: 2px solid transparent; margin-bottom: -2px; }
.tab-btn.active { color: #1e3a5f; border-bottom-color: #1e3a5f; }
.tab-content { display: none; }
.tab-content.active { display: block; }
```

### 5.2 Add tab nav HTML (after header, before main)
```html
<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('picks', this)">Daily Picks</button>
  <button class="tab-btn" onclick="switchTab('bracket', this)">Bracket 2026</button>
  <button class="tab-btn" onclick="switchTab('performance', this)">Model Performance</button>
</div>
```

### 5.3 Wrap existing content in tab-picks panel; add two new panels
```html
<main class="main">
  <div id="tab-picks" class="tab-content active">
    <!-- all existing sections: stat-cards, legend, picks, optimizer, etc. -->
  </div>
  <div id="tab-bracket" class="tab-content">
    <div id="bracket-section"></div>
  </div>
  <div id="tab-performance" class="tab-content">
    <div id="performance-section"></div>
  </div>
</main>
```

### 5.4 Add switchTab() JS function
```javascript
function switchTab(name, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  btn.classList.add('active');
}
```

### 5.5 Add renderBracket() function
- Fetches `bracket_predictions_2026.json` separately (it's large and static)
- Groups games by region → round → renders a `.bracket-region` card per region
- Shows: winner, loser, win%, confidence badge, ⚡ upset flag
- Final Four + Championship rendered in a separate `.bracket-finalfour` section
- Champion displayed as a highlighted `.bracket-champion` card
- Error state: shows "Bracket data not available" if fetch fails

### 5.6 Add renderPerformance() function
- Reads `data.model_performance` from the already-loaded `predictions_latest.json`
- Shows `split_date`, `n_train_games`, `n_test_games` as stat cards
- Per-model table: regressors show test_mae + test_rmse; classifiers show test_acc + test_brier + test_auc
- If `metrics` is empty, shows "Performance metrics available on training days only."

### 5.7 Update init() to call both functions
```javascript
renderPerformance(data);   // reads from already-fetched predictions_latest.json
initBracket();             // separate fetch of bracket_predictions_2026.json
```

---

## Phase 6 — New Evaluation Notebook

**File:** `Python scripts/model_evaluation.ipynb` (new file)

Cell structure (vanilla matplotlib only, no seaborn):

| Cell | Content |
|------|---------|
| 1 | Imports + paths. Load `training_df.csv`, load model cache. |
| 2 | Time-based 80/20 split. Print split date, game counts, feature count. |
| 3 | Regression evaluation: MAE/RMSE table per model + actual-vs-predicted scatter plots. |
| 4 | Classification evaluation: accuracy/Brier/log-loss/AUC table + overlapping ROC curves. |
| 5 | Calibration plots: one subplot per classifier using `sklearn.calibration.calibration_curve`. |
| 6 | Confusion matrices: one heatmap per classifier using `ConfusionMatrixDisplay`. |
| 7 | Feature importance: Ridge coefficients (top 20 by abs value) + XGBoost importances (top 20). Color bars by feature group (home_*, away_*, diff_*). |
| 8 | Backtest: Simulate betting HIGH+MED confidence test-set picks at $5 flat stake. Plot cumulative P&L over time with directional caveat. |

---

## Phase 7 — Bracket Re-Run + Update bracket_2026.md

After model cache is rebuilt:

```bash
python "Python scripts/march_madness_bracket.py" --export-json --skip-refresh
```

Capture console output and update `bracket_2026.md` with fresh predictions + updated header date. Note: the 13 zero-vector teams from the prior run may persist unless `sentiment_features.py` re-fetches them.

---

## Execution Order

```
Phase 0 (data refresh) → must complete first
    ↓
Phase 1 (model_training.py) → delete model_cache/ first, then retrain
    ↓
Phase 2 (predict_today.py) → depends on Phase 1 metadata format
Phase 4 (march_madness_bracket.py) → depends on Phase 1 XGBoost cache
    ↓
Phase 5 (index.html) → depends on Phase 2 (model_performance key) + Phase 4 (bracket JSON)
Phase 3 (workflow YAML) → depends on Phases 1-4
    ↓
Phase 6 (notebook) → independent, can run anytime after Phase 1
Phase 7 (bracket_2026.md) → last step, after all above
```

---

## Critical Files

| File | Change |
|------|--------|
| `Python scripts/model_training.py` | XGBoost replace GB, time-based validation, SCHEMA_VERSION=2.0, validation_metrics in metadata |
| `Python scripts/predict_today.py` | Add `model_performance` key to export_json() payload |
| `Python scripts/march_madness_bracket.py` | Copy bracket JSON to website/ in export_bracket_json() |
| `website/index.html` | Tab nav, bracket tab (fetch+render), performance tab |
| `.github/workflows/daily_predictions.yml` | pip xgboost, Step 8 bracket, git add bracket JSON, cache v2 |
| `Python scripts/model_evaluation.ipynb` | New file — 8-cell evaluation notebook |
| `bracket_2026.md` | Regenerated from fresh model run |

---

## Verification Steps

1. **Phase 0**: Confirm `training_df.csv` row count ≥ 709, `kenpom_ratings.csv` has 365 rows.
2. **Phase 1**: Run `model_training.py` (after clearing `model_cache/`). Confirm: "Training sgd_model (XGBoost)", summary table shows `test_mae`/`test_auc`, home bias 🟢 <72%.
3. **Phase 2**: Run `predict_today.py --export-json`. Open `website/predictions_latest.json` → verify `"model_performance"` key is present with non-empty `"metrics"`.
4. **Phase 4**: Run `march_madness_bracket.py --export-json`. Confirm `website/bracket_predictions_2026.json` created.
5. **Phase 5**: Serve `website/` locally (`python -m http.server`). Verify 3 tabs render; "Bracket 2026" shows bracket; "Model Performance" shows metrics table.
6. **Phase 6**: Run all notebook cells; verify no import errors; calibration curves near diagonal.
7. **Phase 3**: Trigger workflow manually on GitHub Actions; confirm xgboost installs cleanly, Step 8 appears.
