# Phase 1: Backtest Harness - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

A standalone Jupyter notebook (`.ipynb`) that loads `Python scripts/training_df.csv`, performs a time-based train/test split, trains (or loads from cache) the existing 6-model ensemble, and displays a per-model comparison table with accuracy, AUC, log loss, and spread coverage metrics on the held-out test set. All key hyperparameters and notebook controls are in a single config block at the top. The notebook is designed for interactive iteration — change a parameter, re-run, see updated metrics.

Phase 2 (XGBoost + Random Forest) is out of scope here. Website changes are out of scope.

</domain>

<decisions>
## Implementation Decisions

### Spread Coverage Metric (BACK-05)
- **No historical Vegas spreads available** — `training_df.csv` was assembled without odds data (odds_features.py runs at prediction time only)
- **Definition:** Coverage = % of test games where `sign(predicted_diff) == sign(actual_diff)` — i.e., the model correctly called the winning team's direction and magnitude
- **Scope:** All regressors AND the regression ensemble get a spread coverage column: `lr_model`, `sgd_model` (XGBoost), `nn_reg_model`, and the final regression ensemble
- Classifiers show accuracy/AUC/log loss; no spread coverage column for classifiers

### Notebook Code Strategy
- **Standalone, self-contained** — key logic (holdout evaluation, home bias audit, comparison table) is inlined directly into notebook cells
- Do NOT import from `model_training.py` — avoids PYTHONPATH/import path issues and makes the notebook portable
- Model training: load from `Python scripts/model_cache/` by default; config flag `FORCE_RETRAIN = False` at top controls retraining
- Changing a hyperparameter → set `FORCE_RETRAIN = True` → re-run → see updated metrics

### Config Block Scope
- Config block at notebook top contains **both** model hyperparameters and notebook controls:
  - Model hyperparameters: `RIDGE_ALPHA`, `XGB_N_ESTIMATORS`, `XGB_LEARNING_RATE`, `XGB_MAX_DEPTH`, `LOGISTIC_C`, `NN_L2`, `NN_DROPOUT`, ensemble weights
  - Notebook controls: `FORCE_RETRAIN`, `TEST_SPLIT_RATIO` (default 0.20), `SPREAD_COVER_THRESHOLD` if applicable
- `TEST_SPLIT_RATIO` is configurable — user can change from 0.20 to experiment with different holdout sizes

### NN Holdout Transparency
- NN models (nn_reg_model, nn_cls_model) are NOT re-fit on the train split during holdout eval — evaluating pre-trained weights on the test rows only
- Surface this with a **visual flag in the comparison table**: annotate NN rows with `(*)` marker
- Add a footnote below the table: `* NN models evaluated on pre-trained weights — not re-fit on train split; metrics are approximate`
- Do not hide this in comments only — it belongs in the visible output

### Comparison Table Format
- Two sub-tables (consistent with the model type split in `model_training.py`):
  1. **Regressors** — columns: Model, MAE, RMSE, Spread Coverage (%)
  2. **Classifiers** — columns: Model, Accuracy, AUC, Log Loss, Brier
- Displayed as pandas DataFrames rendered inline in the notebook (not just print statements)
- Both tables show the split metadata above: split date, train N, test N

### Cell Header Format
- Follow existing convention from `model_training.py`:
  ```python
  # =============================================================================
  # CELL N — Description
  # =============================================================================
  ```

### Claude's Discretion
- Exact pandas DataFrame styling (color, bold, decimal precision)
- Whether to use `display()` or `print()` for table output
- Exact home bias audit visual format in the notebook
- Whether to add a markdown summary cell above each table

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — Full requirement list; BACK-01 through BACK-06 and UX-01/UX-02 are Phase 1 scope
- `.planning/PROJECT.md` — Core constraints (Python 3.9, notebook format, `# === CELL N` headers)

### Existing implementation to inline from
- `Python scripts/model_training.py` — Contains all reference logic:
  - `evaluate_holdout_metrics()` (lines ~555–680): time-based 80/20 split, per-model metric computation
  - `home_bias_audit()` (lines ~679–750): home win rate audit thresholds
  - `print_summary()` (lines ~860–910): existing comparison table format
  - Config block (lines ~80–160): all existing hyperparameter names and defaults
  - `build_training_arrays()`, `build_team_lookup()`, `load_training_df()`: data loading utilities to inline

### Data
- `Python scripts/training_df.csv` — Source data: 709 rows × 90 cols; `date` column used for time-based split
- `Python scripts/model_cache/` — Pre-trained model files (8 files, schema v2.0)

No external specs beyond the above — requirements are fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluate_holdout_metrics()` in `model_training.py`: complete time-based holdout logic; clone + re-fit sklearn models on train split, evaluate NNs on pre-trained weights. Inline this logic directly.
- `home_bias_audit()` in `model_training.py`: home win rate calculation with 🔴/🟡/🟢 thresholds. Inline.
- `print_summary()` in `model_training.py`: reference for which metrics to show and how to label them. Use as design reference but render as DataFrame instead of print.
- Config block in `model_training.py` (lines ~80–160): canonical hyperparameter names and defaults to copy verbatim.

### Established Patterns
- All hyperparameters exposed at top of the script — notebook must follow this same pattern
- `is_home_idx` must be zeroed (`X_train[:, is_home_idx] = 0.0`) before fitting any model — business rule §9 of CLAUDE.md, must never be removed
- Dataset mirroring: each game → 2 rows (home + away perspective), so split on game indices then map to mirrored rows
- NNs use `EarlyStopping(patience=20)` on val_loss; they are NOT re-fit in holdout eval

### Integration Points
- Notebook reads from `Python scripts/training_df.csv` and `Python scripts/model_cache/` directly
- Notebook is a NEW file at project root or `Python scripts/` — does not modify any existing scripts
- Path resolution: use `os.path` relative to notebook location, not absolute paths (learned from hardcoded path bug in sentiment_features.py)

</code_context>

<specifics>
## Specific Ideas

No specific references or "I want it like X" moments came up — open to standard approaches for notebook layout.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-backtest-harness*
*Context gathered: 2026-03-19*
