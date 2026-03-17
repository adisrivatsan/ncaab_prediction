# NCAAB Model Evaluation & Expansion

## What This Is

A Jupyter notebook for evaluating the existing NCAAB prediction ensemble against this season's games using a time-based backtest split, with XGBoost and Random Forest added as new ensemble members. The notebook is designed for interactive parameter iteration — not one-shot reporting — so hyperparameters and model configs are exposed at the top and results update with each re-run.

## Core Value

The notebook must show clearly how each model performs on held-out games so the user can tune parameters and see the impact immediately.

## Requirements

### Validated

- ✓ 6-model ensemble (Ridge, GradientBoosting, Logistic, GaussianNB, NN regressor, NN classifier) — existing
- ✓ Training data pipeline: CBS scraper → feature assembly → training_df.csv (709 rows × 90 cols) — existing
- ✓ Home bias audit (>72% threshold) — existing
- ✓ Matchup vector construction (118-dim, mirrored rows for home bias elimination) — existing
- ✓ Daily predictions via GitHub Actions + Vercel — existing

### Active

- [ ] Time-based backtest split on cbs_games.csv (train on early games, evaluate on recent games)
- [ ] Per-model accuracy, AUC, log loss, and spread coverage metrics on held-out set
- [ ] XGBoost model added to ensemble (regressor + classifier variants)
- [ ] Random Forest model added to ensemble (regressor + classifier variants)
- [ ] Model comparison table showing all models side-by-side on the same held-out set
- [ ] Hyperparameters exposed at notebook top for easy iteration

### Out of Scope

- Website changes — backtest results stay in the notebook, not surfaced on the dashboard
- Scraping additional seasons — backtest uses existing cbs_games.csv data only
- LightGBM — not requested; can be added later if XGBoost/RF comparisons motivate it
- Automated retraining triggers — user runs the notebook manually to iterate

## Context

The standalone Python scripts are the primary pipeline (not the Colab notebooks). The evaluation notebook should read from `Python scripts/training_df.csv` and `Python scripts/model_cache/` directly so it integrates with the existing local workflow. The existing `model_training.py` uses a schema version check (SCHEMA_VERSION) — adding XGBoost and Random Forest requires bumping the schema version and clearing the cache.

The existing 709 training games span Feb 16–Mar 2 2026. A time-based split (e.g. first 80% of games by date as train, last 20% as test) is the right approach — random splits leak future information.

## Constraints

- **Tech stack**: Python 3.9, scikit-learn, XGBoost, joblib — no new heavy dependencies beyond `pip install xgboost`
- **Schema**: Adding models requires SCHEMA_VERSION bump + cache clear — this is expected and documented
- **Notebook format**: Jupyter (.ipynb), runnable locally, cells with `# === CELL N — Description ===` headers per existing convention

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Time-based split for backtest | Random splits leak future information; time-split mimics real deployment | — Pending |
| XGBoost + Random Forest added to both regressor and classifier slots | Keeps ensemble symmetric; compare directly against existing Ridge/GB | — Pending |
| Hyperparameters at notebook top | User wants to iterate — not buried in cell bodies | — Pending |

---
*Last updated: 2026-03-17 after initialization*
