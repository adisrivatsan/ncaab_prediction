# Requirements: NCAAB Model Evaluation & Expansion

**Defined:** 2026-03-17
**Core Value:** Clear per-model backtest metrics on held-out games so the user can tune parameters and see impact immediately.

## v1 Requirements

### Backtesting

- [ ] **BACK-01**: Notebook splits training_df.csv by date (time-based, not random) into train and held-out test set
- [ ] **BACK-02**: All models evaluated on the same held-out test set for fair comparison
- [ ] **BACK-03**: Per-model win prediction accuracy displayed (% games with correct winner)
- [ ] **BACK-04**: Per-model AUC and log loss displayed for win probability calibration
- [ ] **BACK-05**: Per-model spread coverage rate displayed (predicted spread vs Vegas spread)
- [ ] **BACK-06**: Model comparison table shows all models side-by-side with metrics

### New Models

- [ ] **MODEL-01**: XGBoost regressor added (predicts score differential)
- [ ] **MODEL-02**: XGBoost classifier added (predicts home team win probability)
- [ ] **MODEL-03**: Random Forest regressor added (predicts score differential)
- [ ] **MODEL-04**: Random Forest classifier added (predicts home team win probability)
- [ ] **MODEL-05**: New models integrated into ensemble with configurable weights
- [ ] **MODEL-06**: SCHEMA_VERSION bumped and model cache cleared when new models added

### Notebook UX

- [ ] **UX-01**: All key hyperparameters (alpha, n_estimators, max_depth, weights, etc.) defined at notebook top in a config block
- [ ] **UX-02**: Re-running notebook updates all metrics and comparison tables
- [ ] **UX-03**: Home bias audit runs on the full ensemble after new models are added

## v2 Requirements

### Extended Evaluation

- **EVAL-01**: K-fold cross-validation with time-based folds
- **EVAL-02**: Calibration plots (reliability diagrams) per model
- **EVAL-03**: ROI simulation — if you bet $100 on each HIGH/MED pick, what's the return?

### Website Integration

- **WEB-01**: Surface backtest accuracy metrics on the dashboard
- **WEB-02**: Model version indicator on the website

## Out of Scope

| Feature | Reason |
|---------|--------|
| LightGBM | Not requested; revisit if XGBoost/RF don't add value |
| Scraping older seasons | Backtest uses existing data only |
| Automated retraining | User iterates manually in notebook |
| Dashboard changes | Notebook-only output for this milestone |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BACK-01 | Phase 1 | Pending |
| BACK-02 | Phase 1 | Pending |
| BACK-03 | Phase 1 | Pending |
| BACK-04 | Phase 1 | Pending |
| BACK-05 | Phase 1 | Pending |
| BACK-06 | Phase 1 | Pending |
| MODEL-01 | Phase 2 | Pending |
| MODEL-02 | Phase 2 | Pending |
| MODEL-03 | Phase 2 | Pending |
| MODEL-04 | Phase 2 | Pending |
| MODEL-05 | Phase 2 | Pending |
| MODEL-06 | Phase 2 | Pending |
| UX-01 | Phase 1 | Pending |
| UX-02 | Phase 1 | Pending |
| UX-03 | Phase 2 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-17*
*Last updated: 2026-03-17 after initial definition*
