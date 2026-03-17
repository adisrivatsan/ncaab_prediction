# Roadmap: NCAAB Model Evaluation & Expansion

## Overview

Two phases deliver a runnable evaluation notebook: Phase 1 builds the time-based backtest harness with per-model metrics and a comparison table, exposing all hyperparameters at the top so the user can iterate. Phase 2 adds XGBoost and Random Forest to the ensemble, integrates them into the same comparison table, and confirms the home bias audit passes on the expanded ensemble.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Backtest Harness** - Time-based split, per-model metrics, comparison table, hyperparameter config block
- [ ] **Phase 2: New Models** - XGBoost + Random Forest added to ensemble, schema bumped, home bias audit on full ensemble

## Phase Details

### Phase 1: Backtest Harness
**Goal**: User can run the notebook and see clear per-model accuracy metrics on held-out games, with all hyperparameters configurable at the top
**Depends on**: Nothing (first phase)
**Requirements**: BACK-01, BACK-02, BACK-03, BACK-04, BACK-05, BACK-06, UX-01, UX-02
**Success Criteria** (what must be TRUE):
  1. Running the notebook top-to-bottom produces a model comparison table with accuracy, AUC, log loss, and spread coverage for all existing models on the same held-out test set
  2. The train/test split is time-based (no random shuffling) and the held-out set contains only games from dates after the training cutoff
  3. All key hyperparameters (alpha, n_estimators, max_depth, ensemble weights, train/test split ratio) are defined in a single config block at the top of the notebook
  4. Re-running the notebook after changing a hyperparameter in the config block updates all metric outputs without manual cell edits elsewhere
**Plans**: TBD

### Phase 2: New Models
**Goal**: XGBoost and Random Forest are integrated into the ensemble and compared against existing models on the same held-out test set, with schema updated and home bias confirmed
**Depends on**: Phase 1
**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05, MODEL-06, UX-03
**Success Criteria** (what must be TRUE):
  1. The comparison table includes XGBoost and Random Forest rows alongside the existing 6 models, all evaluated on the same held-out set from Phase 1
  2. XGBoost and Random Forest ensemble weights are adjustable from the Phase 1 config block and affect the final ensemble output
  3. SCHEMA_VERSION is bumped in model_training.py and the notebook documents the required cache-clear step
  4. Home bias audit runs on the expanded ensemble and displays a pass (<=72% predicted home win rate) before predictions are shown
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Backtest Harness | 0/TBD | Not started | - |
| 2. New Models | 0/TBD | Not started | - |
