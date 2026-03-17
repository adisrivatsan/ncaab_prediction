# Testing Patterns

**Analysis Date:** 2026-03-17

## Test Framework

**Runner:**
- Custom test harness (no pytest, unittest, or vitest configuration found)
- Location: `Python scripts/test_feature_assembly.py` (only test file in codebase)
- Manual test execution via Python script
- No CI/CD test pipeline configured (GitHub Actions only runs predictions, not tests)

**Assertion Library:**
- Python `assert` statements (stdlib)
- Manual equality checks and value range validation
- No specialized assertion library (no pytest assert plugins, no chai, no jest matchers)

**Run Commands:**
```bash
# Run the only test harness
python3 "Python scripts/test_feature_assembly.py"

# Output: prints shape checks, null/zero checks, feature breakdown to stdout
# No formal pass/fail exit code (only prints diagnostic output)
```

## Test File Organization

**Location:**
- Tests are co-located in `Python scripts/` directory alongside scripts being tested
- Single test file: `test_feature_assembly.py` (tests the feature assembly pipeline)
- Tests are NOT separate from implementation — no dedicated `tests/` directory

**Naming:**
- Test file prefix: `test_` (following pytest convention, though pytest not used)
- Test functions not explicitly named as test cases (script output is diagnostic, not test report)

**Structure:**
```
Python scripts/
├── feature_assembly.py          ← Module being tested
├── test_feature_assembly.py     ← Test harness for feature_assembly.py
├── cbs_scraper.py
├── sentiment_features.py
├── model_training.py
└── ...
```

## Test Structure

**Suite Organization:**
```python
# test_feature_assembly.py follows a linear verification pattern:
print("=" * 60)
print("TEST: Feature Assembly & X Matrix")
print("=" * 60)

# Load all sources
games_df      = fa.load_games(fa.CBS_CSV_PATH)
sentiment_df  = fa.load_sentiment(fa.SENTIMENT_CSV_PATH)
efficiency_df = fa.load_efficiency(fa.EFFICIENCY_CSV_PATH, fa.EFF_MAPPING_PATH)
kenpom_df     = fa.load_kenpom(fa.KENPOM_CSV_PATH)

# Build master features
features_df = fa.build_master_features(sentiment_df, efficiency_df, kenpom_df)

# Assemble training_df
training_df = fa.assemble_training_df(games_df, features_df)

# THEN: Run checks and print diagnostics
```

**Patterns:**
- No explicit setup/teardown — imports and function calls are run linearly
- No fixtures — test data is loaded from actual CSV files on disk
- No mocking — real data objects are created
- Manual diagnostic printing (no test runner output)
- All intermediate results printed for human inspection

## Specific Test Pattern (Feature Assembly)

**What's being tested:**
```python
print()
print("── Intermediate DataFrames ──────────────────────────────")
print(f"  games_df           : {games_df.shape}")
print(f"  sentiment_df       : {sentiment_df.shape}")
print(f"  efficiency_df      : {efficiency_df.shape}")
print(f"  kenpom_df          : {kenpom_df.shape}")
print(f"  master_features_df : {features_df.shape}")
print(f"  training_df        : {training_df.shape}")
```
- Verifies intermediate DataFrames have expected dimensions
- No assertions — output is visual inspection

**X matrix validation:**
```python
X = training_df[feature_cols].values

# Sanity checks
null_count  = np.isnan(X).sum()
zero_rows   = (X == 0).all(axis=1).sum()
null_cols   = [feature_cols[i] for i in range(X.shape[1]) if np.isnan(X[:, i]).any()]

print()
print("── Sanity Checks ────────────────────────────────────────")
print(f"  NaN values in X     : {null_count}")
print(f"  All-zero rows in X  : {zero_rows}")
print(f"  Cols with any NaN   : {len(null_cols)}")
```
- Checks for NaN values and all-zero rows
- Lists null columns if any exist
- Goal: ensure feature matrix is well-formed before training

**Feature breakdown:**
```python
sent_cols   = [c for c in feature_cols if not c.endswith(...)]
eff_cols    = [c for c in feature_cols if any(c.endswith(f) for f in fa.EFFICIENCY_FEATURE_COLS)]
kenpom_cols = [c for c in feature_cols if any(c.endswith(f) for f in fa.KENPOM_FEATURE_COLS)]

print()
print("── Feature Breakdown ────────────────────────────────────")
print(f"  Sentiment cols      : {len(sent_cols)}")
print(f"  Efficiency cols     : {len(eff_cols)}")
print(f"  KenPom cols         : {len(kenpom_cols)}")
```
- Counts features by source
- Verifies all three feature sources are represented

**Target distribution:**
```python
y_diff   = training_df["score_diff"].values
y_binary = training_df["home_team_won"].values

print(f"  home_team_won dist  : {int(y_binary.sum())} home wins / "
      f"{len(y_binary) - int(y_binary.sum())} away wins  "
      f"({y_binary.mean():.1%} home win rate)")
print(f"  score_diff range    : {y_diff.min():.0f} to {y_diff.max():.0f} pts")
```
- Reports class balance for classification target
- Shows range of regression target (score differential)

## Mocking

**Framework:** None
- No mocking library (no pytest-mock, unittest.mock, sinon)
- All tests use real data from CSV files

**Patterns:**
- Tests run against actual output CSVs from production pipeline
- Example: `test_feature_assembly.py` loads real `cbs_games.csv`, `sentiment_features.csv`, `efficiency_metrics.csv`, `kenpom_ratings.csv`
- Network calls NOT tested (only tested offline with pre-fetched data or in integration with live CBS/ESPN)

**What to Mock (if tests were expanded):**
- HTTP requests in scraper (requests to cbs.com, google news, espn, odds API)
- File I/O (could use tmpdir instead of on-disk CSVs)

**What NOT to Mock:**
- DataFrame operations (test with real pandas)
- Feature computation (test with real data)
- Model training (test with real training pipeline)

## Fixtures and Factories

**Test Data:**
- No fixtures defined — tests use real production data files
- Example: `test_feature_assembly.py` loads actual CSVs generated by previous pipeline runs
- Data directory: same as script location (`Python scripts/`)

**Location:**
- Test data CSVs committed to repo: `cbs_games.csv`, `sentiment_features.csv`, `efficiency_metrics.csv`, `kenpom_ratings.csv`
- These are production outputs, not test fixtures

**If Factories Were Used (not currently implemented):**
```python
# Example pattern NOT currently in use:
def build_game_df(n_games: int) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [date(2026, 3, i).isoformat() for i in range(1, n_games+1)],
        "home_name": ["Team A", "Team B"] * (n_games // 2),
        "away_name": ["Team C", "Team D"] * (n_games // 2),
        # ... other columns
    })
```

## Coverage

**Requirements:** None enforced
- No code coverage measurement configured
- No CI/CD coverage gates
- No coverage.py or Istanbul integration

**View Coverage (if measurement were added):**
```bash
# Would use coverage.py (not currently installed):
# coverage run --source=. test_feature_assembly.py
# coverage report
# coverage html
```

## Test Types

**Unit Tests:**
- None found in codebase
- If expanded, would test individual functions in isolation:
  - `keyword_score()` function behavior
  - `count_keyword_hits()` with various text inputs
  - DataFrame loading and type enforcement

**Integration Tests:**
- `test_feature_assembly.py` is an integration test
- Tests the full pipeline: load CSVs → assemble features → build training matrix
- Verifies dimensions, shapes, and data quality across all components

**E2E Tests:**
- Not implemented
- GitHub Actions workflow runs the full prediction pipeline daily but does not assert on outputs

## Common Patterns

**Diagnostic Output (not assertions):**
```python
# Pattern: Print shapes and let human verify
print(f"X.shape : {X.shape}   ← (n_games × n_features)")
print(f"y_diff.shape : {y_diff.shape}")

# Pattern: Count issues and report
null_count = np.isnan(X).sum()
print(f"NaN values in X : {null_count}")

# Pattern: List specific items if issues exist
null_cols = [feature_cols[i] for i in range(X.shape[1]) if np.isnan(X[:, i]).any()]
if null_cols:
    for c in null_cols[:5]:
        print(f"  - {c}")
```

**No explicit assertions:**
- Currently `test_feature_assembly.py` does NOT assert on expected values
- It prints diagnostic info for manual inspection
- Goal: verify data shapes match expected dimensions and no silent NaNs exist

**Implicit checks in production code (not formal tests):**
```python
# In model_training.py — assertion to catch schema drift:
assert len(MODEL_FEATURE_NAMES) == 31, (
    f"Expected 31 feature columns, got {len(MODEL_FEATURE_NAMES)}"
)

# In sentiment_features.py — validation before training:
feats = {
    k: float(pd.to_numeric(v, errors="coerce") or 0.0)
    for k, v in feats.items()
}
```

## Test Execution in CI/CD

**GitHub Actions (`.github/workflows/daily_predictions.yml`):**
- Runs production pipeline daily, not tests
- No explicit test step
- Logs output from `predict_today.py` for manual inspection
- Home bias audit gate blocks predictions if model overfits to home team

**Pipeline steps in Actions:**
```yaml
- name: Train models (every other day)
  if: env.is_training_day == 'true'
  run: python3 "Python scripts/model_training.py"
  # Implicitly validates via home_bias_audit() function
  # Exit code 1 if audit fails (stops workflow)

- name: Predict today's games
  run: python3 "Python scripts/predict_today.py" --export-json
  # Produces predictions_latest.json for website
```

## Manual Testing Approach

**Current workflow (production pipeline):**
1. Developer runs `python3 "Python scripts/test_feature_assembly.py"` locally
2. Reviews printed diagnostics for expected shapes/counts
3. Runs `python3 "Python scripts/model_training.py"` (includes home_bias_audit)
4. Checks for 🟢 (healthy) or 🔴 (failure) audit result
5. If 🟢, runs `python3 "Python scripts/predict_today.py"` to generate predictions

**Limitations of current approach:**
- No automated regression detection
- No assertion-based pass/fail criteria
- Relies on human inspection of printed output
- Home bias audit is the ONLY automated gate (exit code 1 on overfitting)

---

*Testing analysis: 2026-03-17*
