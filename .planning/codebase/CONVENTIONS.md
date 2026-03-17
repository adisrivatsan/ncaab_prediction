# Coding Conventions

**Analysis Date:** 2026-03-17

## Naming Patterns

**Files:**
- Lowercase with underscores: `cbs_scraper.py`, `sentiment_features.py`, `model_training.py`
- Test files follow standard naming: `test_feature_assembly.py`
- Descriptive names tied to function: `cbs_scraper.py` contains CBS scraping logic, `model_training.py` contains model training code
- Agent specs use title case: `CBS_games.md`, `KenPomRatings.md`

**Functions:**
- Lowercase with underscores: `get_games_for_date_cbs()`, `extract_team_features()`, `build_training_arrays()`, `load_training_df()`
- Private functions prefix with underscore: `_fetch_page()`, `_safe_get()`, `_build_nn_regressor()`, `_fetch_odds_response()`
- Getter functions use `get_` or `load_` prefix
- Builder/constructor functions use `build_` prefix
- Fetch/scrape functions use `fetch_` prefix

**Variables:**
- Snake_case: `game_date`, `home_name`, `away_score`, `feature_cols`, `numeric_cols`
- Canonical DataFrame names (project convention — frozen): `training_df`, `master_games_df`, `master_features_df`, `games_df`, `sentiment_df`
- Model variable names (frozen per requirements): `lr_model` (Ridge, not logistic), `sgd_model` (GradientBoosting, not SGD — documented misnomer), `nn_reg_model`, `logistic_model`, `bayes_model`, `nn_cls_model`
- Lookup/cache dicts use descriptive names: `processed_teams`, `lookup`, `models`, `metadata`
- Loop counters: `i`, `attempt`, `idx`

**Types:**
- PascalCase: `DataFrame`, `StandardScaler`, `LogisticRegression`, `Ridge`, `XGBRegressor`
- Type hints use modern Python 3.9+ syntax: `dict[str, float]`, `list[dict]`, `np.ndarray`, `Optional[...]`
- Union types use pipe: `str | None` (from `__future__ import annotations`)

**Constants:**
- UPPERCASE_WITH_UNDERSCORES: `REQUEST_TIMEOUT`, `RIDGE_ALPHA`, `START_DATE`, `END_DATE`, `HEADERS`, `RETRY_BACKOFF`, `HOME_BIAS_RED`
- Configuration constants grouped at module top under `# CONFIG` section
- Feature schema constants: `EFFICIENCY_FEATURE_COLS`, `KENPOM_FEATURE_COLS`, `MODEL_FEATURE_NAMES`

## Code Style

**Formatting:**
- No explicit formatter configured (no `.prettierrc`, `.eslintrc`, or `pyproject.toml` found)
- Conventions inferred from existing code:
  - 4-space indentation (Python standard)
  - Lines break after long string literals or function signatures
  - Dictionary literals use trailing commas when multi-line
  - Imports organized: `from __future__ import annotations` first, then stdlib, then third-party, then local

**Imports:**
- All Python scripts begin with: `from __future__ import annotations` (enables postponed evaluation for modern type hints)
- Organized into blocks: future imports, stdlib (os, sys, logging, datetime), third-party (pandas, numpy, sklearn, requests), local modules
- Explicit imports preferred over `import *`
- Optional dependencies wrapped in try/except with feature flags:
  ```python
  try:
      import tensorflow as tf
      TF_AVAILABLE = True
  except ImportError:
      TF_AVAILABLE = False
  ```

**Linting:**
- No linting configuration found (no `.eslintrc*`, `.flake8`, `pylintrc`)
- Code follows implicit PEP 8 conventions
- Unused imports and variables are cleaned up (seen in read files)

**Comments:**
- Section headers use visual separators:
  ```python
  # =============================================================================
  # SECTION NAME
  # =============================================================================
  ```
- Cell headers (from notebook convention):
  ```python
  # =============================================================================
  # CELL N — Description
  # =============================================================================
  ```
- Inline comments explain non-obvious logic: `# silently skip`, `# mirrored rows eliminate home bias`
- Business rule references: `# Business Rule 2`, `# Business rule §5`
- No JSDoc-style docstrings — instead use triple-quoted module and function docstrings with plain English

**Docstrings:**
- Module docstrings at file top: triple-quoted with multi-line description of what the module does
- Function docstrings describe inputs, returns, and side effects:
  ```python
  def extract_team_features(team_name: str) -> dict:
      """
      Fetch news from 4 sources and compute all 27 feature scalars for a team.
      Returns a flat dict keyed by feature name. All values are floats.
      """
  ```
- Return type explicitly mentioned when non-obvious
- Side effects documented (e.g., "saves CSV to disk")

## Import Organization

**Order:**
1. `from __future__ import annotations` (always first)
2. stdlib modules (os, sys, logging, datetime, time, json, csv, re, math, warnings, argparse, shutil, xml)
3. Third-party (pandas, numpy, sklearn, requests, tensorflow, feedparser, beautifulsoup4, vaderSentiment, tabulate, joblib)
4. Local modules (relative imports with `import module_name as _alias` when optional)

**Path Aliases:**
- File paths constructed with `os.path.join()` and `os.path.dirname()`:
  ```python
  _SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
  _PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
  CBS_CSV_PATH = os.path.join(_PROJECT_DIR, "cbs_games.csv")
  ```
- This pattern (fixed after hardcoded path bug) ensures portability across environments
- Environment variables accessed via `os.environ.get("VAR_NAME", default)`

## Error Handling

**Patterns:**
- Explicit exceptions over silent failures (project philosophy, CLAUDE.md §4)
- Try/except for HTTP requests with specific exception types:
  ```python
  try:
      resp = requests.get(url, timeout=REQUEST_TIMEOUT)
      return resp
  except requests.exceptions.Timeout:
      return None
  except requests.exceptions.RequestException:
      return None
  ```
- Graceful degradation for optional features (TensorFlow, XGBoost, Odds API):
  ```python
  if TF_AVAILABLE:
      nn_model = keras.Model(...)
  else:
      log.warning("TensorFlow not available — NNs skipped")
  ```
- Card-level parse errors caught and logged in scraper; game skipped:
  ```python
  except Exception as exc:
      print(f"  ❌ Card-level error [{card.get('data-abbrev', '?')}]: {exc}")
      continue
  ```
- Silent fallback ONLY for team feature extraction on network error:
  ```python
  try:
      feats = extract_team_features(team_name)
  except Exception as exc:
      log.error("extract_team_features() failed for '%s': %s", team_name, exc)
      feats = zero_feature_vector()  # neutral values, not None
  ```

**Validation:**
- Type enforcement via `pd.to_numeric(val, errors="coerce").fillna(0.0)` (project convention):
  ```python
  feats = {
      k: float(pd.to_numeric(v, errors="coerce") or 0.0)
      for k, v in feats.items()
  }
  ```
- DataFrame numeric column enforcement before training:
  ```python
  for col in feature_cols:
      df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
  ```
- Assertions for schema integrity:
  ```python
  assert len(MODEL_FEATURE_NAMES) == 31, (
      f"Expected 31 feature columns, got {len(MODEL_FEATURE_NAMES)}"
  )
  ```

## Logging

**Framework:** Python's `logging` module (stdlib)

**Setup pattern:**
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
```

**Patterns:**
- Progress headers for long-running operations:
  ```python
  log.info("Step 1/5 — Loading training data")
  log.info("Step 2/5 — Building matchup vectors")
  ```
- Operation results: `log.info("Models saved        : %s", CACHE_DIR)`
- Warnings for non-fatal issues: `log.warning("Schema version mismatch...")`
- Errors for failures: `log.error("Home bias audit FAILED...")`
- Print statements for progress bars and detail output during iteration:
  ```python
  print(f"[{i}/{total}] Fetching: {team_name}")
  print(f"  sent_overall={feats['sent_overall']:.4f}")
  ```

## Function Design

**Size:**
- Most functions 20–100 lines (seen in samples)
- Longer functions (>200 lines) break into named sub-steps with comments:
  ```python
  # ── Step 1: Load data ────────────────
  # ── Step 2: Build vectors ────────────
  # ── Step 3: Train models ────────────
  ```

**Parameters:**
- Typed with modern syntax: `def func(name: str, items: list[dict]) -> dict:`
- Optional parameters marked: `Optional[pd.DataFrame]`, `scaler: Optional[StandardScaler] = None`
- Position-only used rarely; keyword-only arguments not explicitly marked

**Return Values:**
- Single return value: direct type (e.g., `→ dict`, `→ np.ndarray`)
- Multiple returns via tuple: `→ tuple[dict, list[str]]`
- None explicitly typed when function returns nothing: `→ None`
- Early returns for error cases or guards

## Module Design

**Exports:**
- All functions and constants in the module are implicitly public unless prefixed with underscore
- No explicit `__all__` list found in samples
- Private helpers: `_fetch_page()`, `_safe_get()`, `_build_nn_regressor()`

**Barrel Files:**
- Not used; each script is standalone executable or importable module
- Local imports use direct `import module_name` (e.g., `import feature_assembly as fa`)

**Script Structure Pattern (all standalone scripts follow this):**
```
1. Module docstring (what the script does)
2. from __future__ import annotations
3. Standard library imports
4. Third-party imports
5. # =============================================================================
   # CONFIGURATION
   # =============================================================================
   (Constants, paths, tuples)
6. # =============================================================================
   # LOGGING
   # =============================================================================
   (Setup logging)
7. # =============================================================================
   # HELPER FUNCTIONS / BUSINESS LOGIC
   # =============================================================================
   (Functions grouped by logical section)
8. if __name__ == "__main__":
       main()
```

## Special Patterns

**DataFrame Column Management:**
- Exclude non-feature columns by name: `_GAME_COLS = {"date", "away_name", ...}`
- Feature columns extracted by filtering: `feature_cols = [c for c in df.columns if c not in _GAME_COLS]`
- Home/away prefixes used for feature mirroring: columns are duplicated as `home_*` and `away_*`
- No rename operations in-place; instead create new columns or use prefix/strip:
  ```python
  home_feats = master_features_df.add_prefix('home_').rename(columns={'home_team_name': 'home_name'})
  away_feats = master_features_df.add_prefix('away_').rename(columns={'away_team_name': 'away_name'})
  ```

**Team Name Handling (CRITICAL):**
- Team names flow from CBS HTML text directly to join keys with ZERO normalization (CLAUDE.md §4)
- No case folding, no trimming extra spaces — string matching is exact
- Fuzzy matching used only in external data mapping (efficiency, kenpom), with override table for known problem cases:
  ```python
  OVERRIDE_MAP: dict[str, str] = {
      "Eastern Kentucky Colonels": "E. Kentucky",
      "Miami Hurricanes": "Miami (FL)",
  }
  ```

**NumPy Array Operations:**
- Typed as `np.ndarray` or `np.ndarray[float]` in docstrings
- Created with explicit dtype: `np.zeros((n, m), dtype=np.float32)`
- Indexing uses integer arrays for row selection: `X[sorted_game_idx[:n_train]]`
- Boolean indexing for filtering: `(final_prob >= 0.5).mean()`

---

*Convention analysis: 2026-03-17*
