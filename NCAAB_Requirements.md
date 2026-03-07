# NCAAB Prediction System — Requirements

---

## 0. Implementation Status (as of Mar 7, 2026)

### Standalone Python Scripts (`Python scripts/`)


| Script                     | Status         | Output                                          | Notes                                                                                        |
| -------------------------- | -------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `cbs_scraper.py`           | ✅ **Complete** | `cbs_games.csv` at project root                 | Dynamic 3-week window ending yesterday (updated Mar 7); no longer hardcoded                  |
| `sentiment_features.py`    | ✅ **Complete** | `sentiment_features.csv` — 365 teams × 32 cols  | Hardcoded absolute paths replaced with `__file__`-based paths (fixed Mar 7)                  |
| `efficiency_metrics.py`    | ✅ **Complete** | `efficiency_metrics.csv` — 365 rows × 4 metrics | Hardcoded absolute paths replaced with `__file__`-based paths (fixed Mar 7)                  |
| `kenpom_ratings.py`        | ✅ **Complete** | `kenpom_ratings.csv` — 365 rows                 | adj_em/adj_d swap fixed + CSV regenerated Mar 4 ✅                                            |
| `feature_assembly.py`      | ✅ **Complete** | `training_df.csv` — 709 rows × 90 cols          | X.shape=(709,78); 0 NaN; 61.1% home win rate; dedup fix applied                              |
| `test_feature_assembly.py` | ✅ **Complete** | console output                                  | Test harness verifying X shape, nulls, home win rate, feature inventory                      |
| `model_training.py`        | ✅ **Complete** | `model_cache/` — 8 files                        | 6 models (Ridge, GB, NN Reg, Logistic, GaussianNB, NN Cls); X=(1418,118); home bias 🟢 57.1% |
| `odds_features.py`         | ✅ **Complete** | `odds_name_mapping.csv`, `odds_snapshot.json`   | Vegas ML + spread via The Odds API; snapshot-based line-move tracking; graceful no-op if `ODDS_API_KEY` unset |
| `predict_today.py`         | ✅ **Complete** | console (4 tables) + `predictions_latest.json`  | Step 1.5 odds fetch; top_picks + all predictions enriched with vegas fields and line-move strings            |


### Website (`website/`)


| Component                                 | Status                    | Output                             | Notes                                                                                                                           |
| ----------------------------------------- | ------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `website/index.html`                      | ✅ **Complete**            | Deployed on Vercel                 | Legend; Optimizer; Pick Analysis; Today's Picks + Optimizer show Vegas ML, Spread, Line Move (team-labeled, color-coded)        |
| `website/predictions_latest.json`         | ✅ **Auto-updated daily**  | Written by GitHub Actions cron     | Vercel auto-deploys on push; includes feature_drivers, articles, top_picks, vegas_ml, vegas_spread, odds_movement, line-move    |
| `website/vercel.json`                     | ✅ **Complete Mar 7 2026** | Vercel deploy config               | Cache-Control headers (`s-maxage=3600`); SPA rewrite rule                                                                       |
| `.github/workflows/daily_predictions.yml` | ✅ **Complete Mar 7 2026** | Mon-Fri 3pm EDT + Sat-Sun 10am EDT | Every-other-day training (odd day-of-month); Steps 1–6 conditional; model cache via `actions/cache`; `ODDS_API_KEY` in Step 7  |


### Verified X Matrix — feature_assembly.py output (Mar 3 2026)

```
training_df  : (709, 90)  — 12 game cols + 78 feature cols
X_raw        : (709, 78)  — 709 games × 78 features  [feature_assembly / test harness only]

Features per team: 39
  Sentiment (news)  : 31
  Efficiency (4-Fac): 4
  KenPom / T-Rank   : 4
NaN in X: 0  |  All-zero rows: 0  |  Home win rate: 61.1%
```

### Verified X Matrix — model_training.py (matchup vectors, Mar 3 2026)

```
training_df  : (709, 90)   — loaded from training_df.csv
X            : (1418, 118) — 709 games × 2 mirrored rows × (39+39+39+1) matchup dims
y_diff       : (1418,)     — score differential (regression target)
y_binary     : (1418,)     — home_team_won (classification target)
is_home_idx  : 117         — zeroed to 0.0 during training; 1.0 at prediction time

Home bias audit: 🟢 HEALTHY (57.1%)
```

### Data Sources


| Source                                       | Notebook             | Standalone Script         | Status                                                            |
| -------------------------------------------- | -------------------- | ------------------------- | ----------------------------------------------------------------- |
| CBS Sports scoreboard                        | ✅ Cell 2             | ✅ `cbs_scraper.py`        | Fully operational                                                 |
| Google News + ESPN sentiment                 | ✅ Cell 3             | ✅ `sentiment_features.py` | Fully operational                                                 |
| Efficiency metrics (eFG%, TOV%, ORB%, FTR)   | ❌ Not implemented    | ✅ `efficiency_metrics.py` | Standalone complete; not in notebooks                             |
| KenPom-style ratings (via Bart Torvik)       | ❌ Not implemented    | ✅ `kenpom_ratings.py`     | Bug fixed; CSV needs re-run; not in notebooks                     |
| Feature assembly (all sources → training_df) | ✅ Notebook Cell 5    | ✅ `feature_assembly.py`   | Standalone extends notebook with +8 features                      |
| Model training (6 models)                    | ✅ Cell 10 (5 models) | ✅ `model_training.py`     | Standalone adds GaussianNB (bayes_model); 118-dim matchup vectors |
| Daily predictions                            | ✅ Cells 11–12        | ✅ `predict_today.py`      | Loads from model_cache/; outputs 4 tables; 53 games tested Mar 3  |
| Vegas odds + line moves                      | ❌ Not applicable     | ✅ `odds_features.py`      | The Odds API; h2h + spreads; snapshot-based movement tracking; standalone prediction-time only (no training impact) |


---

## 1. Project Overview

**Purpose:** Produce daily win-probability and score-differential predictions for NCAA Men's Division I Basketball games to support betting decisions.

**Primary outputs:**

- Win probability (home team perspective)
- Score differential (predicted home margin)
- Confidence tier: HIGH / MED / LOW
- Top 3 straight betting picks with expected value
- Parlay combinations from top picks

**Scope:** NCAA Men's Division I Basketball only. Women's basketball is explicitly excluded at every data collection step.

**Environment:** Google Colab + Google Drive. All code lives in Jupyter notebook cells; no local Python package or CLI.

**Current data window:** dynamic 21-day window ending yesterday (computed at runtime by `cbs_scraper.py`; ~700+ final games, 365 unique teams)

---

## 2. System Architecture

### 2.1 Data Flow

```
CBS Sports Scoreboard
        │
        ▼
  get_games_for_date_cbs()
  [game rows: away, home, scores, status]
        │
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
  Google News RSS              ESPN Search API
  fetch_google_news_articles() fetch_espn_news_articles()
  fetch_espn_injury_articles() fetch_espn_lineup_articles()
        │                                                 │
        └──────────────┬──────────────────────────────────┘
                       ▼
             extract_team_features()
             [27-feature vector per team]
                       │
                       ▼
             build_matchup_vector()
             [82-dim: home_27 | away_27 | diff_27 | is_home]
                       │
              ┌────────┴────────┐
              ▼                 ▼
      Regression (×3)   Classification (×2)
      lr_model          logistic_model
      sgd_model         nn_cls_model
      nn_reg_model
              │                 │
              └────────┬────────┘
                       ▼
              Final Ensemble
              (55% reg + 45% cls)
                       │
                       ▼
              Betting Optimizer
              [top 3 picks + parlays]
```

### 2.2 Pipeline Layers


| Layer                       | Components                                          | Output                                  |
| --------------------------- | --------------------------------------------------- | --------------------------------------- |
| Data Acquisition            | 4 agents (CBS, KenPom, Efficiency, Sentiment)       | raw game rows + team features           |
| Feature Engineering         | `extract_team_features()`, `build_matchup_vector()` | 82-dim matchup vectors                  |
| Model Training + Prediction | 5 models across 2 notebooks                         | win prob, score diff, confidence, picks |


---

## 3. Data Sources and Agent Specifications

### 3.1 CBS Game List Collector — **IMPLEMENTED** ✅ (notebook + standalone script)

**Agent file:** `Agents/CBS_games.md`

**URL pattern:** `https://www.cbssports.com/college-basketball/scoreboard/FBS/{YYYYMMDD}/`

**DOM selectors:**

- Game cards: `div.single-score-card`
- Game identity: `card.get('data-abbrev')` — format `..._AWAY@HOME`
- Status: `div.game-status` — must contain "final" (case-insensitive)
- Team rows: all `<tr>` elements where `tr.get('class') is not None` (static HTML renders both rows as class `tiedGame`)
- Row order: row 0 = away, row 1 = home (determined by `data-abbrev`, not by CSS class)
- Team name: `a.team-name-link` text
- Score: `td.total` text, cast to `int`

**Status filter:** Only cards where status contains "final" are processed. Non-final rows are silently skipped.

**Output schema:**


| Column        | Type              | Null policy            |
| ------------- | ----------------- | ---------------------- |
| date          | str (YYYY-MM-DD)  | never null             |
| away_name     | str               | never null             |
| away_score    | int               | null if unparseable    |
| home_name     | str               | never null             |
| home_score    | int               | null if unparseable    |
| status        | str ("Final")     | never null             |
| home_team_won | int (0 or 1)      | null if scores missing |
| winner_name   | str               | null if scores missing |
| winner_score  | int               | null if scores missing |
| loser_name    | str               | null if scores missing |
| loser_score   | int               | null if scores missing |
| score_diff    | int (home − away) | null if scores missing |


**Error handling:** Card-level exceptions are caught and logged; page-level failures emit `❌ Failed to fetch {date_str}`. Parse anomalies log selectors tried. Retries use backoff on 5xx/timeout.

---

### 3.2 KenPom Ratings Harvester — **STANDALONE SCRIPT COMPLETE** ✅ (bug fixed; CSV needs re-run)

**Agent file:** `Agents/KenPomRatings.md`

**Intended data:** Overall rating, adjusted offensive/defensive efficiencies (AdjO, AdjD, AdjEM), rank, and any Elo-like strength proxy defined by the project.

**Hard constraint:** KenPom requires a paid subscription. The agent must **never** attempt auto-login, CAPTCHA bypass, or unauthorized scraping. The only permitted ingestion methods are:

1. User-provided CSV export from kenpom.com
2. A sanctioned data feed or API (if made available)
3. Manual download → ingest workflow

**Planned output schema:**


| Column      | Type             | Notes                                  |
| ----------- | ---------------- | -------------------------------------- |
| team_id     | str              | canonical name matching CBS team names |
| kenpom_rank | int              | overall rank                           |
| adj_em      | float            | adjusted efficiency margin             |
| adj_o       | float            | adjusted offensive efficiency          |
| adj_d       | float            | adjusted defensive efficiency          |
| as_of_date  | str (YYYY-MM-DD) | snapshot date                          |


**Free source used:** Bart Torvik (barttorvik.com) — free public KenPom equivalent providing adj_em, adj_o, adj_d, and rank for all D-I teams. Script: `Python scripts/kenpom_ratings.py`. Includes fuzzy name matching to CBS team names (output: `kenpom_name_mapping.csv`).

**Known bug (fixed):** `_extract_positional_default` had adj_em/adj_d columns swapped — index 5 is a defensive rank integer, index 6 is adj_d; adj_em must be derived as adj_o − adj_d. Fixed in script Mar 3 2026. Existing `kenpom_ratings.csv` is stale and must be regenerated before training.

**Deduplication note:** Fuzzy name matching maps multiple Torvik team names to the same CBS name in some cases (36 duplicate rows detected). `feature_assembly.py` resolves this by averaging numeric values per cbs_name.

**Integration status:** Standalone script complete and tested. Joined to training_df via `feature_assembly.py` (left join on cbs_name). Not yet integrated into notebooks. Adding to notebook schema requires version bump (see Rule 7).

---

### 3.3 Efficiency Metrics Collector — **STANDALONE SCRIPT COMPLETE** ✅ (tested Mar 3 2026)

**Agent file:** `Agents/Efficiency.md`

**Intended metrics with exact formulas:**


| Metric | Formula                        | Unit |
| ------ | ------------------------------ | ---- |
| eFG%   | (FGM + 0.5 × 3PM) / FGA        | 0–1  |
| TOV%   | TOV / (FGA + 0.44 × FTA + TOV) | 0–1  |
| ORB%   | ORB / (ORB + opponent DRB)     | 0–1  |
| FTR    | FTA / FGA                      | 0–1  |


**Planned output schema:** team_id, efg_pct, tov_pct, orb_pct, ftr, win_loss_overall, win_loss_conference (optional), as_of_date, schema_version.

**Free source used:** Bart Torvik (primary) with Sports-Reference CBB advanced stats as fallback. Script: `Python scripts/efficiency_metrics.py`. Includes fuzzy name matching to CBS team names (output: `team_name_mapping.csv`).

**Known bug (non-blocking):** `_parse_torvik_json` tries `int(wins)` on W-L strings like `"19-12"`, raising ValueError for every row. Source returns 0 records. Sports Reference fallback activates automatically and delivers correct data for all 365 teams.

**Deduplication note:** Fuzzy name matching creates 25 duplicate cbs_name rows. `feature_assembly.py` resolves this by averaging numeric values per cbs_name. 9 teams remain unmatched and receive 0.0 for all efficiency features.

**Integration status:** Standalone script complete and tested (365 teams, all 4 metrics via Sports Reference). Joined to training_df via `feature_assembly.py`. Not yet integrated into notebooks. Adding to notebook schema requires version bump (Rule 7).

---

### 3.5 Vegas Odds Fetcher — **STANDALONE SCRIPT COMPLETE** ✅ (Mar 7 2026)

**Script:** `Python scripts/odds_features.py`

**API:** The Odds API — `GET https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/?apiKey=...&regions=us&markets=h2h,spreads&oddsFormat=american`

**Auth:** `ODDS_API_KEY` environment variable. Set as GitHub Actions secret `ODDS_API_KEY`. No key = graceful no-op (returns `[]`, all Vegas fields show `N/A`).

**Output fields per game (added to `top_picks` and all `predictions` entries in JSON):**

| Field               | Type | Description                                                    |
|---------------------|------|----------------------------------------------------------------|
| `vegas_ml`          | str  | Real Vegas moneyline for the recommended bet side (e.g. "-145") |
| `vegas_spread`      | str  | Home team spread (e.g. "-3.5") or "N/A"                        |
| `odds_movement`     | str  | Movement for bet side (e.g. "-145 → -160") — top_picks only   |
| `home_ml_movement`  | str  | Home team ML movement string — all predictions                  |
| `away_ml_movement`  | str  | Away team ML movement string — all predictions                  |

**Line-move tracking:** On each run, `build_odds_lookup()` loads `odds_snapshot.json` (previous run's odds), computes movement strings, then overwrites the snapshot. First run always shows `"—"`. Movement is shown as `"prev → curr"` (e.g., `"-145 → -160"`).

**Bookmaker priority:** Averages across DraftKings, FanDuel, BetMGM, Caesars if any are present; falls back to all available books.

**Name mapping:** 6-step: hardcoded override → exact → case-insensitive exact → difflib fuzzy (full name) → difflib fuzzy (first-2-word prefix) → case-insensitive fuzzy. Writes `odds_name_mapping.csv` for audit. `OVERRIDE_MAP` has ~15 known problem cases (e.g. "Eastern Kentucky Colonels" → "E. Kentucky").

**Error handling:** HTTP 422/429 → log warning, return `[]`. All retries exhausted → return `[]`. JSON parse failure → return `[]`. No exception propagates to the calling script.

---

### 3.4 News Sentiment Feature Builder — **IMPLEMENTED** ✅ (notebook + standalone script)

**Agent file:** `Agents/sentiment.md`

**Four fetch functions:**


| Function                       | Source                     | Default limit                                  |
| ------------------------------ | -------------------------- | ---------------------------------------------- |
| `fetch_google_news_articles()` | Google News RSS            | 20 articles; full body fetched for top 3 links |
| `fetch_espn_news_articles()`   | ESPN common search API     | 15 articles                                    |
| `fetch_espn_injury_articles()` | ESPN search, injury suffix | 10 articles                                    |
| `fetch_espn_lineup_articles()` | ESPN search, lineup suffix | 10 articles                                    |


**Women's filter:** `is_womens_article(title, full_text)` applies to every fetched article before it enters feature computation.

- 24 filter terms (e.g., "women's basketball", "wnba", "lady ", "wbb ", "she ", " her ")
- Title: 1 matching term is sufficient to reject the article
- Body (first 2000 chars): 2+ matching terms required to reject (avoids false positives)

**Rate limits:** 0.4 s sleep between team fetches; 10–12 s request timeout; redirect errors and timeouts are silently swallowed (no retry loops).

**Article retention policy:** Full article bodies are used only for transient feature computation. Only feature scalars are stored. No body text is written to disk.

---

## 4. Data Schemas

### 4.1 Game Row Schema

12 columns as described in §3.1. Key integrity rules:

- `status` is always "Final" (only finals enter any downstream use)
- `score_diff` = `home_score` − `away_score`
- `home_team_won` = 1 if home_score > away_score, 0 if away_score > home_score, null if tied or missing

### 4.2 Team Feature Schema

**Notebook schema:** 31 numeric features produced by `extract_team_features()`, grouped into 6 categories (note: CLAUDE.md §3 states 27; actual count from implementation is 31 — see groups A–F below). All values are floats in [0, 1] or [-1, 1] (sentiment).

**Standalone script schema:** 39 features per team (31 sentiment + 4 efficiency + 4 kenpom). The extra 8 features are only available in the standalone pipeline via `feature_assembly.py`.

**[A] Sentiment (7 features)**


| Feature                    | Computation                             | Range   |
| -------------------------- | --------------------------------------- | ------- |
| sent_overall               | mean VADER compound across all articles | [-1, 1] |
| sent_espn                  | mean VADER compound, ESPN articles only | [-1, 1] |
| sent_google                | mean VADER compound, Google News only   | [-1, 1] |
| sent_headlines             | mean VADER compound, titles only        | [-1, 1] |
| sent_recent                | mean VADER compound, last 5 articles    | [-1, 1] |
| sent_pct_positive_articles | fraction with compound > 0.05           | [0, 1]  |
| sent_pct_negative_articles | fraction with compound < −0.05          | [0, 1]  |


**[B] Injury (5 features)**


| Feature                | Computation                                                    | Range      |
| ---------------------- | -------------------------------------------------------------- | ---------- |
| inj_mention_rate       | fraction of (all + injury) articles hitting any INJURY keyword | [0, 1]     |
| inj_severity_score     | max keyword_score() across combined texts                      | [0, 1]     |
| inj_article_count_norm | min(injury_article_count / 10, 1.0)                            | [0, 1]     |
| inj_key_player_flag    | 1.0 if star-player injury regex matches                        | {0.0, 1.0} |
| inj_sentiment          | mean VADER compound, injury articles only                      | [-1, 1]    |


**[C] Lineup / Roster (5 features)**


| Feature                     | Computation                                                               | Range   |
| --------------------------- | ------------------------------------------------------------------------- | ------- |
| lineup_change_signal        | count_keyword_hits(combined_lineup, LINEUP_KEYWORDS)                      | [0, 1]  |
| lineup_starter_mention_rate | count_keyword_hits(combined_lineup, ['starter','starting'])               | [0, 1]  |
| lineup_benched_signal       | count_keyword_hits(combined_lineup, ['bench','benched','moved to bench']) | [0, 1]  |
| lineup_roster_instability   | 0.5 × lineup_change_signal + 0.5 × inj_mention_rate, capped at 1.0        | [0, 1]  |
| lineup_sentiment            | mean VADER compound, lineup articles only                                 | [-1, 1] |


**[D] Momentum / Form (6 features)**


| Feature                    | Computation                                      | Range   |
| -------------------------- | ------------------------------------------------ | ------- |
| momentum_win_mention_rate  | count_keyword_hits(all_texts, WIN_KEYWORDS)      | [0, 1]  |
| momentum_loss_mention_rate | count_keyword_hits(all_texts, LOSS_KEYWORDS)     | [0, 1]  |
| momentum_score             | count_keyword_hits(all_texts, MOMENTUM_KEYWORDS) | [0, 1]  |
| momentum_slump_score       | count_keyword_hits(all_texts, SLUMP_KEYWORDS)    | [0, 1]  |
| momentum_net               | momentum_score − momentum_slump_score            | [-1, 1] |
| momentum_win_loss_ratio    | win_rate / max(loss_rate, 0.01)                  | [0, ∞)  |


**[E] Context / Environment (5 features)**


| Feature                   | Computation                                       | Range      |
| ------------------------- | ------------------------------------------------- | ---------- |
| ctx_coaching_mention_rate | count_keyword_hits(all_texts, COACHING_KEYWORDS)  | [0, 1]     |
| ctx_coaching_instability  | 1.0 if coaching-change regex matches              | {0.0, 1.0} |
| ctx_ranking_mention_rate  | count_keyword_hits(all_texts, RANKING_KEYWORDS)   | [0, 1]     |
| ctx_fatigue_signal        | count_keyword_hits(all_texts, FATIGUE_KEYWORDS)   | [0, 1]     |
| ctx_home_away_context     | count_keyword_hits(all_texts, HOME_AWAY_KEYWORDS) | [0, 1]     |


**[F] Data Quality (3 features)**


| Feature                  | Computation                                    | Range                     |
| ------------------------ | ---------------------------------------------- | ------------------------- |
| data_total_articles_norm | min(total_articles / 35.0, 1.0)                | [0, 1]                    |
| data_source_diversity    | (gn>0) + (espn>0) + (inj>0) + (lineup>0) / 4.0 | {0, 0.25, 0.5, 0.75, 1.0} |
| data_confidence          | 0.5 × articles_norm + 0.5 × source_diversity   | [0, 1]                    |


**Metadata columns (not features):** `team_name`

### 4.3 Matchup Feature Vector

`build_matchup_vector(home_name, away_name, lookup, cols)` returns a 82-dim numpy array:

```
[home_feat_1 … home_feat_27 | away_feat_1 … away_feat_27 | diff_feat_1 … diff_feat_27 | is_home]
 ←────── 27 ──────────────→   ←────── 27 ──────────────→   ←────── 27 ─────────────→    ← 1 →
```

- `diff_*` = home_vec − away_vec
- `is_home` = 1.0 at prediction time (home team perspective), **0.0 during training** (see Rule 9)

**Dataset mirroring:** Each game produces two training rows — one from the home perspective (is_home=0.0) and one from the away perspective (swapped vectors, negated score_diff) — to eliminate home-team positional bias during training.

### 4.4 Prediction Output Schema

`predict_game()` returns a dict with these keys:


| Key                   | Description                                         |
| --------------------- | --------------------------------------------------- |
| Home Team             | home team name                                      |
| Away Team             | away team name                                      |
| Ridge Diff            | score differential from `lr_model`                  |
| GB Diff               | score differential from `sgd_model`                 |
| NN Reg Diff           | score differential from `nn_reg_model`              |
| Reg Ensemble Diff     | weighted avg: 30% Ridge + 30% GB + 40% NN Reg       |
| Logistic Prob         | win probability from `logistic_model`               |
| NN Cls Prob           | win probability from `nn_cls_model`                 |
| Cls Ensemble Prob     | weighted avg: 40% Logistic + 60% NN Cls             |
| Final Win Prob (Home) | 55% reg_implied_prob + 45% cls_ensemble_prob        |
| Predicted Winner      | team name                                           |
| Confidence            | HIGH / MED / LOW                                    |
| Home ML               | American moneyline for home team                    |
| Away ML               | American moneyline for away team                    |
| Spread Comparison     | Vegas vs model spread string (N/A if no Vegas line) |
| Spread Edge           | model recommendation vs Vegas                       |


---

## 5. Model Specifications

### 5.1 Model Inventory


| Variable         | Type       | Target        | Algorithm                 | Key Hyperparameters                                                   | Pipeline        |
| ---------------- | ---------- | ------------- | ------------------------- | --------------------------------------------------------------------- | --------------- |
| `lr_model`       | Regressor  | score_diff    | Ridge Regression          | alpha=20.0                                                            | Both            |
| `sgd_model`      | Regressor  | score_diff    | GradientBoostingRegressor | n_estimators=200, lr=0.05, max_depth=4, min_samples=3, subsample=0.85 | Both            |
| `nn_reg_model`   | Regressor  | score_diff    | Neural Network            | hidden=[128,64], L2=0.01, dropout=0.05, lr=0.0003, batch=32           | Both            |
| `logistic_model` | Classifier | home_team_won | LogisticRegression        | C=0.5, elasticnet, class_weight='balanced'                            | Both            |
| `bayes_model`    | Classifier | home_team_won | Gaussian Naive Bayes      | var_smoothing=1e-9                                                    | Standalone only |
| `nn_cls_model`   | Classifier | home_team_won | Neural Network            | hidden=[128,64], L2=0.01, dropout=0.05, lr=0.0003, batch=32           | Both            |


> **Note:** `sgd_model` is a `GradientBoostingRegressor`, not an SGD model. The variable name is a known misnomer (see §9, Limitation 6).
>
> **Note:** `bayes_model` (GaussianNB) is a Bayesian probability classifier added to the standalone pipeline only. It models per-feature Gaussian distributions conditioned on class label. Not yet ported to the notebook.

### 5.2 Ensemble Weights

**Regression ensemble** (produces predicted score differential):

```
reg_ensemble_diff = 0.30 × lr_diff + 0.30 × sgd_diff + 0.40 × nn_reg_diff
```

**Classification ensemble** (produces win probability):

*Notebook (5 models — no bayes_model):*

```
cls_ensemble_prob = 0.40 × logistic_prob + 0.60 × nn_cls_prob
```

*Standalone (6 models — includes bayes_model):*

```
cls_ensemble_prob = 0.35 × logistic_prob + 0.20 × bayes_prob + 0.45 × nn_cls_prob
```

**Final ensemble** (combined win probability — same in both pipelines):

```
final_win_prob = 0.55 × reg_implied_prob + 0.45 × cls_ensemble_prob
```

where `reg_implied_prob` = logistic transform of `reg_ensemble_diff` with scale factor 10.

### 5.3 Confidence Tiers


| Tier | Condition                        |
| ---- | -------------------------------- |
| HIGH | std_dev < 0.08 AND margin > 0.15 |
| MED  | std_dev < 0.15 AND margin > 0.08 |
| LOW  | all other cases                  |


`std_dev` = standard deviation across model win probabilities (reg_implied, logistic, bayes, nn_cls in standalone; 5 model probs in notebook). `margin` = |final_win_prob − 0.5|.

### 5.4 Home Bias Audit

Run before every production prediction session. Computes the fraction of training predictions where home team is predicted to win.


| Home Win Rate | Status               | Action                                                                   |
| ------------- | -------------------- | ------------------------------------------------------------------------ |
| > 72%         | 🔴 OVERFIT HOME      | Increase regularization (RIDGE_ALPHA, GB_MIN_SAMPLES, NN_L2, NN_DROPOUT) |
| 62–72%        | 🟡 CAUTION           | Monitor; consider mild regularization increase                           |
| 50–62%        | 🟢 HEALTHY           | No action required                                                       |
| < 50%         | 🟡 UNDERVALUING HOME | Decrease regularization                                                  |


Predictions must not run if status is 🔴.

### 5.5 Model Cache System

**Notebook storage path:** `/content/drive/MyDrive/ncaab_model_cache/`

**Standalone storage path:** `Python scripts/model_cache/`

**Cached files:**


| File                    | Contents                                                                                           | Pipeline        |
| ----------------------- | -------------------------------------------------------------------------------------------------- | --------------- |
| `scaler.joblib`         | fitted StandardScaler                                                                              | Both            |
| `ridge_model.joblib`    | fitted `lr_model`                                                                                  | Both            |
| `gb_model.joblib`       | fitted `sgd_model`                                                                                 | Both            |
| `logistic_model.joblib` | fitted `logistic_model`                                                                            | Both            |
| `bayes_model.joblib`    | fitted `bayes_model` (GaussianNB)                                                                  | Standalone only |
| `nn_regressor.keras`    | fitted `nn_reg_model`                                                                              | Both            |
| `nn_classifier.keras`   | fitted `nn_cls_model`                                                                              | Both            |
| `metadata.joblib`       | schema_version, feature_names, numeric_cols, n_features, is_home_idx, ensemble_weights, as_of_date | Standalone only |


**Behavior matrix:**


| USE_CACHE | FORCE_RETRAIN | Result                                  |
| --------- | ------------- | --------------------------------------- |
| True      | False         | Load from cache if exists; train if not |
| True      | True          | Always retrain and overwrite cache      |
| False     | False         | Always train; never read or write cache |
| False     | True          | Same as above (USE_CACHE=False wins)    |


---

## 6. Website

### 6.1 Architecture Overview

```
predict_today.py  ──(daily GitHub Actions cron)──►  predictions_latest.json
                                                              │
                                              committed to repo (or Vercel Blob)
                                                              │
                                                    Vercel static frontend
                                                              │
                                                    Reads JSON → renders tables
```

### 6.2 GitHub Actions Workflow Spec

File: `.github/workflows/daily_predictions.yml`


| Field            | Value                                                                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Trigger          | Mon-Fri `cron: '0 19 * * 1-5'` (3pm EDT / 19:00 UTC) + Sat-Sun `cron: '0 14 * * 0,6'` (10am EDT / 14:00 UTC) + `workflow_dispatch` |
| Runner           | `ubuntu-latest`                                                                                                                    |
| Python version   | `3.9`                                                                                                                              |
| Training days    | Odd calendar day-of-month only (Steps 1–6 have `if: steps.training_check.outputs.is_training_day == 'true'`)                      |
| Model cache      | `actions/cache/restore@v4` before install; `actions/cache/save@v4` after Step 6 (training days only); key prefix `ncaab-model-v1-` |
| Steps            | checkout → training_check → restore cache → install → Steps 1–6 (conditional) → save cache → Step 7 → commit JSON                |
| Step 7 env       | `ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}` injected at prediction step only                                                       |
| Commit message   | `chore: update predictions [YYYY-MM-DD]`                                                                                           |
| Auth             | GitHub Actions bot token (built-in `GITHUB_TOKEN`)                                                                                 |
| UTC note         | 19:00 UTC = 3pm EDT (off by 1hr during EST months Nov–Mar — acceptable for this season)                                            |
| Failure behavior | Workflow fails loudly; previous JSON remains; frontend shows stale data warning                           |


Dependencies to install:

```
pip install feedparser vaderSentiment joblib scikit-learn numpy pandas tensorflow tabulate requests beautifulsoup4 lxml
```

### 6.3 predict_today.py Modification — JSON Export

Add `--export-json` CLI flag (or always export alongside console output).

**Output file:** `Python scripts/predictions_latest.json`

**JSON schema:**

```json
{
  "generated_at": "YYYY-MM-DDTHH:MM:SS",
  "date": "YYYY-MM-DD",
  "summary": {
    "total_games": 53,
    "home_picks": 30,
    "away_picks": 23,
    "tensorflow_available": true,
    "confidence": { "HIGH": 6, "MED": 16, "LOW": 31 }
  },
  "predictions": [
    {
      "home_team": "string",
      "away_team": "string",
      "ridge_diff": 3.14,
      "gb_diff": 2.87,
      "nn_reg_diff": 3.42,
      "reg_ensemble_diff": 3.14,
      "logistic_prob": 0.62,
      "bayes_prob": 0.58,
      "nn_cls_prob": 0.65,
      "cls_ensemble_prob": 0.63,
      "reg_implied_prob": 0.58,
      "final_win_prob_home": 0.60,
      "final_win_prob_away": 0.40,
      "predicted_winner": "string",
      "home_ml": "-150",
      "away_ml": "+130",
      "model_std_dev": 0.054,
      "confidence": "HIGH",
      "home_ml_movement": "-145 → -150",
      "away_ml_movement": "+125 → +130",
      "feature_drivers": [],
      "articles_home": [],
      "articles_away": []
    }
  ],
  "top_picks": [
    {
      "rank": 1,
      "pick": "string",
      "matchup": "Away @ Home",
      "home_team": "string",
      "away_team": "string",
      "bet_side": "HOME",
      "ml": "-150",
      "win_prob": 0.62,
      "confidence": "MED",
      "stake": 5.0,
      "if_win": 3.33,
      "ev": 0.287,
      "vegas_ml": "-145",
      "vegas_spread": "-3.5",
      "odds_movement": "-140 → -145",
      "home_ml_movement": "-140 → -145",
      "away_ml_movement": "+120 → +125"
    }
  ]
}
```

### 6.4 Frontend Page Layout

**Design:** Analytics Dashboard — dark navy gradient header, stat number cards (Total/HIGH/MED/LOW/Home/Away), picks table with confidence-colored left-border rows, collapsible data sections. Light gray page background (`#eef2f7`), white card surfaces.

**File structure:**

```
website/
├── index.html        ← main page (self-contained HTML/CSS/JS — no build step)
└── vercel.json       ← Vercel config (rewrites, headers)
```

**Page layout (top → bottom):**

1. **Header** — "NCAAB Predictions — [Date]" + generated timestamp + game count
2. **Table 4 (PRIMARY)** — HIGH/MED Confidence Picks
  - Columns: Conf badge, Pick, Matchup (Away @ Home), Win Prob, Moneyline, Line Move, Std Dev
  - Line Move cell: shows team name whose line moved toward (green if favors pick, red if against) + movement string (e.g., "-145 → -160"); shows "—" if no movement or first run
  - HIGH picks highlighted (e.g., green badge); MED picks secondary (yellow)
  - Sorted: HIGH first, then by descending win probability
3. **Summary Stats bar** — Total games | Home picks N | Away picks N | HIGH: N | MED: N | LOW: N
4. **Collapsible Section: "All Games — Final Ensemble" (Table 3)**
  - Columns: Home, Away, Reg→Prob, Cls Prob, Final%, Predicted Winner, Home ML, Away ML, Std Dev, Conf
  - Collapsed by default; expand on click
5. **Collapsible Section: "Model Breakdown — Classification" (Table 2)**
  - Columns: Home, Away, Logistic, Bayes, NN Cls, Cls Ensemble
  - Collapsed by default
6. **Collapsible Section: "Model Breakdown — Regression" (Table 1)**
  - Columns: Home, Away, Ridge, GB, NN Reg, Ensemble Diff
  - Collapsed by default
7. **Footer** — "Predictions auto-refresh daily. Last updated: [timestamp]"

**Stale data handling:** If `generated_at` is > 36 hours old, show a warning banner: "⚠️ Predictions may be stale — last updated [date]"

### 6.5 Vercel Deployment Config

File: `website/vercel.json`

```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [{ "key": "Cache-Control", "value": "s-maxage=3600, stale-while-revalidate" }]
    }
  ]
}
```

Vercel reads `predictions_latest.json` directly from the repo. No API routes needed.

### 6.6 Betting Optimizer Table (Top Picks)

Columns: # | Pick & Matchup | Conf | ML (model) | Win Prob | Vegas ML | Spread | Line Move | Stake | If Win | EV

- **Vegas ML** — real moneyline for bet side from The Odds API; styled red (favorite) / blue (underdog); "N/A" in gray if unavailable
- **Spread** — home team spread from Odds API; "N/A" if unavailable
- **Line Move** — same team-labeled cell as Today's Picks table (team name + movement string, color-coded)

### 6.7 Data Freshness Flow

```
Mon-Fri 3pm EDT / Sat-Sun 10am EDT (daily):
  GitHub Actions → training_check (odd day?)
    → [odd day] Steps 1–6: scrape → features → assemble → train → save model cache
    → [every day] Step 7: predict_today.py --export-json (with ODDS_API_KEY)
                        → fetches CBS games
                        → fetches Vegas odds (The Odds API)
                        → runs models
                        → writes predictions_latest.json
                        → git commit + push to main
                        → Vercel auto-deploys (triggered by push)
                        → site updated within ~1 min of commit
```

Manual re-run: trigger `workflow_dispatch` from GitHub Actions UI.

---

## 7. Business Rules (Non-Negotiable)

1. **"Final means final."** Only games with status "Final" enter training data or prediction targets. Do not infer, approximate, or backfill incomplete games.
2. **"One row per game."** The natural game key is the `data-abbrev` attribute from CBS. Deduplication must be deterministic; no heuristic merging.
3. **"If it's not licensed, it's not data."** KenPom data requires a paid subscription. The system must never attempt automated login, CAPTCHA bypass, or paywall circumvention. Ingest only via user-provided CSV or sanctioned export.
4. **"Define the stat or don't ship it."** Every efficiency metric must have a documented formula and unit before it is added to the feature schema. Undocumented metrics may not be computed or stored.
5. **Women's filter must apply to all article fetches.** `is_womens_article()` must be called on every article from every fetch function before the article enters feature computation. No exceptions for "fast path" or cached fetches.
6. **Home bias audit must pass before any production predictions.** If the audit returns 🔴 (> 72% predicted home win rate), predictions must not be presented. Adjust hyperparameters and retrain before proceeding.
7. **Schema version bump required for any new feature column.** Adding a column to the team feature schema requires: updating both notebooks, deleting the model cache, and incrementing the schema version field in stored metadata.
8. **Missing values stay missing.** If a stat cannot be computed or fetched, its value must be null/NaN with a logged explanation. Do not impute with means, medians, or zeroes except at the explicit `pd.to_numeric(..., errors='coerce').fillna(0.0)` step in the numeric-enforcement pass.
9. `**is_home` must be zeroed to 0.0 during training.** The flag is set to 1.0 during inference to communicate the home-team perspective to the model. Training on 1.0 would leak home-team identity. This zeroing must never be removed.
10. **No full article body storage beyond transient computation.** Article bodies are used only to compute scalar features in memory. No body text is written to any CSV, Excel file, or Drive storage.

---

## 8. Feature Engineering Specifications

### 8.1 Keyword Dictionaries (verbatim)

```python
INJURY_KEYWORDS    = ['injur','hurt','out for','day-to-day','doubtful','questionable',
                       'sidelined','knee','ankle','concussion','surgery','absence',
                       'missed','will not play',"won't play"]

LINEUP_KEYWORDS    = ['starting lineup','starting five','lineup change',
                       'inserted into the starting','moved to the bench','benched',
                       'starter','rotation change','depth chart','replacing','new starter']

WIN_KEYWORDS       = ['win','victory','beat','defeated','upset','dominant','blowout',
                       'rolled','cruised','overcame','rallied','comeback','unbeaten','streak']

LOSS_KEYWORDS      = ['loss','lost','defeated','fell to','blown out','collapse',
                       'losing streak','skid','slide','struggle','winless']

MOMENTUM_KEYWORDS  = ['hot','on fire','rolling','clicking','momentum','confidence',
                       'energized','surging','impressive run','on a roll','winning streak']

SLUMP_KEYWORDS     = ['slump','cold','struggling','disappointing','slow','inconsistent',
                       'frustrating','skidding','dropped','winless','rough patch']

COACHING_KEYWORDS  = ['coach','head coach','staff','scheme','strategy','adjustment',
                       'system','game plan','coaching staff','fired','hired','contract',
                       'press conference']

RANKING_KEYWORDS   = ['ranked','ranking','top 25','ap poll','net ranking','bracketology',
                       'seed','projection','poll','ballot','moved up','dropped','climbed']

FATIGUE_KEYWORDS   = ['back-to-back','travel','tired','rest','fatigue','load management',
                       'minutes','heavy schedule','third game','quick turnaround']

HOME_AWAY_KEYWORDS = ['home court','home crowd','road game','away game',
                       'hostile environment','sold out','neutral site',
                       'home advantage','visiting']

WOMENS_FILTER_TERMS = [
    "women's basketball","womens basketball","women's ncaa","womens ncaa",
    "wnba","w-nba","lady ","ladies ","wbb "," wbb",
    "ncaa women","women's college basketball","womens college basketball",
    "girls basketball","female basketball","women's team","womens team",
    "she "," her "," hers ","women's march madness",
]
```

### 8.2 Helper Function Specs

`**keyword_score(text, keywords)**`

- Returns: float in [0, 1]
- Logic: `min(hit_count / max(len(keywords) × 0.15, 1), 1.0)`
- Use: single-article density; used for `inj_severity_score`

`**count_keyword_hits(texts, keywords)**`

- Returns: float in [0, 1]
- Logic: fraction of texts where at least one keyword appears (substring match, lowercased)
- Use: rate features (mention rates, signals)

`**mean_vader(texts)**`

- Returns: float in [-1, 1], or 0.0 if texts is empty
- Logic: mean of `SentimentIntensityAnalyzer().polarity_scores(t)['compound']` across all texts

---

## 9. Known Limitations


| #   | Name                                       | Impact                                                                                                                                                                    | Recommended Fix                                                                |
| --- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 1   | **Tiny training window**                   | 21-day rolling window (~700+ games) is insufficient for reliable generalization; models will overfit to recent game patterns                                              | Expand date range to full season (Nov – Mar)                                   |
| 2   | **KenPom not in notebooks**                | adj_em/adj_o/adj_d absent from notebook models; standalone `kenpom_ratings.py` collects data and `feature_assembly.py` joins it to training_df, but notebooks not updated | Port to notebooks; bump schema version; clear model cache                      |
| 3   | **Efficiency metrics not in notebooks**    | eFG%, TOV%, ORB%, FTR absent from notebook models; standalone `efficiency_metrics.py` collects data and `feature_assembly.py` joins it, but notebooks not updated         | Port to notebooks; bump schema version; clear model cache                      |
| 4   | **No out-of-sample validation**            | Cross-validation folds share teams across train/test splits; reported CV metrics overstate real-world accuracy                                                            | Use time-based splits or leave-out-team CV                                     |
| 5   | **Manual Vegas moneylines**                | `compare_to_spread()` always returns "N/A" in practice; no automated odds ingestion                                                                                       | Integrate a public odds API or manual entry workflow                           |
| 6   | **Variable name confusion**                | `sgd_model` is a `GradientBoostingRegressor`, not an SGD model; misleads future developers                                                                                | Rename variable to `gb_model` in both notebooks; clear cache after rename      |
| 7   | **No women's filter in training notebook** | Training Data notebook does not call `is_womens_article()`, so women's basketball articles may pollute team features during data collection                               | Port the filter from the predictor notebook to Cell 3 of the training notebook |
| 8   | **Team name brittleness**                  | Team names come directly from CBS HTML text (e.g., "Boston U.", "Bethune-Cook."); no alias table exists, so any CBS text change breaks joins between games and features   | Build a canonical alias table; add a normalization step to all join keys       |


---

## 10. Tech Stack Reference


| Library          | Purpose                                                         | Version    |
| ---------------- | --------------------------------------------------------------- | ---------- |
| pandas           | DataFrames, CSV I/O, merge operations                           | pip latest |
| numpy            | Array operations, vector math                                   | pip latest |
| requests         | HTTP fetching (CBS, ESPN)                                       | pip latest |
| beautifulsoup4   | CBS HTML parsing                                                | pip latest |
| lxml             | HTML parser backend                                             | pip latest |
| feedparser       | Google News RSS parsing                                         | pip latest |
| vaderSentiment   | Sentiment scoring                                               | pip latest |
| scikit-learn     | Ridge, GradientBoosting, LogisticRegression, StandardScaler, CV | pip latest |
| TensorFlow/Keras | Neural network models (regressor + classifier)                  | pip latest |
| joblib           | Model serialization (sklearn models)                            | pip latest |
| openpyxl         | Excel output formatting                                         | pip latest |
| tabulate         | Console table formatting                                        | pip latest |


**Runtime:** Google Colab (Python 3.x). Models and data CSVs persist on Google Drive at `/content/drive/MyDrive/Colab Notebooks/` and `/content/drive/MyDrive/ncaab_model_cache/`.