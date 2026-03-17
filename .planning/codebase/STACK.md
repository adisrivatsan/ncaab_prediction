# Technology Stack

**Analysis Date:** 2026-03-17

## Languages

**Primary:**
- Python 3.9 - All data collection, ML training, and prediction scripts
- JavaScript/HTML/CSS - Frontend dashboard

**Secondary:**
- YAML - GitHub Actions workflow configuration
- JSON - Configuration and data export formats
- CSV - Data interchange format between pipeline stages

## Runtime

**Environment:**
- Python 3.9 (local development, GitHub Actions runner)
  - Uses `from __future__ import annotations` for forward compatibility

**Package Manager:**
- pip (Python package manager)
- No lockfile (`requirements.txt` not present; dependencies listed in `.github/workflows/daily_predictions.yml`)

## Frameworks

**Core ML & Data Processing:**
- scikit-learn 1.x - Ridge regression, LogisticRegression, GradientBoostingRegressor, GaussianNB, StandardScaler
- TensorFlow/Keras (tensorflow-cpu) - Neural network models (regressor and classifier)
- XGBoost (xgboost) - Optional XGBoost regressor (fallback to GradientBoostingRegressor if not installed)
- pandas - DataFrame operations, CSV I/O
- NumPy - Array operations and numerical computation

**Data Collection & Processing:**
- requests - HTTP library for web scraping and API calls
- BeautifulSoup4 (bs4) - HTML parsing for CBS Sports scraper
- lxml - XML/HTML parser backend for BeautifulSoup
- feedparser - RSS feed parsing for Google News
- vaderSentiment - Sentiment analysis for news articles

**Utilities:**
- joblib - Model serialization and caching
- tabulate - Console table formatting for prediction output

**Frontend:**
- Vanilla JavaScript (no framework) - DOM manipulation and JSON rendering
- HTML5/CSS3 - Dashboard layout and styling

## Key Dependencies

**Critical:**
- scikit-learn - Core ML models (Ridge, Logistic, GaussianNB, GradientBoosting)
- TensorFlow/Keras - Neural network models (regressor, classifier)
- pandas - All data transformation and aggregation
- requests - Web scraping and API integration (CBS, ESPN, Google News RSS, Odds API, Bart Torvik)

**Infrastructure:**
- BeautifulSoup4 + lxml - HTML parsing from CBS Sports
- feedparser - RSS parsing from Google News
- vaderSentiment - Sentiment intensity scoring for feature extraction
- joblib - Model persistence and feature scaler serialization
- tabulate - Pretty-printed console output for predictions table

**Optional (graceful degradation if missing):**
- TensorFlow - NN models skipped if not installed; system continues with 4 other models
- XGBoost - Falls back to sklearn.ensemble.GradientBoostingRegressor if not installed

## Configuration

**Environment Variables:**
- `ODDS_API_KEY` - API key for The Odds API (https://the-odds-api.com)
  - Used by `Python scripts/odds_features.py` in Step 7 of prediction pipeline
  - Optional: if empty, Vegas odds fetch gracefully skips
  - Injected into GitHub Actions via `secrets.ODDS_API_KEY` in `.github/workflows/daily_predictions.yml` line 113

**Build & Runtime Configuration:**
- `.github/workflows/daily_predictions.yml` - GitHub Actions cron schedule
  - Mon-Fri: 3pm EDT (19:00 UTC)
  - Sat-Sun: 10am EDT (14:00 UTC)
  - Every-other-day training (odd day-of-month); daily prediction
  - Python 3.9 setup, pip dependency installation, 75-minute timeout

**Vercel Deployment:**
- `website/vercel.json` - Static site config with SPA rewrites and cache headers
  - Cache-Control: `s-maxage=3600, stale-while-revalidate` (1-hour server cache)
  - SPA rewrite: all routes serve `/index.html` for client-side routing

## Platform Requirements

**Development:**
- Local machine: macOS (zsh shell)
- Python 3.9
- Ability to run standalone Python scripts or Jupyter notebooks (Google Colab for notebooks)
- Google Drive for notebook data persistence (training CSVs, model cache)

**Production:**
- GitHub Actions runner (ubuntu-latest) - daily cron job runs prediction pipeline
- Vercel - static site hosting for `website/index.html` and `website/predictions_latest.json`
- No database required; all data stored as CSV and JSON files in git repo

**Data Pipeline Execution:**
- Step 1 (Scraper): HTTP requests to CBS Sports (https://www.cbssports.com/college-basketball/scoreboard/)
- Step 2-4 (Feature Collection): HTTP requests to Google News RSS, ESPN APIs, Bart Torvik T-Rank
- Step 5 (Odds): HTTP requests to The Odds API
- Step 6 (Training): Local model training using scikit-learn + TensorFlow
- Step 7 (Prediction): Loads cached models, generates JSON export

## Dependency List (from GitHub Actions installer, line 60-72)

```
feedparser              # RSS parsing
vaderSentiment         # Sentiment analysis
joblib                 # Model serialization
scikit-learn           # ML models
xgboost                # XGBoost regressor (optional)
numpy                  # Numerical computing
pandas                 # Data frames
tensorflow-cpu         # Neural networks (CPU-only in GitHub Actions)
tabulate               # Console tables
requests               # HTTP library
beautifulsoup4         # HTML parsing
lxml                   # XML/HTML parser
```

---

*Stack analysis: 2026-03-17*
