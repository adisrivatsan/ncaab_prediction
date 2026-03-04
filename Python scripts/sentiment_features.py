"""
NCAAB Sentiment Feature Extractor
Reads team names from cbs_games.csv, fetches news articles from Google News RSS
and ESPN APIs, and computes a 27-feature sentiment/context vector per team.
Output: sentiment_features.csv
"""

from __future__ import annotations

import time
import re
import logging
import xml.etree.ElementTree as ET

import requests
import feedparser
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =============================================================================
# SECTION — Configuration and Constants
# =============================================================================

CBS_GAMES_PATH       = "/Users/adityasrivatsan/Documents/NCAAB Prediction/cbs_games.csv"
OUTPUT_PATH          = "/Users/adityasrivatsan/Documents/NCAAB Prediction/Python scripts/sentiment_features.csv"
RATE_LIMIT_SLEEP     = 0.4   # seconds between teams
REQUEST_TIMEOUT      = 11    # seconds (within 10-12s window)
FULL_BODY_TOP_N      = 3     # fetch full body for top N Google News articles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

analyzer = SentimentIntensityAnalyzer()

# =============================================================================
# SECTION — Keyword Dictionaries
# =============================================================================

INJURY_KEYWORDS = [
    "injur", "hurt", "out for", "day-to-day", "doubtful", "questionable",
    "sidelined", "knee", "ankle", "concussion", "surgery", "absence",
    "missed", "will not play", "won't play",
]

LINEUP_KEYWORDS = [
    "starting lineup", "starting five", "lineup change",
    "inserted into the starting", "moved to the bench", "benched",
    "starter", "rotation change", "depth chart", "replacing", "new starter",
]

WIN_KEYWORDS = [
    "win", "victory", "beat", "defeated", "upset", "dominant", "blowout",
    "rolled", "cruised", "overcame", "rallied", "comeback", "unbeaten", "streak",
]

LOSS_KEYWORDS = [
    "loss", "lost", "defeated", "fell to", "blown out", "collapse",
    "losing streak", "skid", "slide", "struggle", "winless",
]

MOMENTUM_KEYWORDS = [
    "hot", "on fire", "rolling", "clicking", "momentum", "confidence",
    "energized", "surging", "impressive run", "on a roll", "winning streak",
]

SLUMP_KEYWORDS = [
    "slump", "cold", "struggling", "disappointing", "slow", "inconsistent",
    "frustrating", "skidding", "dropped", "winless", "rough patch",
]

COACHING_KEYWORDS = [
    "coach", "head coach", "staff", "scheme", "strategy", "adjustment",
    "system", "game plan", "coaching staff", "fired", "hired", "contract",
    "press conference",
]

RANKING_KEYWORDS = [
    "ranked", "ranking", "top 25", "ap poll", "net ranking", "bracketology",
    "seed", "projection", "poll", "ballot", "moved up", "dropped", "climbed",
]

FATIGUE_KEYWORDS = [
    "back-to-back", "travel", "tired", "rest", "fatigue", "load management",
    "minutes", "heavy schedule", "third game", "quick turnaround",
]

HOME_AWAY_KEYWORDS = [
    "home court", "home crowd", "road game", "away game",
    "hostile environment", "sold out", "neutral site",
    "home advantage", "visiting",
]

WOMENS_FILTER_TERMS = [
    "women's basketball", "womens basketball", "women's ncaa", "womens ncaa",
    "wnba", "w-nba", "lady ", "ladies ", "wbb ", " wbb",
    "ncaa women", "women's college basketball", "womens college basketball",
    "girls basketball", "female basketball", "women's team", "womens team",
    "she ", " her ", " hers ", "women's march madness",
]

# Regex patterns
_STAR_PLAYER_INJURY_RE = re.compile(
    r"\b(star|starter|key player|leading scorer|top scorer|best player"
    r"|captain|point guard|center|forward|guard)\b.{0,60}"
    r"\b(injur|out for|sidelined|doubtful|questionable|missed)\b",
    re.IGNORECASE | re.DOTALL,
)

_COACHING_CHANGE_RE = re.compile(
    r"\b(fired|resigned|stepped down|replaced|interim coach|new head coach"
    r"|coaching change|hired as|takes over as)\b",
    re.IGNORECASE,
)

# =============================================================================
# SECTION — Women's Filter
# =============================================================================

def is_womens_article(title: str, full_text: str) -> bool:
    """Return True if the article is about women's basketball and should be filtered out."""
    title_lower = title.lower()
    if any(term in title_lower for term in WOMENS_FILTER_TERMS):
        return True
    body_lower = full_text[:2000].lower()
    hits = sum(1 for term in WOMENS_FILTER_TERMS if term in body_lower)
    return hits >= 2

# =============================================================================
# SECTION — Helper Functions
# =============================================================================

def keyword_score(text: str, keywords: list) -> float:
    """Returns float in [0, 1] representing keyword density."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(hits / max(len(keywords) * 0.15, 1), 1.0)


def count_keyword_hits(texts: list, keywords: list) -> float:
    """Returns fraction of texts where at least one keyword appears."""
    if not texts:
        return 0.0
    hits = sum(1 for t in texts if any(kw in t.lower() for kw in keywords))
    return hits / len(texts)


def mean_vader(texts: list) -> float:
    """Returns mean compound VADER score, 0.0 if empty."""
    if not texts:
        return 0.0
    return float(np.mean([analyzer.polarity_scores(t)["compound"] for t in texts]))

# =============================================================================
# SECTION — Article Fetch Functions
# =============================================================================

def _safe_get(url: str, **kwargs) -> requests.Response | None:
    """
    GET request with timeout. Returns Response or None on timeout/redirect error.
    Silently swallows timeout and redirect exceptions (no retry loop).
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, **kwargs)
        return resp
    except requests.exceptions.Timeout:
        return None
    except requests.exceptions.TooManyRedirects:
        return None
    except requests.exceptions.RequestException:
        return None


def fetch_article_body(url: str) -> str:
    """Fetch full page text from a URL. Returns empty string on failure."""
    resp = _safe_get(url, allow_redirects=True)
    if resp is None or resp.status_code != 200:
        return ""
    # Strip HTML tags with a basic regex to get readable text
    raw = resp.text
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_google_news_articles(team_name: str) -> list[dict]:
    """
    Fetch up to 20 Google News RSS items for the team.
    Fetches full body for top FULL_BODY_TOP_N links.
    Each dict: {title, summary, body, source}
    Women's articles are filtered out before returning.
    """
    query = f"{team_name} mens college basketball NCAAB"
    url = (
        f"https://news.google.com/rss/search"
        f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    articles = []
    try:
        feed = feedparser.parse(url)
        entries = feed.entries[:20]
        for idx, entry in enumerate(entries):
            title   = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            link    = getattr(entry, "link", "")
            body    = ""
            if idx < FULL_BODY_TOP_N and link:
                body = fetch_article_body(link)
            combined_check = body if body else summary
            if is_womens_article(title, combined_check):
                continue
            articles.append({
                "title":   title,
                "summary": summary,
                "body":    body,
                "source":  "google_news",
            })
    except Exception as exc:
        log.warning("Google News fetch failed for '%s': %s", team_name, exc)
    return articles


def fetch_espn_articles(query: str, limit: int) -> list[dict]:
    """
    Query the ESPN common search API.
    Each dict: {title, summary, source}
    Women's articles are filtered out before returning.
    """
    url = (
        f"https://site.api.espn.com/apis/common/v3/search"
        f"?query={requests.utils.quote(query)}&limit={limit}"
        f"&type=article&sport=mens-college-basketball"
    )
    articles = []
    resp = _safe_get(url)
    if resp is None or resp.status_code != 200:
        return articles
    try:
        data = resp.json()
        # ESPN search response: top-level 'results' list, each item has 'contents'
        results = data.get("results", [])
        for result_group in results:
            contents = result_group.get("contents", [])
            for item in contents:
                title       = item.get("headline", item.get("title", ""))
                description = item.get("description", item.get("summary", ""))
                if is_womens_article(title, description):
                    continue
                articles.append({
                    "title":   title,
                    "summary": description,
                    "source":  "espn",
                })
                if len(articles) >= limit:
                    return articles
    except Exception as exc:
        log.warning("ESPN parse failed for query '%s': %s", query, exc)
    return articles


def fetch_espn_news_articles(team_name: str) -> list[dict]:
    """Fetch 15 ESPN search results for team + college basketball."""
    query = f"{team_name} college basketball"
    return fetch_espn_articles(query, limit=15)


def fetch_espn_targeted_injury(team_name: str) -> list[dict]:
    """Fetch 10 ESPN results for team injury context."""
    query = f"{team_name} injury OR injured OR out"
    return fetch_espn_articles(query, limit=10)


def fetch_espn_targeted_lineup(team_name: str) -> list[dict]:
    """Fetch 10 ESPN results for team lineup context."""
    query = f"{team_name} lineup OR starter OR rotation OR benched"
    return fetch_espn_articles(query, limit=10)

# =============================================================================
# SECTION — Feature Extraction
# =============================================================================

def extract_team_features(team_name: str) -> dict:
    """
    Fetch news from 4 sources and compute all 27 feature scalars for a team.
    Returns a flat dict keyed by feature name. All values are floats.
    """
    # --- Fetch all four article sources ---
    gn_articles      = fetch_google_news_articles(team_name)
    espn_articles    = fetch_espn_news_articles(team_name)
    inj_articles     = fetch_espn_targeted_injury(team_name)
    lineup_articles  = fetch_espn_targeted_lineup(team_name)

    # --- Build text pools ---
    # For each article, the text to use = body (if present) else summary, plus title
    def article_texts(articles: list[dict]) -> list[str]:
        out = []
        for a in articles:
            body    = a.get("body", "").strip()
            summary = a.get("summary", "").strip()
            title   = a.get("title", "").strip()
            content = body if body else summary
            combined = (title + " " + content).strip()
            if combined:
                out.append(combined)
        return out

    def title_texts(articles: list[dict]) -> list[str]:
        return [a.get("title", "") for a in articles if a.get("title", "").strip()]

    gn_texts      = article_texts(gn_articles)
    espn_texts    = article_texts(espn_articles)
    inj_texts     = article_texts(inj_articles)
    lineup_texts  = article_texts(lineup_articles)
    all_texts     = gn_texts + espn_texts + inj_texts + lineup_texts
    all_titles    = title_texts(gn_articles + espn_articles + inj_articles + lineup_articles)

    total_articles      = len(all_texts)
    gn_count            = len(gn_texts)
    espn_count          = len(espn_texts)
    inj_count           = len(inj_texts)
    lineup_count        = len(lineup_texts)

    combined_all        = " ".join(all_texts)
    combined_inj        = " ".join(inj_texts)
    combined_lineup     = " ".join(lineup_texts)

    # --- [A] Sentiment (7 features) ---
    sent_overall = mean_vader(all_texts)
    sent_espn    = mean_vader(espn_texts)
    sent_google  = mean_vader(gn_texts)
    sent_headlines = mean_vader(all_titles)
    sent_recent  = mean_vader(all_texts[-5:]) if all_texts else 0.0

    all_compounds = [analyzer.polarity_scores(t)["compound"] for t in all_texts]
    if all_compounds:
        sent_pct_positive_articles = float(
            sum(1 for c in all_compounds if c > 0.05) / len(all_compounds)
        )
        sent_pct_negative_articles = float(
            sum(1 for c in all_compounds if c < -0.05) / len(all_compounds)
        )
    else:
        sent_pct_positive_articles = 0.0
        sent_pct_negative_articles = 0.0

    # --- [B] Injury (5 features) ---
    # inj_mention_rate: fraction of ALL + injury articles hitting any injury keyword
    combined_for_inj_rate = all_texts  # all articles (includes injury-targeted ones)
    inj_mention_rate = count_keyword_hits(combined_for_inj_rate, INJURY_KEYWORDS)

    # inj_severity_score: max keyword_score across combined injury texts
    inj_severity_score = keyword_score(combined_inj, INJURY_KEYWORDS) if combined_inj else 0.0

    inj_article_count_norm = min(inj_count / 10.0, 1.0)

    inj_key_player_flag = 1.0 if (
        combined_inj and _STAR_PLAYER_INJURY_RE.search(combined_inj)
    ) else 0.0

    inj_sentiment = mean_vader(inj_texts)

    # --- [C] Lineup (5 features) ---
    lineup_change_signal = count_keyword_hits(lineup_texts, LINEUP_KEYWORDS)
    lineup_starter_mention_rate = count_keyword_hits(lineup_texts, ["starter", "starting"])
    lineup_benched_signal = count_keyword_hits(lineup_texts, ["bench", "benched", "moved to bench"])
    lineup_roster_instability = min(
        0.5 * lineup_change_signal + 0.5 * inj_mention_rate, 1.0
    )
    lineup_sentiment = mean_vader(lineup_texts)

    # --- [D] Momentum (6 features) ---
    momentum_win_mention_rate  = count_keyword_hits(all_texts, WIN_KEYWORDS)
    momentum_loss_mention_rate = count_keyword_hits(all_texts, LOSS_KEYWORDS)
    momentum_score             = count_keyword_hits(all_texts, MOMENTUM_KEYWORDS)
    momentum_slump_score       = count_keyword_hits(all_texts, SLUMP_KEYWORDS)
    momentum_net               = momentum_score - momentum_slump_score
    win_rate  = momentum_win_mention_rate
    loss_rate = momentum_loss_mention_rate
    momentum_win_loss_ratio = win_rate / max(loss_rate, 0.01)

    # --- [E] Context (5 features) ---
    ctx_coaching_mention_rate = count_keyword_hits(all_texts, COACHING_KEYWORDS)
    ctx_coaching_instability  = 1.0 if (
        combined_all and _COACHING_CHANGE_RE.search(combined_all)
    ) else 0.0
    ctx_ranking_mention_rate  = count_keyword_hits(all_texts, RANKING_KEYWORDS)
    ctx_fatigue_signal        = count_keyword_hits(all_texts, FATIGUE_KEYWORDS)
    ctx_home_away_context     = count_keyword_hits(all_texts, HOME_AWAY_KEYWORDS)

    # --- [F] Data quality (3 features) ---
    data_total_articles_norm = min(total_articles / 35.0, 1.0)
    data_source_diversity    = (
        (1 if gn_count > 0 else 0)
        + (1 if espn_count > 0 else 0)
        + (1 if inj_count > 0 else 0)
        + (1 if lineup_count > 0 else 0)
    ) / 4.0
    data_confidence = 0.5 * data_total_articles_norm + 0.5 * data_source_diversity

    return {
        # [A] Sentiment
        "sent_overall":                float(sent_overall),
        "sent_espn":                   float(sent_espn),
        "sent_google":                 float(sent_google),
        "sent_headlines":              float(sent_headlines),
        "sent_recent":                 float(sent_recent),
        "sent_pct_positive_articles":  float(sent_pct_positive_articles),
        "sent_pct_negative_articles":  float(sent_pct_negative_articles),
        # [B] Injury
        "inj_mention_rate":            float(inj_mention_rate),
        "inj_severity_score":          float(inj_severity_score),
        "inj_article_count_norm":      float(inj_article_count_norm),
        "inj_key_player_flag":         float(inj_key_player_flag),
        "inj_sentiment":               float(inj_sentiment),
        # [C] Lineup
        "lineup_change_signal":        float(lineup_change_signal),
        "lineup_starter_mention_rate": float(lineup_starter_mention_rate),
        "lineup_benched_signal":       float(lineup_benched_signal),
        "lineup_roster_instability":   float(lineup_roster_instability),
        "lineup_sentiment":            float(lineup_sentiment),
        # [D] Momentum
        "momentum_win_mention_rate":   float(momentum_win_mention_rate),
        "momentum_loss_mention_rate":  float(momentum_loss_mention_rate),
        "momentum_score":              float(momentum_score),
        "momentum_slump_score":        float(momentum_slump_score),
        "momentum_net":                float(momentum_net),
        "momentum_win_loss_ratio":     float(momentum_win_loss_ratio),
        # [E] Context
        "ctx_coaching_mention_rate":   float(ctx_coaching_mention_rate),
        "ctx_coaching_instability":    float(ctx_coaching_instability),
        "ctx_ranking_mention_rate":    float(ctx_ranking_mention_rate),
        "ctx_fatigue_signal":          float(ctx_fatigue_signal),
        "ctx_home_away_context":       float(ctx_home_away_context),
        # [F] Data quality
        "data_total_articles_norm":    float(data_total_articles_norm),
        "data_source_diversity":       float(data_source_diversity),
        "data_confidence":             float(data_confidence),
    }

# =============================================================================
# SECTION — Zero Feature Vector (silent fallback)
# =============================================================================

# All output feature columns in spec order (groups A-E = 28 model features + F = 3 data-quality)
MODEL_FEATURE_NAMES = [
    # [A] Sentiment (7)
    "sent_overall", "sent_espn", "sent_google", "sent_headlines", "sent_recent",
    "sent_pct_positive_articles", "sent_pct_negative_articles",
    # [B] Injury (5)
    "inj_mention_rate", "inj_severity_score", "inj_article_count_norm",
    "inj_key_player_flag", "inj_sentiment",
    # [C] Lineup (5)
    "lineup_change_signal", "lineup_starter_mention_rate", "lineup_benched_signal",
    "lineup_roster_instability", "lineup_sentiment",
    # [D] Momentum (6)
    "momentum_win_mention_rate", "momentum_loss_mention_rate",
    "momentum_score", "momentum_slump_score", "momentum_net",
    "momentum_win_loss_ratio",
    # [E] Context (5)
    "ctx_coaching_mention_rate", "ctx_coaching_instability",
    "ctx_ranking_mention_rate", "ctx_fatigue_signal", "ctx_home_away_context",
    # [F] Data quality (3)
    "data_total_articles_norm", "data_source_diversity", "data_confidence",
]

assert len(MODEL_FEATURE_NAMES) == 31, (
    f"Expected 31 feature columns (A=7 + B=5 + C=5 + D=6 + E=5 + F=3), "
    f"got {len(MODEL_FEATURE_NAMES)}"
)


def zero_feature_vector() -> dict:
    """Return a dict of all 31 feature columns set to 0.0 (neutral values)."""
    return {name: 0.0 for name in MODEL_FEATURE_NAMES}

# =============================================================================
# SECTION — Main Execution
# =============================================================================

def load_team_names(csv_path: str) -> list[str]:
    """Read cbs_games.csv and return sorted list of unique team names."""
    df = pd.read_csv(csv_path)
    home_names = df["home_name"].dropna().tolist()
    away_names = df["away_name"].dropna().tolist()
    unique_teams = sorted(set(home_names + away_names))
    return unique_teams


def run(csv_path: str = CBS_GAMES_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Main entry point.
    1. Load unique team names from cbs_games.csv.
    2. For each team, call extract_team_features() with silent fallback on error.
    3. Save result to sentiment_features.csv.
    Returns the features DataFrame.
    """
    log.info("Loading team names from: %s", csv_path)
    team_names = load_team_names(csv_path)
    total = len(team_names)
    log.info("Found %d unique teams to process.", total)

    records = []
    processed_teams: dict = {}  # in-memory cache — mirrors training notebook convention

    for i, team_name in enumerate(team_names, start=1):
        # Progress header
        print(f"[{i}/{total}] Fetching: {team_name}")

        if team_name in processed_teams:
            feats = processed_teams[team_name]
            print(
                f"  (cached) sent_overall={feats['sent_overall']:.4f}  "
                f"data_confidence={feats['data_confidence']:.4f}"
            )
            records.append({"team_name": team_name, **feats})
            continue

        try:
            feats = extract_team_features(team_name)
        except Exception as exc:
            log.error("extract_team_features() failed for '%s': %s", team_name, exc)
            feats = zero_feature_vector()

        # Enforce numeric types per project convention
        feats = {
            k: float(pd.to_numeric(v, errors="coerce") or 0.0)
            for k, v in feats.items()
        }

        processed_teams[team_name] = feats
        print(
            f"  sent_overall={feats['sent_overall']:.4f}  "
            f"data_confidence={feats['data_confidence']:.4f}"
        )
        records.append({"team_name": team_name, **feats})

        if i < total:
            time.sleep(RATE_LIMIT_SLEEP)

    # Build output DataFrame — column order: team_name + 31 features in spec order
    output_cols = ["team_name"] + MODEL_FEATURE_NAMES
    df_out = pd.DataFrame(records, columns=output_cols)

    # Enforce all feature columns as float64
    for col in MODEL_FEATURE_NAMES:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(0.0)

    df_out.to_csv(output_path, index=False)
    log.info("Saved %d rows x %d columns -> %s", len(df_out), len(df_out.columns), output_path)
    print(f"\nDone. {len(df_out)} teams saved to: {output_path}")
    return df_out


if __name__ == "__main__":
    run()
