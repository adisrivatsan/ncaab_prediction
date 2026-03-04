"""
CBS Sports NCAAB Scoreboard Scraper
Scrapes final game scores for NCAA Men's D-I Basketball over a date range.
Window: 3 weeks ending yesterday (computed dynamically at runtime).
"""

from __future__ import annotations

import time
import pandas as pd
from datetime import date, timedelta

import requests
from bs4 import BeautifulSoup

# =============================================================================
# CONFIG
# =============================================================================

END_DATE   = date.today() - timedelta(days=1)       # yesterday
START_DATE = END_DATE - timedelta(days=21)           # 3 weeks back

BASE_URL      = "https://www.cbssports.com/college-basketball/scoreboard/FBS/{date_str}/"
REQUEST_TIMEOUT  = 20          # seconds
RETRY_ATTEMPTS   = 3
RETRY_BACKOFF    = [2, 5, 10]  # seconds between retries
POLITENESS_SLEEP = 1.5         # seconds between page requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# =============================================================================
# SCRAPER
# =============================================================================

def _fetch_page(url: str) -> str | None:
    """Fetch a URL with retry/backoff. Returns HTML text or None on failure."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            print(f"  ⚠️  HTTP {resp.status_code} on attempt {attempt + 1} — {url}")
        except requests.exceptions.Timeout:
            print(f"  ⚠️  Timeout on attempt {attempt + 1} — {url}")
        except requests.exceptions.RequestException as exc:
            print(f"  ⚠️  Request error on attempt {attempt + 1}: {exc}")

        if attempt < RETRY_ATTEMPTS - 1:
            wait = RETRY_BACKOFF[attempt]
            print(f"  ↩️  Retrying in {wait}s …")
            time.sleep(wait)

    return None


def get_games_for_date_cbs(game_date: date) -> list[dict]:
    """
    Scrape CBS Sports scoreboard for one date.
    Returns a list of game dicts (finals only). Non-final cards are silently skipped.
    Card-level parse errors are caught and logged; the card is then skipped.

    Output keys per game:
        date, away_name, away_score, home_name, home_score, status,
        home_team_won, winner_name, winner_score, loser_name, loser_score, score_diff
    """
    date_str = game_date.strftime("%Y%m%d")
    url = BASE_URL.format(date_str=date_str)

    html = _fetch_page(url)
    if html is None:
        print(f"❌ Failed to fetch {date_str}")
        return []

    soup = BeautifulSoup(html, "lxml")
    cards = soup.find_all("div", class_="single-score-card")
    print(f"  📋 {game_date.isoformat()} — {len(cards)} cards found")

    games    = []
    seen_abbrevs = set()
    finals_count = 0

    for card in cards:
        try:
            # ── Game identity ─────────────────────────────────────────────
            abbrev = card.get("data-abbrev", "")
            if not abbrev:
                print(f"  ⚠️  Card missing data-abbrev — skipping")
                continue

            # Deduplication: one row per game (Business Rule 2)
            if abbrev in seen_abbrevs:
                print(f"  ⚠️  Duplicate data-abbrev '{abbrev}' — skipping duplicate")
                continue
            seen_abbrevs.add(abbrev)

            # ── Status filter — "Final means final" (Business Rule 1) ─────
            status_tag = card.find("div", class_="game-status")
            status_raw = status_tag.get_text(strip=True) if status_tag else ""
            if "final" not in status_raw.lower():
                continue  # non-final: silently skip
            finals_count += 1

            # ── Team rows ─────────────────────────────────────────────────
            # Static HTML renders both rows with class="tiedGame".
            # Row order (away=0, home=1) is set by data-abbrev, not by CSS class.
            tr_rows = [tr for tr in card.find_all("tr") if tr.get("class") is not None]

            if len(tr_rows) < 2:
                print(f"  ⚠️  [{abbrev}] Expected 2 team rows, found {len(tr_rows)} — skipping")
                continue

            away_row, home_row = tr_rows[0], tr_rows[1]

            def parse_team_name(row) -> str | None:
                tag = row.find("a", class_="team-name-link")
                return tag.get_text(strip=True) if tag else None

            def parse_score(row) -> int | None:
                tag = row.find("td", class_="total")
                if tag is None:
                    return None
                try:
                    return int(tag.get_text(strip=True))
                except (ValueError, TypeError):
                    return None

            away_name  = parse_team_name(away_row)
            home_name  = parse_team_name(home_row)
            away_score = parse_score(away_row)
            home_score = parse_score(home_row)

            if not away_name or not home_name:
                print(f"  ⚠️  [{abbrev}] Could not parse team names — skipping")
                continue

            # ── Derived columns ───────────────────────────────────────────
            scores_present = (away_score is not None) and (home_score is not None)

            home_team_won = None
            winner_name   = None
            winner_score  = None
            loser_name    = None
            loser_score   = None
            score_diff    = None

            if scores_present:
                score_diff = home_score - away_score
                if home_score > away_score:
                    home_team_won = 1
                    winner_name, winner_score = home_name, home_score
                    loser_name,  loser_score  = away_name, away_score
                elif away_score > home_score:
                    home_team_won = 0
                    winner_name, winner_score = away_name, away_score
                    loser_name,  loser_score  = home_name, home_score
                # tied: home_team_won remains None

            games.append({
                "date":          game_date.isoformat(),
                "away_name":     away_name,
                "away_score":    away_score,
                "home_name":     home_name,
                "home_score":    home_score,
                "status":        "Final",
                "home_team_won": home_team_won,
                "winner_name":   winner_name,
                "winner_score":  winner_score,
                "loser_name":    loser_name,
                "loser_score":   loser_score,
                "score_diff":    score_diff,
            })

        except Exception as exc:
            print(f"  ❌ Card-level error [{card.get('data-abbrev', '?')}]: {exc}")
            continue

    print(f"  ✅ {finals_count} finals parsed for {game_date.isoformat()}")
    return games


# =============================================================================
# MAIN LOOP
# =============================================================================

def scrape_date_range(start: date, end: date) -> pd.DataFrame:
    """
    Iterate day-by-day from start to end (inclusive), scrape CBS, and
    return a single consolidated DataFrame.
    """
    all_games: list[dict] = []
    total_days = (end - start).days + 1

    print(f"\n{'='*60}")
    print(f"CBS NCAAB Scraper — {start.isoformat()} to {end.isoformat()}")
    print(f"{'='*60}\n")

    current = start
    while current <= end:
        games = get_games_for_date_cbs(current)
        all_games.extend(games)
        current += timedelta(days=1)
        if current <= end:
            time.sleep(POLITENESS_SLEEP)

    if not all_games:
        print("\n⚠️  No games collected — returning empty DataFrame.")
        return pd.DataFrame(columns=[
            "date", "away_name", "away_score", "home_name", "home_score",
            "status", "home_team_won", "winner_name", "winner_score",
            "loser_name", "loser_score", "score_diff",
        ])

    df = pd.DataFrame(all_games)

    # Enforce numeric types (project convention)
    for col in ["away_score", "home_score", "home_team_won",
                "winner_score", "loser_score", "score_diff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Date range : {start.isoformat()} → {end.isoformat()} ({total_days} days)")
    print(f"  Total games: {len(df)}")
    print(f"  Unique teams: {len(set(df['home_name'].tolist() + df['away_name'].tolist()))}")
    print(f"  Games with scores: {df['score_diff'].notna().sum()}")
    print(f"{'='*60}\n")

    return df


if __name__ == "__main__":
    df = scrape_date_range(START_DATE, END_DATE)

    if not df.empty:
        output_path = "cbs_games.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {len(df)} rows → {output_path}")
        print(df.head(10).to_string(index=False))
