"""
Vegas Odds Fetcher — The Odds API
==================================
Fetches current moneylines + spreads for NCAA Men's Basketball games
from The Odds API (https://the-odds-api.com).

Requires environment variable:
    ODDS_API_KEY — your API key from the-odds-api.com

Used by predict_today.py Step 7 to enrich top_picks with real Vegas lines.
If ODDS_API_KEY is not set, fetch_odds() returns [] — graceful no-op.

Output:
    odds_name_mapping.csv — audit CSV of Odds API name → CBS name mapping
    (No ratings CSV — odds are transient, not persisted between runs)
"""

from __future__ import annotations

import csv
import difflib
import json
import logging
import os
import sys
import time
from typing import Optional

import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL  = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/"

ODDS_MAPPING_PATH  = os.path.join(_SCRIPT_DIR, "odds_name_mapping.csv")
ODDS_SNAPSHOT_PATH = os.path.join(_SCRIPT_DIR, "odds_snapshot.json")

REQUEST_TIMEOUT = 15
RETRY_ATTEMPTS  = 2
FUZZY_CUTOFF    = 0.6

# Preferred bookmakers for averaging (checked against bk["key"].lower())
PREFERRED_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars"}

# Hardcoded overrides for known problem cases: Odds API full name → CBS short name
OVERRIDE_MAP: dict[str, str] = {
    "Eastern Kentucky Colonels":        "E. Kentucky",
    "Western Kentucky Hilltoppers":     "W. Kentucky",
    "Southern Illinois Salukis":        "S. Illinois",
    "Northern Illinois Huskies":        "N. Illinois",
    "Western Michigan Broncos":         "W. Michigan",
    "Eastern Michigan Eagles":          "E. Michigan",
    "Central Michigan Chippewas":       "C. Michigan",
    "Middle Tennessee Blue Raiders":    "Mid. Tennessee",
    "UT Martin Skyhawks":               "UT Martin",
    "Louisiana Monroe Warhawks":        "ULM",
    "North Carolina A&T Aggies":        "NC A&T",
    "Miami (OH) RedHawks":              "Miami (OH)",
    "Miami Hurricanes":                 "Miami (FL)",
    "Louisiana Lafayette Ragin Cajuns": "Louisiana",
    "Arkansas-Little Rock Trojans":     "UALR",
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# =============================================================================
# HTTP UTILITIES
# =============================================================================

def _fetch_odds_response() -> Optional[requests.Response]:
    """
    GET The Odds API endpoint with retry/backoff.
    Returns the Response on HTTP 200, or None on failure.
    Logs x-requests-remaining header for quota monitoring in GitHub Actions.
    """
    if not ODDS_API_KEY:
        log.warning("ODDS_API_KEY is not set — skipping Vegas odds fetch")
        return None

    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    "h2h,spreads",
        "oddsFormat": "american",
    }

    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(ODDS_API_URL, params=params, timeout=REQUEST_TIMEOUT)
            remaining = resp.headers.get("x-requests-remaining", "unknown")
            log.info(
                "Odds API — HTTP %d  |  requests remaining: %s",
                resp.status_code, remaining,
            )
            if resp.status_code == 200:
                return resp
            if resp.status_code in (422, 429):
                log.warning("Odds API HTTP %d — skipping odds fetch", resp.status_code)
                return None
            log.warning("Odds API HTTP %d on attempt %d", resp.status_code, attempt + 1)
        except requests.exceptions.Timeout:
            log.warning("Odds API timeout on attempt %d", attempt + 1)
        except requests.exceptions.RequestException as exc:
            log.warning("Odds API request error on attempt %d: %s", attempt + 1, exc)

        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(2)

    log.warning("All Odds API attempts failed — proceeding without Vegas lines")
    return None


# =============================================================================
# PARSE ODDS RESPONSE
# =============================================================================

def _avg_outcome_ml(outcomes: list[dict], team_name: str) -> Optional[float]:
    """Average American moneyline across all outcomes matching team_name."""
    vals = [float(o["price"]) for o in outcomes if o.get("name", "").strip() == team_name]
    return round(sum(vals) / len(vals)) if vals else None


def _avg_spread(outcomes: list[dict], team_name: str) -> Optional[float]:
    """Average spread point value for team_name across all spread outcomes."""
    vals = [
        float(o["point"])
        for o in outcomes
        if o.get("name", "").strip() == team_name and "point" in o
    ]
    return round(sum(vals) / len(vals), 1) if vals else None


def _filter_bookmakers(bookmakers: list[dict]) -> list[dict]:
    """Return preferred bookmakers if available; otherwise return all."""
    preferred = [b for b in bookmakers if b.get("key", "").lower() in PREFERRED_BOOKS]
    return preferred if preferred else bookmakers


def fetch_odds() -> list[dict]:
    """
    Fetch current NCAAB moneylines + spreads from The Odds API.

    Returns a list of game dicts:
        [{
            "home_team_raw": str,   # full team name from Odds API
            "away_team_raw": str,
            "home_ml":       str,   # e.g. "-145" or "+120" or "N/A"
            "away_ml":       str,
            "spread":        str,   # home team spread, e.g. "-3.5" or "N/A"
        }, ...]

    Returns [] if ODDS_API_KEY is not set or on HTTP error (graceful no-op).
    Moneylines are averaged across preferred bookmakers (DK/FD/BetMGM/Caesars),
    falling back to all available books if none of the preferred are present.
    """
    resp = _fetch_odds_response()
    if resp is None:
        return []

    try:
        games_data = resp.json()
    except Exception as exc:
        log.warning("Odds API JSON parse error: %s", exc)
        return []

    if not isinstance(games_data, list):
        log.warning("Odds API unexpected response format — expected list, got %s", type(games_data))
        return []

    def _fmt_ml(v: Optional[float]) -> str:
        if v is None:
            return "N/A"
        v_int = int(round(v))
        return f"+{v_int}" if v_int > 0 else str(v_int)

    def _fmt_spread(v: Optional[float]) -> str:
        if v is None:
            return "N/A"
        return f"+{v}" if v > 0 else str(v)

    results = []
    for game in games_data:
        home_raw = game.get("home_team", "").strip()
        away_raw = game.get("away_team", "").strip()
        if not home_raw or not away_raw:
            continue

        bookmakers = _filter_bookmakers(game.get("bookmakers", []))

        h2h_outcomes: list[dict]    = []
        spread_outcomes: list[dict] = []
        for bk in bookmakers:
            for market in bk.get("markets", []):
                key = market.get("key", "")
                if key == "h2h":
                    h2h_outcomes.extend(market.get("outcomes", []))
                elif key == "spreads":
                    spread_outcomes.extend(market.get("outcomes", []))

        results.append({
            "home_team_raw": home_raw,
            "away_team_raw": away_raw,
            "home_ml":       _fmt_ml(_avg_outcome_ml(h2h_outcomes, home_raw)),
            "away_ml":       _fmt_ml(_avg_outcome_ml(h2h_outcomes, away_raw)),
            "spread":        _fmt_spread(_avg_spread(spread_outcomes, home_raw)),
        })

    log.info("Odds API — parsed %d games", len(results))
    return results


# =============================================================================
# ODDS SNAPSHOT — movement tracking across runs
# =============================================================================

def _load_snapshot() -> dict:
    """
    Load the previous odds snapshot from disk.
    Returns dict[cbs_home_name → {home_ml, away_ml, spread}], or {} if not found.
    """
    if not os.path.isfile(ODDS_SNAPSHOT_PATH):
        return {}
    try:
        with open(ODDS_SNAPSHOT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("games", {})
    except Exception as exc:
        log.warning("Could not load odds snapshot: %s", exc)
        return {}


def _save_snapshot(lookup: dict) -> None:
    """
    Save the current odds lookup to disk as the new snapshot.
    Stores only home_ml / away_ml per game (spread not needed for movement).
    """
    try:
        snapshot = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "games": {
                cbs_name: {
                    "home_ml": entry["home_ml"],
                    "away_ml": entry["away_ml"],
                }
                for cbs_name, entry in lookup.items()
            },
        }
        with open(ODDS_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        log.info("Odds snapshot saved -> %s  (%d games)", ODDS_SNAPSHOT_PATH, len(lookup))
    except Exception as exc:
        log.warning("Could not save odds snapshot: %s", exc)


def _movement_str(prev_ml: Optional[str], curr_ml: Optional[str]) -> str:
    """
    Build a human-readable movement string.
    Returns '—' if unchanged or no previous data, 'prev → curr' if moved.
    """
    if not prev_ml or prev_ml == "N/A" or not curr_ml or curr_ml == "N/A":
        return "—"
    if prev_ml == curr_ml:
        return "—"
    return f"{prev_ml} → {curr_ml}"


# =============================================================================
# NAME MAPPING
# =============================================================================

def _strip_mascot(name: str) -> str:
    """Take first 2 words of Odds API name to improve fuzzy matching with CBS short names."""
    parts = name.strip().split()
    return " ".join(parts[:2]) if len(parts) >= 2 else name


def build_odds_lookup(
    odds_games: list[dict],
    cbs_names: list[str],
) -> dict[str, dict]:
    """
    Map Odds API game entries to CBS team names, then build a lookup keyed by
    CBS home team name.

    Matching strategy for each Odds API team name (applied in order):
        1. Hardcoded override (OVERRIDE_MAP)
        2. Exact match (case-sensitive)
        3. Exact match (case-insensitive)
        4. difflib fuzzy match on full name (cutoff=FUZZY_CUTOFF)
        5. difflib fuzzy match on first-2-word prefix (improves short-name matching)
        6. Case-insensitive fuzzy

    Writes odds_name_mapping.csv for audit.

    Returns:
        dict[cbs_home_name → {
            "home_ml":  str,   # e.g. "-145" or "N/A"
            "away_ml":  str,
            "spread":   str,   # home team spread
        }]
    """
    if not odds_games:
        return {}

    cbs_lower_map: dict[str, str] = {n.lower(): n for n in cbs_names}
    mapping_rows: list[dict] = []

    def _map_name(raw: str) -> tuple[str, str]:
        """Map a single Odds API raw name to (cbs_name, match_type)."""
        # 1. Override
        if raw in OVERRIDE_MAP:
            cbs = OVERRIDE_MAP[raw]
            if cbs in cbs_names:
                return cbs, "override"

        # 2. Exact
        if raw in cbs_names:
            return raw, "exact"

        # 3. Case-insensitive exact
        raw_lower = raw.lower()
        if raw_lower in cbs_lower_map:
            return cbs_lower_map[raw_lower], "exact_ci"

        # 4. Fuzzy on full name
        matches = difflib.get_close_matches(raw, cbs_names, n=1, cutoff=FUZZY_CUTOFF)
        if matches:
            return matches[0], "fuzzy"

        # 5. Fuzzy on first-2-word prefix
        prefix = _strip_mascot(raw)
        prefix_matches = difflib.get_close_matches(prefix, cbs_names, n=1, cutoff=FUZZY_CUTOFF)
        if prefix_matches:
            return prefix_matches[0], "fuzzy_prefix"

        # 6. Case-insensitive fuzzy
        matches_ci = difflib.get_close_matches(
            raw_lower, list(cbs_lower_map.keys()), n=1, cutoff=FUZZY_CUTOFF,
        )
        if matches_ci:
            return cbs_lower_map[matches_ci[0]], "fuzzy"

        return "", "unmatched"

    # Load previous snapshot for movement comparison
    prev_snapshot = _load_snapshot()

    lookup: dict[str, dict] = {}
    for game in odds_games:
        home_raw = game["home_team_raw"]
        away_raw = game["away_team_raw"]

        home_cbs, home_mtype = _map_name(home_raw)
        away_cbs, away_mtype = _map_name(away_raw)

        mapping_rows.append({"odds_api_name": home_raw, "cbs_name": home_cbs, "match_type": home_mtype})
        mapping_rows.append({"odds_api_name": away_raw, "cbs_name": away_cbs, "match_type": away_mtype})

        if home_cbs and away_cbs:
            prev = prev_snapshot.get(home_cbs, {})
            lookup[home_cbs] = {
                "home_ml":          game["home_ml"],
                "away_ml":          game["away_ml"],
                "spread":           game["spread"],
                "home_ml_movement": _movement_str(prev.get("home_ml"), game["home_ml"]),
                "away_ml_movement": _movement_str(prev.get("away_ml"), game["away_ml"]),
            }

    # Write audit CSV
    try:
        with open(ODDS_MAPPING_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["odds_api_name", "cbs_name", "match_type"])
            writer.writeheader()
            writer.writerows(mapping_rows)
        log.info("Odds mapping saved  -> %s  (%d rows)", ODDS_MAPPING_PATH, len(mapping_rows))
    except Exception as exc:
        log.warning("Could not write odds mapping CSV: %s", exc)

    matched = sum(1 for r in mapping_rows if r["match_type"] != "unmatched")
    log.info(
        "Odds name mapping   : %d/%d teams matched  (%d games covered)",
        matched, len(mapping_rows), len(lookup),
    )

    # Save snapshot for next run's movement comparison
    _save_snapshot(lookup)

    return lookup
