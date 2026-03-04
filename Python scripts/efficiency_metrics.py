"""
NCAAB Efficiency Metrics Collector
====================================
Collects four-factor efficiency metrics for all D-I teams:
  - eFG%  : (FGM + 0.5 * 3PM) / FGA
  - TOV%  : TOV / (FGA + 0.44 * FTA + TOV)
  - ORB%  : ORB / (ORB + opponent DRB)
  - FTR   : FTA / FGA

Also collects win_loss_overall and win_loss_conference where available.

Primary source  : Barttorvik JSON API   (https://barttorvik.com/2026_team_results.json)
Fallback 1      : Sports Reference advanced school stats HTML table
Fallback 2      : Sports Reference basic school stats HTML table (manual formula)

Outputs:
  efficiency_metrics.csv   -- one row per source team, with CBS name mapped in
  team_name_mapping.csv    -- source_name -> cbs_name with match confidence
"""

from __future__ import annotations

import csv
import difflib
import json
import logging
import math
import os
import time
from datetime import date
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# =============================================================================
# CONFIG
# =============================================================================

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

AS_OF_DATE     = date.today().isoformat()          # "2026-03-03"
SCHEMA_VERSION = "1.0"

CBS_CSV_PATH        = os.path.join(_PROJECT_DIR, "cbs_games.csv")
OUTPUT_METRICS_PATH = os.path.join(_SCRIPT_DIR, "efficiency_metrics.csv")
OUTPUT_MAPPING_PATH = os.path.join(_SCRIPT_DIR, "team_name_mapping.csv")

# Barttorvik
TORVIK_JSON_URL = "https://barttorvik.com/2026_team_results.json"
TORVIK_HTML_URL = "https://barttorvik.com/trank.php#"

# Sports Reference
SREF_ADVANCED_URL = (
    "https://www.sports-reference.com/cbb/seasons/men/2026-advanced-school-stats.html"
)
SREF_BASIC_URL = (
    "https://www.sports-reference.com/cbb/seasons/men/2026-school-stats.html"
)

REQUEST_TIMEOUT  = 20
RETRY_ATTEMPTS   = 3
RETRY_BACKOFF    = [3, 7, 15]
POLITENESS_SLEEP = 2.0          # seconds between page fetches

FUZZY_CUTOFF     = 0.60         # minimum difflib ratio to accept a match
FUZZY_N          = 1            # top-N candidates from get_close_matches

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
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# HTTP HELPERS
# =============================================================================

def _fetch(url: str, as_json: bool = False):
    """Fetch URL with retry/backoff. Returns text, parsed JSON, or None."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                if as_json:
                    return resp.json()
                return resp.text
            log.warning("HTTP %s on attempt %d — %s", resp.status_code, attempt + 1, url)
        except requests.exceptions.Timeout:
            log.warning("Timeout on attempt %d — %s", attempt + 1, url)
        except requests.exceptions.JSONDecodeError as exc:
            log.warning("JSON decode error on attempt %d: %s", attempt + 1, exc)
            return None
        except requests.exceptions.RequestException as exc:
            log.warning("Request error on attempt %d: %s", attempt + 1, exc)

        if attempt < RETRY_ATTEMPTS - 1:
            wait = RETRY_BACKOFF[attempt]
            log.info("Retrying in %ds ...", wait)
            time.sleep(wait)

    log.error("All %d attempts failed for %s", RETRY_ATTEMPTS, url)
    return None

# =============================================================================
# CBS TEAM NAMES
# =============================================================================

def load_cbs_team_names(csv_path: str) -> list[str]:
    """
    Read home_name and away_name columns from the CBS games CSV and return
    a deduplicated, sorted list of unique team name strings exactly as
    they appear in the CSV (no normalization).
    """
    df = pd.read_csv(csv_path)
    names = set(df["home_name"].dropna().tolist()) | set(df["away_name"].dropna().tolist())
    names = sorted(names)
    log.info("Loaded %d unique CBS team names from %s", len(names), csv_path)
    return names

# =============================================================================
# FUZZY NAME MATCHING
# =============================================================================

def _normalize_for_matching(name: str) -> str:
    """
    Lowercase, strip, remove trailing punctuation like periods in abbreviations,
    expand a few very common CBS shorthand patterns to improve match rate.
    This normalization is used ONLY internally for matching — the original
    strings are preserved in all outputs.
    """
    s = name.lower().strip()
    # Remove trailing dot (e.g. "Boston U." -> "boston u")
    s = s.rstrip(".")
    # Collapse multiple spaces
    s = " ".join(s.split())
    # A handful of explicit alias expansions for common CBS shorthands
    _aliases = {
        "n. carolina":     "north carolina",
        "s. carolina":     "south carolina",
        "n. illinois":     "northern illinois",
        "s. illinois":     "southern illinois",
        "n. iowa":         "northern iowa",
        "e. michigan":     "eastern michigan",
        "w. michigan":     "western michigan",
        "c. michigan":     "central michigan",
        "n. kentucky":     "northern kentucky",
        "e. kentucky":     "eastern kentucky",
        "w. kentucky":     "western kentucky",
        "s. dak. st":      "south dakota state",
        "n. dakota st":    "north dakota state",
        "ga. tech":        "georgia tech",
        "ga. southern":    "georgia southern",
        "ark.-pine bluff": "arkansas pine bluff",
        "bethune-cook":    "bethune cookman",
        "boston u":        "boston university",
        "va. tech":        "virginia tech",
        "la. tech":        "louisiana tech",
        "lmu":             "loyola marymount",
        "lbsu":            "long beach state",
        "pfw":             "purdue fort wayne",
        "iui":             "indiana university indianapolis",
        "fgcu":            "florida gulf coast",
        "fau":             "florida atlantic",
        "fiu":             "florida international",
        "uic":             "illinois chicago",
        "uab":             "alabama birmingham",
        "uncg":            "unc greensboro",
        "uncw":            "unc wilmington",
        "unc-ash":         "unc asheville",
        "siue":            "southern illinois edwardsville",
        "utsa":            "texas san antonio",
        "utep":            "texas el paso",
        "ucsb":            "uc santa barbara",
        "ucf":             "central florida",
        "vcu":             "virginia commonwealth",
        "csn":             "cal state northridge",
        "csnorthridge":    "cal state northridge",
        "e. washington":   "eastern washington",
        "w. georgia":      "western georgia",
        "n. alabama":      "north alabama",
        "so. miss":        "southern mississippi",
        "so. utah":        "southern utah",
        "so. indiana":     "southern indiana",
        "cent. arkansas":  "central arkansas",
        "cal-baker":       "cal state bakersfield",
        "tx a&m-cc":       "texas a&m corpus christi",
        "ut-rio grande valley": "texas rio grande valley",
        "ut martin":       "tennessee martin",
        "tenn. tech":      "tennessee tech",
        "se louisiana":    "southeastern louisiana",
        "se missouri st":  "southeast missouri state",
        "sf austin":       "stephen f. austin",
        "n. mex. st":      "new mexico state",
        "jax. state":      "jacksonville state",
        "n.j. tech":       "new jersey institute of technology",
        "loyola-md":       "loyola maryland",
        "loyola chi":      "loyola chicago",
        "st. bona":        "st. bonaventure",
        "miami (ohio)":    "miami ohio",
        "st. thomas (mn)": "st. thomas",
        "mt st mary's":    "mount st. mary's",
        "s. dak. st.":     "south dakota state",
    }
    return _aliases.get(s, s)


def build_fuzzy_mapping(
    source_names: list[str],
    cbs_names: list[str],
) -> dict[str, tuple[str, float]]:
    """
    For each source_name, find the best-matching CBS name via difflib.
    Returns {source_name: (cbs_name, confidence)} where confidence in [0, 1].
    Only matches above FUZZY_CUTOFF are retained; others map to ("", 0.0).
    """
    cbs_normalized = [_normalize_for_matching(n) for n in cbs_names]
    mapping: dict[str, tuple[str, float]] = {}

    for src in source_names:
        src_norm = _normalize_for_matching(src)

        # Exact normalized match first (fast path)
        if src_norm in cbs_normalized:
            idx = cbs_normalized.index(src_norm)
            mapping[src] = (cbs_names[idx], 1.0)
            continue

        # Fuzzy match
        matches = difflib.get_close_matches(
            src_norm, cbs_normalized, n=FUZZY_N, cutoff=FUZZY_CUTOFF
        )
        if matches:
            best_norm = matches[0]
            idx = cbs_normalized.index(best_norm)
            confidence = difflib.SequenceMatcher(None, src_norm, best_norm).ratio()
            mapping[src] = (cbs_names[idx], round(confidence, 4))
        else:
            mapping[src] = ("", 0.0)

    return mapping

# =============================================================================
# METRIC VALIDATION
# =============================================================================

def _validate_metric(name: str, team: str, value: Optional[float]) -> Optional[float]:
    """
    Return value if it is a finite float in [0, 1].
    Log a warning and return NaN if out of range or non-finite.
    Return NaN silently if value is None.
    """
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        log.warning("Non-numeric %s for team '%s': %r", name, team, value)
        return float("nan")
    if math.isnan(v) or math.isinf(v):
        return float("nan")
    if not (0.0 <= v <= 1.0):
        log.warning(
            "Out-of-range %s=%.4f for team '%s' (expected [0,1]) — setting NaN",
            name, v, team
        )
        return float("nan")
    return v

# =============================================================================
# SOURCE 1: BARTTORVIK JSON
# =============================================================================

def _parse_torvik_json(data) -> list[dict]:
    """
    Parse the Barttorvik JSON payload into a list of metric dicts.

    The 2026_team_results.json endpoint returns a list of lists or list of dicts.
    Known field positions when the payload is a list of lists (as of 2025 season):
      Index 0  : team name
      Index 1  : conference
      Index 4  : wins (season)
      Index 5  : losses (season)
      Index 11 : eFG% (offensive) -- stored as a decimal e.g. 0.512
      Index 12 : eFG% allowed
      Index 13 : TOV% (offensive turnover rate)
      Index 14 : TOV% allowed
      Index 15 : ORB% (offensive rebounding rate)
      Index 16 : DRB% (defensive rebounding rate)
      Index 17 : FTR  (free throw rate = FTA/FGA, offensive)
      Index 18 : FTR allowed

    If the structure is a list of dicts, we try known key names.
    If either approach fails, we fall back to attempting common key names.

    Returns list of dicts with keys:
        team_id, efg_pct, tov_pct, orb_pct, ftr,
        win_loss_overall, win_loss_conference
    """
    records = []

    if not data:
        return records

    # Determine payload type
    first = data[0] if isinstance(data, list) and len(data) > 0 else None

    if isinstance(first, dict):
        # List of dicts — use key names
        # Try common Torvik key variants
        for row in data:
            team_name = (
                row.get("team")
                or row.get("team_name")
                or row.get("TeamName")
                or ""
            )
            if not team_name:
                continue

            conf = row.get("conf") or row.get("conference") or ""
            wins = row.get("wins") or row.get("w") or ""
            losses = row.get("losses") or row.get("l") or ""

            efg  = row.get("efg") or row.get("eFG") or row.get("efg_pct") or row.get("oefg")
            tov  = row.get("tov") or row.get("tov_pct") or row.get("otov") or row.get("to_pct")
            orb  = row.get("orb") or row.get("orb_pct") or row.get("oor") or row.get("or_pct")
            ftr  = row.get("ftr") or row.get("ft_rate") or row.get("oftr")

            wl_overall = f"{wins}-{losses}" if wins and losses else ""

            records.append({
                "team_id":             str(team_name).strip(),
                "efg_pct":             efg,
                "tov_pct":             tov,
                "orb_pct":             orb,
                "ftr":                 ftr,
                "win_loss_overall":    wl_overall,
                "win_loss_conference": str(conf).strip(),
            })

    elif isinstance(first, list):
        # List of lists — positional indexing based on observed Torvik layout
        # We probe the first row for heuristics to determine column layout.
        # Torvik's trank API for 2026 typically returns 40+ element arrays.
        for row in data:
            try:
                if len(row) < 10:
                    continue

                team_name = str(row[0]).strip()
                if not team_name:
                    continue

                # Attempt positional parsing — indices based on Torvik 2025/2026 API
                # The exact positions vary slightly by season; we use heuristics.
                # Layout observed: [team, conf, g, w, l, adj_o, adj_d, adj_em, ...]
                # Four-factor columns typically start around index 11-18.

                conf = str(row[1]).strip() if len(row) > 1 else ""

                # Try to extract W-L from positions 3 and 4
                wins   = row[3] if len(row) > 3 else None
                losses = row[4] if len(row) > 4 else None
                wl_overall = (
                    f"{int(wins)}-{int(losses)}"
                    if wins is not None and losses is not None
                    else ""
                )

                # Four Factors: probe indices 11-18 (Torvik 2026 layout)
                efg = row[11] if len(row) > 11 else None
                tov = row[13] if len(row) > 13 else None
                orb = row[15] if len(row) > 15 else None
                ftr = row[17] if len(row) > 17 else None

                # Torvik stores these as percentages (e.g., 51.2) not decimals (0.512)
                # Detect and convert: if value > 1, divide by 100
                def maybe_pct(v):
                    if v is None:
                        return None
                    try:
                        f = float(v)
                        return f / 100.0 if f > 1.0 else f
                    except (TypeError, ValueError):
                        return None

                records.append({
                    "team_id":             team_name,
                    "efg_pct":             maybe_pct(efg),
                    "tov_pct":             maybe_pct(tov),
                    "orb_pct":             maybe_pct(orb),
                    "ftr":                 maybe_pct(ftr),
                    "win_loss_overall":    wl_overall,
                    "win_loss_conference": conf,
                })
            except (IndexError, TypeError, ValueError) as exc:
                log.warning("Error parsing Torvik row: %s — %r", exc, row[:5])
                continue
    else:
        log.warning("Unexpected Torvik JSON structure: top-level type %s", type(data))

    return records


def fetch_barttorvik() -> list[dict]:
    """
    Attempt to fetch four-factor data from Barttorvik JSON API.
    Returns list of metric dicts, or empty list on failure.
    """
    log.info("Fetching Barttorvik JSON: %s", TORVIK_JSON_URL)
    data = _fetch(TORVIK_JSON_URL, as_json=True)
    if data is None:
        log.warning("Barttorvik JSON fetch returned None.")
        return []

    records = _parse_torvik_json(data)
    log.info("Barttorvik JSON: parsed %d team records.", len(records))
    return records

# =============================================================================
# SOURCE 2: SPORTS REFERENCE ADVANCED STATS (HTML)
# =============================================================================

def _parse_sref_advanced_html(html: str) -> list[dict]:
    """
    Parse Sports Reference advanced school stats table.
    Target table id: 'div_adv_school_stats' or 'adv_school_stats'.
    Columns of interest (header text):
      School, G, W, L, Conf W, Conf L, eFG%, TOV%, ORB%, FTR
    Sports Reference stores these as plain decimals (e.g., 0.512).
    """
    soup = BeautifulSoup(html, "lxml")
    records = []

    # Locate the advanced stats table — Sports Reference wraps it in a div
    # with id="div_adv_school_stats"; the table itself has id="adv_school_stats"
    table = soup.find("table", {"id": "adv_school_stats"})
    if table is None:
        # Try without the div wrapper
        table = soup.find("table", id=lambda x: x and "adv" in x.lower())
    if table is None:
        log.warning("Could not find advanced stats table in Sports Reference HTML.")
        return records

    # Parse header row(s) — Sports Reference uses multiple <thead> <tr> rows
    headers = []
    thead = table.find("thead")
    if thead:
        # Last header row typically has the actual column names
        header_rows = thead.find_all("tr")
        if header_rows:
            headers = [
                th.get_text(strip=True).lower()
                for th in header_rows[-1].find_all(["th", "td"])
            ]

    log.info("Sports Reference advanced table headers (first 25): %s", headers[:25])

    # Map column names to indices — Sports Reference uses specific names
    col_map = {}
    for i, h in enumerate(headers):
        h_clean = h.replace("%", "_pct").replace(" ", "_").strip("_")
        col_map[h_clean] = i
        col_map[h] = i   # also keep raw

    # Column name variants across seasons
    school_idx = (
        col_map.get("school")
        or col_map.get("school_name")
        or 0
    )
    w_idx      = col_map.get("w")   or col_map.get("wins")
    l_idx      = col_map.get("l")   or col_map.get("losses")
    cw_idx     = col_map.get("w.1") or col_map.get("conf_w") or col_map.get("cw")
    cl_idx     = col_map.get("l.1") or col_map.get("conf_l") or col_map.get("cl")
    efg_idx    = col_map.get("efg_pct") or col_map.get("efg%") or col_map.get("efg")
    tov_idx    = col_map.get("tov_pct") or col_map.get("tov%") or col_map.get("to_pct")
    orb_idx    = col_map.get("orb_pct") or col_map.get("orb%") or col_map.get("or_pct")
    ftr_idx    = col_map.get("ftr")     or col_map.get("ft_rate") or col_map.get("ftr_pct")

    log.info(
        "Column index mapping — school:%s w:%s l:%s efg:%s tov:%s orb:%s ftr:%s",
        school_idx, w_idx, l_idx, efg_idx, tov_idx, orb_idx, ftr_idx,
    )

    # Parse body rows
    tbody = table.find("tbody")
    if tbody is None:
        tbody = table

    for tr in tbody.find_all("tr"):
        # Skip header repetition rows and separator rows
        if tr.get("class") and ("thead" in tr.get("class") or "over_header" in tr.get("class")):
            continue

        cells = tr.find_all(["th", "td"])
        if len(cells) < 5:
            continue

        def cell_text(idx):
            if idx is None or idx >= len(cells):
                return ""
            return cells[idx].get_text(strip=True)

        def cell_float(idx):
            t = cell_text(idx)
            if not t:
                return None
            # Remove asterisks (tournament markers) and commas
            t = t.replace("*", "").replace(",", "")
            try:
                return float(t)
            except ValueError:
                return None

        school = cell_text(school_idx).replace("*", "").strip()
        if not school:
            continue

        wins    = cell_text(w_idx)   if w_idx   is not None else ""
        losses  = cell_text(l_idx)   if l_idx   is not None else ""
        c_wins  = cell_text(cw_idx)  if cw_idx  is not None else ""
        c_loss  = cell_text(cl_idx)  if cl_idx  is not None else ""

        wl_overall = f"{wins}-{losses}" if wins and losses else ""
        wl_conf    = f"{c_wins}-{c_loss}" if c_wins and c_loss else ""

        efg = cell_float(efg_idx)
        tov = cell_float(tov_idx)
        orb = cell_float(orb_idx)
        ftr = cell_float(ftr_idx)

        # Sports Reference stores these as decimals already (e.g. 0.512)
        # but sometimes as percentages. Detect and convert.
        def maybe_pct(v):
            if v is None:
                return None
            return v / 100.0 if v > 1.0 else v

        records.append({
            "team_id":             school,
            "efg_pct":             maybe_pct(efg),
            "tov_pct":             maybe_pct(tov),
            "orb_pct":             maybe_pct(orb),
            "ftr":                 maybe_pct(ftr),
            "win_loss_overall":    wl_overall,
            "win_loss_conference": wl_conf,
        })

    log.info("Sports Reference advanced HTML: parsed %d team records.", len(records))
    return records


def fetch_sref_advanced() -> list[dict]:
    """
    Fallback 1: fetch Sports Reference advanced school stats.
    """
    log.info("Fetching Sports Reference advanced stats: %s", SREF_ADVANCED_URL)
    time.sleep(POLITENESS_SLEEP)
    html = _fetch(SREF_ADVANCED_URL)
    if html is None:
        log.warning("Sports Reference advanced fetch returned None.")
        return []
    return _parse_sref_advanced_html(html)

# =============================================================================
# SOURCE 3: SPORTS REFERENCE BASIC STATS (HTML) — MANUAL FORMULA
# =============================================================================

def _parse_sref_basic_html(html: str) -> list[dict]:
    """
    Parse Sports Reference basic school stats table and compute four factors
    manually from raw counting stats.

    Column names in the basic table (approximate):
      School, G, W, L, Conf W, Conf L, FG, FGA, 3P, FTA, FT, ORB, TRB, AST, STL, BLK, TOV, PF

    Formulas:
      eFG%  = (FGM + 0.5 * 3PM) / FGA
      TOV%  = TOV / (FGA + 0.44 * FTA + TOV)
      ORB%  = ORB / (ORB + opp_DRB)  -- opp_DRB not available in basic table,
              so we approximate: ORB / (ORB + (opp_TRB - opp_ORB))
              Without opponent data we use: ORB / TRB (team's own rebounding share)
              as a rough proxy that at least puts it in [0,1].
      FTR   = FTA / FGA
    """
    soup = BeautifulSoup(html, "lxml")
    records = []

    table = soup.find("table", {"id": "basic_school_stats"})
    if table is None:
        table = soup.find("table", id=lambda x: x and "basic" in x.lower() and "school" in x.lower())
    if table is None:
        log.warning("Could not find basic stats table in Sports Reference HTML.")
        return records

    headers = []
    thead = table.find("thead")
    if thead:
        header_rows = thead.find_all("tr")
        if header_rows:
            headers = [
                th.get_text(strip=True).lower()
                for th in header_rows[-1].find_all(["th", "td"])
            ]

    log.info("Sports Reference basic table headers (first 25): %s", headers[:25])

    col_map = {}
    for i, h in enumerate(headers):
        col_map[h] = i

    school_idx = col_map.get("school") or 0
    w_idx      = col_map.get("w")
    l_idx      = col_map.get("l")
    cw_idx     = col_map.get("w.1")
    cl_idx     = col_map.get("l.1")
    fg_idx     = col_map.get("fg")
    fga_idx    = col_map.get("fga")
    threepm_idx = col_map.get("3p") or col_map.get("3pm")
    fta_idx    = col_map.get("fta")
    orb_idx    = col_map.get("orb")
    trb_idx    = col_map.get("trb")
    tov_idx    = col_map.get("tov") or col_map.get("to")

    log.info(
        "Basic col mapping — fg:%s fga:%s 3p:%s fta:%s orb:%s trb:%s tov:%s",
        fg_idx, fga_idx, threepm_idx, fta_idx, orb_idx, trb_idx, tov_idx,
    )

    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        if tr.get("class") and ("thead" in tr.get("class") or "over_header" in tr.get("class")):
            continue
        cells = tr.find_all(["th", "td"])
        if len(cells) < 5:
            continue

        def cell_text(idx):
            if idx is None or idx >= len(cells):
                return ""
            return cells[idx].get_text(strip=True)

        def cell_float(idx) -> Optional[float]:
            t = cell_text(idx).replace("*", "").replace(",", "")
            if not t:
                return None
            try:
                return float(t)
            except ValueError:
                return None

        school = cell_text(school_idx).replace("*", "").strip()
        if not school:
            continue

        wins   = cell_text(w_idx)  if w_idx  is not None else ""
        losses = cell_text(l_idx)  if l_idx  is not None else ""
        c_wins = cell_text(cw_idx) if cw_idx is not None else ""
        c_loss = cell_text(cl_idx) if cl_idx is not None else ""

        wl_overall = f"{wins}-{losses}" if wins and losses else ""
        wl_conf    = f"{c_wins}-{c_loss}" if c_wins and c_loss else ""

        fg   = cell_float(fg_idx)
        fga  = cell_float(fga_idx)
        threepm = cell_float(threepm_idx)
        fta  = cell_float(fta_idx)
        orb  = cell_float(orb_idx)
        trb  = cell_float(trb_idx)
        tov  = cell_float(tov_idx)

        # eFG% = (FGM + 0.5 * 3PM) / FGA
        efg = None
        if fg is not None and threepm is not None and fga and fga > 0:
            efg = (fg + 0.5 * threepm) / fga

        # TOV% = TOV / (FGA + 0.44 * FTA + TOV)
        tov_pct = None
        if tov is not None and fga is not None and fta is not None:
            denom = fga + 0.44 * fta + tov
            if denom > 0:
                tov_pct = tov / denom

        # ORB% approximation (no opponent data): ORB / TRB
        orb_pct = None
        if orb is not None and trb is not None and trb > 0:
            orb_pct = orb / trb

        # FTR = FTA / FGA
        ftr = None
        if fta is not None and fga is not None and fga > 0:
            ftr = fta / fga

        records.append({
            "team_id":             school,
            "efg_pct":             efg,
            "tov_pct":             tov_pct,
            "orb_pct":             orb_pct,
            "ftr":                 ftr,
            "win_loss_overall":    wl_overall,
            "win_loss_conference": wl_conf,
        })

    log.info("Sports Reference basic HTML: parsed %d team records.", len(records))
    return records


def fetch_sref_basic() -> list[dict]:
    """
    Fallback 2: fetch Sports Reference basic school stats and compute formulas.
    """
    log.info("Fetching Sports Reference basic stats: %s", SREF_BASIC_URL)
    time.sleep(POLITENESS_SLEEP)
    html = _fetch(SREF_BASIC_URL)
    if html is None:
        log.warning("Sports Reference basic fetch returned None.")
        return []
    return _parse_sref_basic_html(html)

# =============================================================================
# SOURCE ORCHESTRATION
# =============================================================================

def collect_efficiency_data() -> tuple[list[dict], str]:
    """
    Try sources in order: Barttorvik JSON -> Sports Reference Advanced -> Sports Reference Basic.
    Returns (records, source_label) where records is the best available non-empty list.
    """
    records = fetch_barttorvik()
    if records:
        return records, "barttorvik_json"

    log.warning("Barttorvik JSON returned no records — trying Sports Reference advanced stats.")
    time.sleep(POLITENESS_SLEEP)
    records = fetch_sref_advanced()
    if records:
        return records, "sref_advanced"

    log.warning("Sports Reference advanced returned no records — trying Sports Reference basic stats.")
    time.sleep(POLITENESS_SLEEP)
    records = fetch_sref_basic()
    if records:
        return records, "sref_basic"

    log.error("All data sources failed or returned empty results.")
    return [], "none"

# =============================================================================
# VALIDATE AND ASSEMBLE OUTPUT ROWS
# =============================================================================

def validate_and_build_output(
    raw_records: list[dict],
    fuzzy_mapping: dict[str, tuple[str, float]],
) -> list[dict]:
    """
    For each raw record:
      1. Validate each metric against [0, 1].
      2. Attach the CBS mapped name (if found).
      3. Add as_of_date and schema_version.
    Returns list of output dicts matching the CSV schema.
    """
    output = []
    for rec in raw_records:
        team_id = rec.get("team_id", "").strip()
        if not team_id:
            continue

        cbs_name, confidence = fuzzy_mapping.get(team_id, ("", 0.0))

        efg = _validate_metric("efg_pct", team_id, rec.get("efg_pct"))
        tov = _validate_metric("tov_pct", team_id, rec.get("tov_pct"))
        orb = _validate_metric("orb_pct", team_id, rec.get("orb_pct"))
        ftr = _validate_metric("ftr",     team_id, rec.get("ftr"))

        output.append({
            "team_id":             team_id,
            "cbs_name":            cbs_name,
            "match_confidence":    confidence,
            "efg_pct":             efg,
            "tov_pct":             tov,
            "orb_pct":             orb,
            "ftr":                 ftr,
            "win_loss_overall":    rec.get("win_loss_overall", ""),
            "win_loss_conference": rec.get("win_loss_conference", ""),
            "as_of_date":          AS_OF_DATE,
            "schema_version":      SCHEMA_VERSION,
        })

    return output

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_metrics_csv(output_rows: list[dict], path: str) -> None:
    """
    Save efficiency metrics to CSV.
    Columns: team_id, efg_pct, tov_pct, orb_pct, ftr,
             win_loss_overall, win_loss_conference, as_of_date, schema_version
    (cbs_name and match_confidence excluded — those go in the mapping file.)
    """
    fieldnames = [
        "team_id", "efg_pct", "tov_pct", "orb_pct", "ftr",
        "win_loss_overall", "win_loss_conference", "as_of_date", "schema_version",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    log.info("Saved %d rows to %s", len(output_rows), path)


def save_mapping_csv(
    fuzzy_mapping: dict[str, tuple[str, float]],
    path: str,
) -> None:
    """
    Save team name mapping to CSV.
    Columns: source_name, cbs_name, match_confidence
    """
    fieldnames = ["source_name", "cbs_name", "match_confidence"]
    rows = [
        {"source_name": src, "cbs_name": cbs, "match_confidence": conf}
        for src, (cbs, conf) in sorted(fuzzy_mapping.items())
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved %d mapping rows to %s", len(rows), path)

# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(
    cbs_names: list[str],
    output_rows: list[dict],
    source_label: str,
) -> None:
    total_fetched   = len(output_rows)
    matched_to_cbs  = sum(1 for r in output_rows if r.get("cbs_name"))
    complete_metrics = sum(
        1 for r in output_rows
        if not any(
            math.isnan(r.get(m, float("nan")))
            for m in ("efg_pct", "tov_pct", "orb_pct", "ftr")
        )
    )

    print("\n" + "=" * 60)
    print("EFFICIENCY METRICS SUMMARY")
    print("=" * 60)
    print(f"  Source used          : {source_label}")
    print(f"  As-of date           : {AS_OF_DATE}")
    print(f"  CBS unique teams     : {len(cbs_names)}")
    print(f"  Total teams fetched  : {total_fetched}")
    print(f"  Matched to CBS names : {matched_to_cbs} / {total_fetched}")
    print(f"  Teams with all 4     : {complete_metrics} / {total_fetched}")
    print(f"  Output metrics file  : {OUTPUT_METRICS_PATH}")
    print(f"  Output mapping file  : {OUTPUT_MAPPING_PATH}")
    print("=" * 60 + "\n")

    # Log unmatched CBS teams
    matched_cbs_set = {r["cbs_name"] for r in output_rows if r.get("cbs_name")}
    unmatched_cbs = sorted(set(cbs_names) - matched_cbs_set)
    if unmatched_cbs:
        log.warning(
            "%d CBS teams had no source match: %s",
            len(unmatched_cbs),
            unmatched_cbs,
        )

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("NCAAB Efficiency Metrics Collector")
    print(f"Date: {AS_OF_DATE}  |  Schema: {SCHEMA_VERSION}")
    print("=" * 60 + "\n")

    # Step 1 — Load CBS team names
    cbs_names = load_cbs_team_names(CBS_CSV_PATH)

    # Step 2 — Collect data from best available source
    raw_records, source_label = collect_efficiency_data()

    if not raw_records:
        log.error(
            "No data collected from any source. "
            "Check network access and source URLs."
        )
        print("\nNo data was collected. Exiting without saving files.")
        return

    # Step 3 — Build fuzzy name mapping: source names -> CBS names
    source_names = [r["team_id"] for r in raw_records if r.get("team_id")]
    log.info("Building fuzzy name mapping for %d source teams ...", len(source_names))
    fuzzy_mapping = build_fuzzy_mapping(source_names, cbs_names)

    matched_count = sum(1 for (cbs, conf) in fuzzy_mapping.values() if cbs)
    log.info(
        "Fuzzy mapping complete: %d / %d source names matched to CBS names.",
        matched_count,
        len(source_names),
    )

    # Step 4 — Validate metrics and build output rows
    output_rows = validate_and_build_output(raw_records, fuzzy_mapping)
    log.info("Assembled %d validated output rows.", len(output_rows))

    # Step 5 — Save outputs
    save_metrics_csv(output_rows, OUTPUT_METRICS_PATH)
    save_mapping_csv(fuzzy_mapping, OUTPUT_MAPPING_PATH)

    # Step 6 — Print summary
    print_summary(cbs_names, output_rows, source_label)
    print(f"Row count in metrics CSV: {len(output_rows)}")


if __name__ == "__main__":
    main()
