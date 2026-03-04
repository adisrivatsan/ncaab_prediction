"""
KenPom-Style Ratings Harvester — Bart Torvik (T-Rank) Source
=============================================================
Fetches KenPom-equivalent adjusted efficiency ratings for all NCAA Men's D-I
teams from barttorvik.com (free, public, no authentication required).

Reads CBS team names from:
    ../cbs_games.csv  (columns: home_name, away_name)

Writes two output files:
    kenpom_ratings.csv        — one row per D-I team with efficiency metrics
    kenpom_name_mapping.csv   — source_name → cbs_name mapping audit table

Source priority (tried in order until one succeeds):
    1. https://barttorvik.com/trank.php?json=1          (JSON, preferred)
    2. https://barttorvik.com/2026_team_results.json     (JSON, alternate)
    3. https://barttorvik.com/trankwebb_2026.json        (JSON, alternate)
    4. https://barttorvik.com/trank.php?csv=1            (CSV fallback)

Schema version: 1.0
"""

from __future__ import annotations

import csv
import difflib
import json
import logging
import os
import sys
from datetime import date
from io import StringIO
from typing import Any

import requests
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON_YEAR = 2026
AS_OF_DATE  = date.today().isoformat()
SOURCE_LABEL = "barttorvik"
SCHEMA_VERSION = "1.0"

# Paths — all absolute, relative to this script's location
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

CBS_CSV_PATH      = os.path.join(_PROJECT_DIR, "cbs_games.csv")
RATINGS_OUT_PATH  = os.path.join(_SCRIPT_DIR, "kenpom_ratings.csv")
MAPPING_OUT_PATH  = os.path.join(_SCRIPT_DIR, "kenpom_name_mapping.csv")

REQUEST_TIMEOUT  = 20   # seconds
RETRY_ATTEMPTS   = 3
RETRY_BACKOFF    = [2, 5, 10]

FUZZY_CUTOFF = 0.6  # difflib cutoff for close-match team name mapping

# Validation ranges
ADJ_EM_RANGE = (-40.0, 50.0)
ADJ_OD_RANGE = (60.0, 150.0)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Ordered list of endpoint specs: (url, content_type_hint)
# content_type_hint: "json" or "csv"
ENDPOINTS: list[tuple[str, str]] = [
    (f"https://barttorvik.com/trank.php?json=1&year={SEASON_YEAR}", "json"),
    (f"https://barttorvik.com/{SEASON_YEAR}_team_results.json",     "json"),
    (f"https://barttorvik.com/trankwebb_{SEASON_YEAR}.json",        "json"),
    (f"https://barttorvik.com/trank.php?csv=1&year={SEASON_YEAR}",  "csv"),
    # Non-year-parameterised fallbacks (returns current season)
    ("https://barttorvik.com/trank.php?json=1",                     "json"),
    ("https://barttorvik.com/trank.php?csv=1",                      "csv"),
]

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

def _fetch_url(url: str) -> requests.Response | None:
    """
    GET the given URL with retry/backoff.
    Returns the Response on HTTP 200, or None after all attempts fail.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            log.warning("HTTP %s on attempt %d — %s", resp.status_code, attempt + 1, url)
        except requests.exceptions.Timeout:
            log.warning("Timeout on attempt %d — %s", attempt + 1, url)
        except requests.exceptions.RequestException as exc:
            log.warning("Request error on attempt %d: %s", attempt + 1, exc)

        if attempt < RETRY_ATTEMPTS - 1:
            import time
            wait = RETRY_BACKOFF[attempt]
            log.info("Retrying in %ds ...", wait)
            time.sleep(wait)

    log.error("All %d attempts failed for: %s", RETRY_ATTEMPTS, url)
    return None

# =============================================================================
# PARSERS  —  Bart Torvik response formats
# =============================================================================

def _parse_json_response(data: Any) -> list[dict]:
    """
    Parse a Bart Torvik JSON payload into a normalised list of team dicts.

    Bart Torvik JSON format variants observed:

    Variant A — trank.php?json=1
        Returns a list of objects, each with keys such as:
            "team", "conf", "adj_o", "adj_d", "adj_t", "adj_em", "rank", ...
        (field names may vary; we try both snake_case and camelCase spellings)

    Variant B — YYYY_team_results.json
        Returns an array of arrays; column names in a separate key "cols" or
        inferred by position. Position order documented on barttorvik.com:
            0: rank, 1: team, 2: conf, 3: record, 4: adj_o, 5: adj_d,
            6: adj_em, 7: adj_t, 8: luck, 9: sos_adj_em, ...
        Some versions wrap the array data under a key "data".

    We attempt both variants and return whichever yields non-empty results.
    """
    teams: list[dict] = []

    # ── Variant A: list of dicts ──────────────────────────────────────────────
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        for item in data:
            team = _extract_dict_fields(item)
            if team:
                teams.append(team)
        if teams:
            log.info("Parsed %d teams from JSON dict-list format (Variant A)", len(teams))
            return teams

    # ── Variant B: {"data": [...]} or {"cols": [...], "data": [...]} ──────────
    if isinstance(data, dict):
        rows = data.get("data") or data.get("teams") or data.get("results")
        cols = data.get("cols") or data.get("columns") or []
        if rows and isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    team = _extract_dict_fields(row)
                elif isinstance(row, list) and cols:
                    team = _extract_positional(row, cols)
                elif isinstance(row, list):
                    team = _extract_positional_default(row)
                else:
                    continue
                if team:
                    teams.append(team)
            if teams:
                log.info("Parsed %d teams from JSON dict format (Variant B)", len(teams))
                return teams

    # ── Variant C: top-level array of arrays (positional, no column header) ───
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        for row in data:
            team = _extract_positional_default(row)
            if team:
                teams.append(team)
        if teams:
            log.info("Parsed %d teams from JSON array-of-arrays format (Variant C)", len(teams))
            return teams

    log.warning("Could not identify JSON structure; no teams parsed")
    return []


def _extract_dict_fields(item: dict) -> dict | None:
    """
    Extract team/rank/efficiency fields from a Bart Torvik JSON object.
    Tries multiple spellings for each field.
    Returns a normalised dict or None if the team name is missing.
    """
    # Team name candidates
    name = (
        item.get("team")
        or item.get("name")
        or item.get("TeamName")
        or item.get("teamName")
        or item.get("TEAM")
        or item.get("team_name")
    )
    if not name:
        return None

    # Rank
    rank_raw = (
        item.get("rank")
        or item.get("rk")
        or item.get("Rank")
        or item.get("ranking")
        or item.get("rpi_rank")
        or item.get("adj_em_rank")
    )

    # Adjusted Efficiency Margin
    adj_em_raw = (
        item.get("adj_em")
        or item.get("adjEM")
        or item.get("AdjEM")
        or item.get("adjEm")
        or item.get("net_rating")
        or item.get("rating")
    )

    # Adjusted Offensive Efficiency
    adj_o_raw = (
        item.get("adj_o")
        or item.get("adjO")
        or item.get("AdjO")
        or item.get("adj_oe")
        or item.get("adjOE")
        or item.get("off_rating")
        or item.get("orating")
    )

    # Adjusted Defensive Efficiency
    adj_d_raw = (
        item.get("adj_d")
        or item.get("adjD")
        or item.get("AdjD")
        or item.get("adj_de")
        or item.get("adjDE")
        or item.get("def_rating")
        or item.get("drating")
    )

    try:
        rank   = int(float(rank_raw))   if rank_raw   is not None else None
        adj_em = float(adj_em_raw)       if adj_em_raw is not None else None
        adj_o  = float(adj_o_raw)        if adj_o_raw  is not None else None
        adj_d  = float(adj_d_raw)        if adj_d_raw  is not None else None
    except (ValueError, TypeError):
        rank = adj_em = adj_o = adj_d = None

    # If adj_em is missing but adj_o and adj_d are present, derive it
    if adj_em is None and adj_o is not None and adj_d is not None:
        adj_em = round(adj_o - adj_d, 2)

    return {
        "team_id":  str(name).strip(),
        "rank":     rank,
        "adj_em":   adj_em,
        "adj_o":    adj_o,
        "adj_d":    adj_d,
    }


def _extract_positional(row: list, cols: list) -> dict | None:
    """Extract fields from a positional row using a provided column list."""
    col_map = {str(c).lower(): i for i, c in enumerate(cols)}
    item = {str(cols[i]).lower(): v for i, v in enumerate(row) if i < len(cols)}
    return _extract_dict_fields(item)


def _extract_positional_default(row: list) -> dict | None:
    """
    Extract fields from a positional row using Bart Torvik's documented
    default column order for the team ratings table.

    2026_team_results.json layout (rank-first):
        0: rank  1: team  2: conf  3: record  4: adj_o  5: def_rank
        6: adj_d  7: adj_t  8: wab  9: luck  10: sos_adj_em  ...
    Note: adj_em is NOT a direct column — it is derived as adj_o − adj_d.
    Index 5 is a defensive rank integer (1–365), not adj_d.

    Some endpoint variants omit the leading rank column:
        0: team  1: conf  2: record  3: adj_o  4: adj_d  5: adj_em ...
    """
    if len(row) < 3:
        return None

    # Heuristic: if first element is numeric it is likely a rank
    try:
        float(row[0])
        has_rank_first = True
    except (ValueError, TypeError):
        has_rank_first = False

    if has_rank_first:
        # rank, team, conf, record, adj_o, def_rank, adj_d, adj_t, ...
        # Index 5 is a defensive rank value (1-365), not adj_d — skip it.
        # adj_em is derived as adj_o - adj_d (no direct column).
        rank_raw   = row[0]
        name_raw   = row[1] if len(row) > 1 else None
        adj_o_raw  = row[4] if len(row) > 4 else None
        adj_d_raw  = row[6] if len(row) > 6 else None
        adj_em_raw = None  # derived as adj_o - adj_d in _extract_dict_fields
    else:
        # team, conf, record, adj_o, adj_d, adj_em, rank, ...
        rank_raw   = row[6] if len(row) > 6 else None
        name_raw   = row[0]
        adj_o_raw  = row[3] if len(row) > 3 else None
        adj_d_raw  = row[4] if len(row) > 4 else None
        adj_em_raw = row[5] if len(row) > 5 else None

    if not name_raw:
        return None

    synthetic = {
        "team":   str(name_raw).strip(),
        "rank":   rank_raw,
        "adj_o":  adj_o_raw,
        "adj_d":  adj_d_raw,
        "adj_em": adj_em_raw,
    }
    return _extract_dict_fields(synthetic)


def _parse_csv_response(text: str) -> list[dict]:
    """
    Parse a Bart Torvik CSV response.

    Expected header row contains some subset of:
        Rk, Team, Conf, Record, AdjOE, AdjDE, AdjEM, AdjT, Luck, ...

    Column name matching is case-insensitive and tolerates minor variations.
    """
    teams: list[dict] = []
    reader = csv.DictReader(StringIO(text))

    # Normalise header keys to lowercase for flexible matching
    for row in reader:
        norm = {k.strip().lower(): v.strip() for k, v in row.items() if k}

        # Team name
        name = (
            norm.get("team")
            or norm.get("name")
            or norm.get("teamname")
        )
        if not name or name.lower() in ("team", ""):
            continue

        # Rank
        rank_raw = (
            norm.get("rk")
            or norm.get("rank")
            or norm.get("#")
            or norm.get("ranking")
        )

        # Efficiencies
        adj_o_raw  = norm.get("adjoe") or norm.get("adj_o") or norm.get("adjo") or norm.get("off_eff")
        adj_d_raw  = norm.get("adjde") or norm.get("adj_d") or norm.get("adjd") or norm.get("def_eff")
        adj_em_raw = (
            norm.get("adjem")
            or norm.get("adj_em")
            or norm.get("adjefficiency")
            or norm.get("net")
        )

        try:
            rank   = int(float(rank_raw))   if rank_raw   else None
            adj_o  = float(adj_o_raw)        if adj_o_raw  else None
            adj_d  = float(adj_d_raw)        if adj_d_raw  else None
            adj_em = float(adj_em_raw)       if adj_em_raw else None
        except (ValueError, TypeError):
            rank = adj_o = adj_d = adj_em = None

        if adj_em is None and adj_o is not None and adj_d is not None:
            adj_em = round(adj_o - adj_d, 2)

        teams.append({
            "team_id":  str(name).strip(),
            "rank":     rank,
            "adj_em":   adj_em,
            "adj_o":    adj_o,
            "adj_d":    adj_d,
        })

    if teams:
        log.info("Parsed %d teams from CSV format", len(teams))
    return teams

# =============================================================================
# ENDPOINT FETCHING  —  tries each endpoint until one returns data
# =============================================================================

def fetch_barttorvik_ratings() -> list[dict]:
    """
    Attempt each endpoint in ENDPOINTS order.
    Returns the first non-empty list of parsed team dicts.
    Raises RuntimeError if all endpoints fail.
    """
    for url, hint in ENDPOINTS:
        log.info("Trying endpoint: %s", url)
        resp = _fetch_url(url)
        if resp is None:
            log.warning("  Skipping — fetch failed")
            continue

        content = resp.text.strip()
        if not content:
            log.warning("  Skipping — empty response body")
            continue

        # Determine parse strategy
        content_type = resp.headers.get("Content-Type", "").lower()
        is_json = (
            hint == "json"
            or "json" in content_type
            or content.startswith("[")
            or content.startswith("{")
        )

        if is_json:
            try:
                data = json.loads(content)
                teams = _parse_json_response(data)
                if teams:
                    log.info("  SUCCESS — %d teams parsed as JSON from %s", len(teams), url)
                    return teams
                else:
                    log.warning("  JSON parsed but 0 teams extracted — trying next endpoint")
            except json.JSONDecodeError as exc:
                log.warning("  JSON decode error: %s — trying CSV fallback", exc)
                # Fall through to CSV parsing
                teams = _parse_csv_response(content)
                if teams:
                    log.info("  SUCCESS — %d teams parsed as CSV from %s", len(teams), url)
                    return teams
        else:
            teams = _parse_csv_response(content)
            if teams:
                log.info("  SUCCESS — %d teams parsed as CSV from %s", len(teams), url)
                return teams
            else:
                log.warning("  CSV parsed but 0 teams extracted — trying next endpoint")

    raise RuntimeError(
        "All Bart Torvik endpoints failed. Possible causes:\n"
        "  - Network connectivity issue\n"
        "  - barttorvik.com is temporarily unavailable\n"
        "  - The URL structure has changed (check https://barttorvik.com manually)\n"
        "Fallback: download the T-Rank table manually as CSV and pass it to "
        "load_manual_csv() in this script."
    )

# =============================================================================
# MANUAL CSV FALLBACK
# =============================================================================

def load_manual_csv(path: str) -> list[dict]:
    """
    Ingest a manually downloaded Bart Torvik CSV export.

    Instructions for manual download:
        1. Go to https://barttorvik.com/trank.php
        2. Select the desired season year
        3. Click "Export CSV" (or use Ctrl+S on a table-only view)
        4. Save to disk and pass the file path to this function

    Expected columns (any order, case-insensitive):
        Rk, Team, Conf, Record, AdjOE, AdjDE, AdjEM, AdjT, ...
    """
    log.info("Loading manual CSV from: %s", path)
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        raise FileNotFoundError(f"Could not open manual CSV: {exc}") from exc

    teams = _parse_csv_response(content)
    if not teams:
        raise ValueError(
            f"No teams parsed from manual CSV at {path}. "
            "Please verify the file contains expected Bart Torvik columns "
            "(Rk/rank, Team, AdjOE/adj_o, AdjDE/adj_d, AdjEM/adj_em)."
        )
    log.info("Loaded %d teams from manual CSV", len(teams))
    return teams

# =============================================================================
# CBS TEAM NAME LOADING
# =============================================================================

def load_cbs_names(csv_path: str) -> list[str]:
    """
    Read all unique team names from home_name and away_name columns of
    cbs_games.csv. Returns a sorted deduplicated list.
    """
    if not os.path.isfile(csv_path):
        log.error("CBS CSV not found at: %s", csv_path)
        raise FileNotFoundError(f"CBS CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["home_name", "away_name"])
    names = set(df["home_name"].dropna().tolist()) | set(df["away_name"].dropna().tolist())
    names_sorted = sorted(names)
    log.info("Loaded %d unique CBS team names from %s", len(names_sorted), csv_path)
    return names_sorted

# =============================================================================
# TEAM NAME MAPPING
# =============================================================================

def build_name_mapping(
    source_names: list[str],
    cbs_names: list[str],
) -> dict[str, tuple[str, str]]:
    """
    Map source team names to CBS team names.

    Strategy:
        1. Exact match (case-sensitive)
        2. Exact match (case-insensitive)
        3. difflib.get_close_matches with cutoff=FUZZY_CUTOFF

    Returns:
        dict mapping source_name -> (cbs_name, match_type)
        match_type is one of: "exact", "exact_ci", "fuzzy", "unmatched"
    """
    cbs_lower_map: dict[str, str] = {n.lower(): n for n in cbs_names}
    mapping: dict[str, tuple[str, str]] = {}

    unmatched: list[str] = []

    for src in source_names:
        # 1. Exact match
        if src in cbs_names:
            mapping[src] = (src, "exact")
            continue

        # 2. Case-insensitive exact match
        src_lower = src.lower()
        if src_lower in cbs_lower_map:
            mapping[src] = (cbs_lower_map[src_lower], "exact_ci")
            continue

        # 3. Fuzzy match
        matches = difflib.get_close_matches(src, cbs_names, n=1, cutoff=FUZZY_CUTOFF)
        if matches:
            mapping[src] = (matches[0], "fuzzy")
        else:
            # Also try case-insensitive fuzzy
            matches_ci = difflib.get_close_matches(src_lower, list(cbs_lower_map.keys()), n=1, cutoff=FUZZY_CUTOFF)
            if matches_ci:
                mapping[src] = (cbs_lower_map[matches_ci[0]], "fuzzy")
            else:
                mapping[src] = ("", "unmatched")
                unmatched.append(src)

    exact_count   = sum(1 for _, t in mapping.values() if t == "exact")
    exact_ci_count = sum(1 for _, t in mapping.values() if t == "exact_ci")
    fuzzy_count   = sum(1 for _, t in mapping.values() if t == "fuzzy")
    unmatched_count = len(unmatched)

    log.info(
        "Name mapping complete: %d exact, %d exact_ci, %d fuzzy, %d unmatched",
        exact_count, exact_ci_count, fuzzy_count, unmatched_count,
    )
    if unmatched:
        log.warning(
            "%d source team names could not be mapped to CBS names:", unmatched_count
        )
        for nm in unmatched[:20]:   # cap log output to 20
            log.warning("  UNMATCHED: %s", nm)
        if unmatched_count > 20:
            log.warning("  ... and %d more (see kenpom_name_mapping.csv)", unmatched_count - 20)

    return mapping

# =============================================================================
# VALIDATION
# =============================================================================

def validate_ratings(teams: list[dict]) -> int:
    """
    Flag out-of-range values and log warnings.
    Returns the count of validation warnings issued.
    """
    warn_count = 0
    for t in teams:
        name = t.get("team_id", "?")

        adj_em = t.get("adj_em")
        if adj_em is not None:
            lo, hi = ADJ_EM_RANGE
            if not (lo <= adj_em <= hi):
                log.warning(
                    "VALIDATION: adj_em=%.2f out of range [%.0f, %.0f] for team '%s'",
                    adj_em, lo, hi, name,
                )
                warn_count += 1

        for metric in ("adj_o", "adj_d"):
            val = t.get(metric)
            if val is not None:
                lo, hi = ADJ_OD_RANGE
                if not (lo <= val <= hi):
                    log.warning(
                        "VALIDATION: %s=%.2f out of range [%.0f, %.0f] for team '%s'",
                        metric, val, lo, hi, name,
                    )
                    warn_count += 1

    if warn_count == 0:
        log.info("Validation passed — all efficiency values within expected ranges")
    else:
        log.warning("Validation complete — %d out-of-range values flagged", warn_count)

    return warn_count

# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def build_ratings_df(
    teams: list[dict],
    mapping: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """
    Combine raw team data with CBS name mapping and metadata into the
    final ratings DataFrame with the required output schema.
    """
    rows = []
    for i, t in enumerate(teams):
        src_name = t["team_id"]
        cbs_name, _match_type = mapping.get(src_name, ("", "unmatched"))

        # Assign rank by sorted adj_em if not already present
        rank = t.get("rank") if t.get("rank") is not None else (i + 1)

        rows.append({
            "team_id":        src_name,
            "cbs_name":       cbs_name,
            "kenpom_rank":    rank,
            "adj_em":         t.get("adj_em"),
            "adj_o":          t.get("adj_o"),
            "adj_d":          t.get("adj_d"),
            "as_of_date":     AS_OF_DATE,
            "source":         SOURCE_LABEL,
            "schema_version": SCHEMA_VERSION,
        })

    df = pd.DataFrame(rows, columns=[
        "team_id", "cbs_name", "kenpom_rank",
        "adj_em", "adj_o", "adj_d",
        "as_of_date", "source", "schema_version",
    ])

    # Enforce numeric types (project convention)
    for col in ("kenpom_rank", "adj_em", "adj_o", "adj_d"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Assign sequential rank by adj_em if ranks are all missing
    if df["kenpom_rank"].isna().all() and df["adj_em"].notna().any():
        log.info("No explicit ranks in source — deriving kenpom_rank from adj_em descending")
        df = df.sort_values("adj_em", ascending=False).reset_index(drop=True)
        df["kenpom_rank"] = range(1, len(df) + 1)
    else:
        df = df.sort_values("kenpom_rank", na_position="last").reset_index(drop=True)

    return df


def build_mapping_df(
    mapping: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Build the name mapping audit DataFrame."""
    rows = [
        {"source_name": src, "cbs_name": cbs, "match_type": mtype}
        for src, (cbs, mtype) in sorted(mapping.items())
    ]
    return pd.DataFrame(rows, columns=["source_name", "cbs_name", "match_type"])


def write_outputs(ratings_df: pd.DataFrame, mapping_df: pd.DataFrame) -> None:
    """Write ratings CSV and mapping CSV to disk."""
    ratings_df.to_csv(RATINGS_OUT_PATH, index=False)
    log.info("Ratings saved  -> %s  (%d rows)", RATINGS_OUT_PATH, len(ratings_df))

    mapping_df.to_csv(MAPPING_OUT_PATH, index=False)
    log.info("Mapping saved  -> %s  (%d rows)", MAPPING_OUT_PATH, len(mapping_df))

# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary(
    ratings_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    warn_count: int,
) -> None:
    """Print a human-readable summary of results."""
    total          = len(ratings_df)
    matched        = (mapping_df["match_type"] != "unmatched").sum()
    unmatched      = (mapping_df["match_type"] == "unmatched").sum()
    exact          = (mapping_df["match_type"] == "exact").sum()
    fuzzy          = mapping_df["match_type"].isin(["fuzzy"]).sum()
    missing_adj_em = ratings_df["adj_em"].isna().sum()
    missing_adj_o  = ratings_df["adj_o"].isna().sum()
    missing_adj_d  = ratings_df["adj_d"].isna().sum()

    print()
    print("=" * 60)
    print("KENPOM RATINGS HARVEST SUMMARY")
    print("=" * 60)
    print(f"  Source           : {SOURCE_LABEL}")
    print(f"  As-of date       : {AS_OF_DATE}")
    print(f"  Season year      : {SEASON_YEAR}")
    print()
    print(f"  Total D-I teams fetched   : {total}")
    print(f"  Matched to CBS names      : {matched}  ({100*matched/max(total,1):.1f}%)")
    print(f"    - Exact matches         : {exact}")
    print(f"    - Fuzzy matches         : {fuzzy}")
    print(f"    - Unmatched             : {unmatched}")
    print()
    print(f"  adj_em missing            : {missing_adj_em}")
    print(f"  adj_o  missing            : {missing_adj_o}")
    print(f"  adj_d  missing            : {missing_adj_d}")
    print(f"  Validation warnings       : {warn_count}")
    print()
    print(f"  Outputs:")
    print(f"    {RATINGS_OUT_PATH}")
    print(f"    {MAPPING_OUT_PATH}")
    print("=" * 60)

    if unmatched > 0:
        unmatched_names = mapping_df.loc[
            mapping_df["match_type"] == "unmatched", "source_name"
        ].tolist()
        print()
        print(f"  Unmatched source teams ({unmatched}):")
        for nm in unmatched_names[:30]:
            print(f"    - {nm}")
        if unmatched > 30:
            print(f"    ... and {unmatched - 30} more (see kenpom_name_mapping.csv)")

    print()

    # Top-10 preview
    preview_cols = ["cbs_name", "kenpom_rank", "adj_em", "adj_o", "adj_d"]
    available = [c for c in preview_cols if c in ratings_df.columns]
    if not ratings_df.empty and len(available) > 1:
        print("  Top 10 teams by T-Rank:")
        print(
            ratings_df.head(10)[available]
            .to_string(index=False)
        )
        print()

# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    log.info("=" * 60)
    log.info("Bart Torvik (T-Rank) Ratings Harvester")
    log.info("Season: %d  |  As-of: %s", SEASON_YEAR, AS_OF_DATE)
    log.info("=" * 60)

    # Step 1: Load CBS team names
    log.info("Step 1/5 — Loading CBS team names from %s", CBS_CSV_PATH)
    cbs_names = load_cbs_names(CBS_CSV_PATH)

    # Step 2: Fetch ratings from Bart Torvik
    log.info("Step 2/5 — Fetching T-Rank ratings from barttorvik.com")
    try:
        raw_teams = fetch_barttorvik_ratings()
    except RuntimeError as exc:
        log.error(str(exc))
        log.error(
            "Manual fallback: call load_manual_csv('/path/to/trank_export.csv') "
            "to ingest a manually downloaded file."
        )
        sys.exit(1)

    # Step 3: Validate
    log.info("Step 3/5 — Validating efficiency metrics")
    warn_count = validate_ratings(raw_teams)

    # Step 4: Build name mapping
    log.info("Step 4/5 — Mapping source team names to CBS team names")
    source_names = [t["team_id"] for t in raw_teams]
    mapping = build_name_mapping(source_names, cbs_names)

    # Step 5: Assemble and write outputs
    log.info("Step 5/5 — Assembling DataFrames and writing output files")
    ratings_df = build_ratings_df(raw_teams, mapping)
    mapping_df = build_mapping_df(mapping)
    write_outputs(ratings_df, mapping_df)

    # Summary
    print_summary(ratings_df, mapping_df, warn_count)


if __name__ == "__main__":
    main()
