"""
odds_service.py — Fetches fixtures + odds from The Odds API,
runs predictions through the model, and caches results to JSON.

Architecture:
  The Odds API → odds_service.py → cache.json → GET /matches → Frontend

Usage:
  - Set ODDS_API_KEY env var (get free key at https://the-odds-api.com)
  - Call refresh() to fetch + predict + cache
  - Call get_cached_matches() to read cached data (what the frontend consumes)
  - Without API key, falls back to jornada/jornada_config.py
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_FILE = Path(__file__).parent.parent / "data" / "matches_cache.json"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

# Mapping from The Odds API team names → your canonical names
TEAM_NAME_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton and Hove Albion": "Brighton",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich Town": "Ipswich",
    "Ipswich": "Ipswich",
    "Leicester City": "Leicester",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Man City": "Man City",
    "Manchester United": "Man United",
    "Man United": "Man United",
    "Newcastle United": "Newcastle",
    "Newcastle": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",
    "West Ham United": "West Ham",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
}


def _map_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def _fetch_from_odds_api() -> list[dict]:
    """Fetch upcoming EPL matches with odds from The Odds API."""
    import urllib.request
    import urllib.parse

    ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
    if not ODDS_API_KEY:
        logger.warning("No ODDS_API_KEY set, cannot fetch from The Odds API")
        return []

    params = urllib.parse.urlencode({
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
    })

    url = f"{ODDS_API_URL}?{params}"
    logger.info(f"Fetching odds from The Odds API...")

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.info(f"Got {len(data)} matches. API requests remaining: {remaining}")
            return data
    except Exception as e:
        logger.error(f"Error fetching from The Odds API: {e}")
        return []


def _extract_best_odds(bookmakers: list[dict], home_team: str = "", away_team: str = "") -> dict:
    """Extract average odds from bookmakers for h2h market.

    The Odds API returns outcomes keyed by team name (not "Home"/"Away"),
    plus "Draw". We identify home/away by matching against team names.
    """
    all_h, all_d, all_a = [], [], []

    for bm in bookmakers:
        for market in bm.get("markets", []):
            if market["key"] != "h2h":
                continue
            for o in market["outcomes"]:
                name = o["name"]
                price = o["price"]
                if name == "Draw":
                    all_d.append(price)
                elif name == home_team:
                    all_h.append(price)
                elif name == away_team:
                    all_a.append(price)
                # fallback: if team names don't match, try positional
            if not all_h and not all_a:
                outcomes = [o for o in market["outcomes"] if o["name"] != "Draw"]
                if len(outcomes) >= 2:
                    all_h.append(outcomes[0]["price"])
                    all_a.append(outcomes[1]["price"])

    if not all_h or not all_d or not all_a:
        return {}

    return {
        "home": round(sum(all_h) / len(all_h), 2),
        "draw": round(sum(all_d) / len(all_d), 2),
        "away": round(sum(all_a) / len(all_a), 2),
        "n_bookmakers": len(bookmakers),
    }


def _fallback_from_config() -> list[dict]:
    """Load matches from jornada_config.py as fallback."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from jornada.jornada_config import PARTIDOS_JORNADA, NUMERO_JORNADA

        matches = []
        for p in PARTIDOS_JORNADA:
            matches.append({
                "home": p.local,
                "away": p.visitante,
                "odds": {
                    "home": p.cuota_h,
                    "draw": p.cuota_d,
                    "away": p.cuota_a,
                    "n_bookmakers": 0,
                },
                "commence_time": None,
                "source": "config",
            })
        return matches
    except Exception as e:
        logger.error(f"Error loading fallback config: {e}")
        return []


def _run_predictions(matches: list[dict], predictor) -> list[dict]:
    """Run the ML model on each match and attach predictions."""
    from core.models import Partido

    results = []
    for m in matches:
        odds = m.get("odds", {})
        oh = odds.get("home", 2.80)
        od = odds.get("draw", 3.40)
        oa = odds.get("away", 2.80)

        try:
            partido = Partido(
                local=m["home"],
                visitante=m["away"],
                cuota_h=oh,
                cuota_d=od,
                cuota_a=oa,
            )
            pred = predictor.predecir_partido(partido)
            if pred is None:
                m["prediction"] = None
                results.append(m)
                continue

            # Binary markets
            binaria = predictor.predecir_mercados_binarios(partido)

            m["prediction"] = {
                "result": pred.resultado_predicho,
                "confidence": round(pred.confianza, 4),
                "prob_home": round(pred.prob_local, 4),
                "prob_draw": round(pred.prob_empate, 4),
                "prob_away": round(pred.prob_visitante, 4),
                "market_prob_home": round(pred.prob_mercado_local, 4) if pred.prob_mercado_local else None,
                "market_prob_draw": round(pred.prob_mercado_empate, 4) if pred.prob_mercado_empate else None,
                "market_prob_away": round(pred.prob_mercado_visitante, 4) if pred.prob_mercado_visitante else None,
                "edge": round(pred.diferencia_valor, 4) if pred.diferencia_valor else 0,
                "form_home": pred.forma_local,
                "form_away": pred.forma_visitante,
                "over25_prob": round(binaria.prob_over25, 4) if binaria and binaria.prob_over25 else None,
                "over35_cards_prob": round(binaria.prob_over35_cards, 4) if binaria and binaria.prob_over35_cards else None,
                "over95_corners_prob": round(binaria.prob_over95_corners, 4) if binaria and binaria.prob_over95_corners else None,
            }
        except Exception as e:
            logger.error(f"Error predicting {m['home']} vs {m['away']}: {e}")
            m["prediction"] = None

        results.append(m)

    return results


def _detect_odds_changes(old_matches: list[dict], new_matches: list[dict]) -> list[dict]:
    """Compare old and new odds, mark changes."""
    old_map = {}
    for m in old_matches:
        key = f"{m['home']}_{m['away']}"
        old_map[key] = m.get("odds", {})

    for m in new_matches:
        key = f"{m['home']}_{m['away']}"
        old_odds = old_map.get(key, {})
        new_odds = m.get("odds", {})

        changed = False
        if old_odds:
            for k in ("home", "draw", "away"):
                if abs(old_odds.get(k, 0) - new_odds.get(k, 0)) >= 0.03:
                    changed = True
                    break

        m["odds_changed"] = changed
        if old_odds and changed:
            m["odds_previous"] = {
                "home": old_odds.get("home"),
                "draw": old_odds.get("draw"),
                "away": old_odds.get("away"),
            }

    return new_matches


def refresh(predictor) -> dict:
    """
    Main refresh function:
    1. Fetch fixtures + odds from The Odds API (or fallback to config)
    2. Compare with cached odds to detect changes
    3. Run predictions on all matches
    4. Save to cache file
    """
    # Load existing cache for comparison
    old_data = get_cached_matches()
    old_matches = old_data.get("matches", []) if old_data else []

    # Fetch fresh data
    if os.environ.get("ODDS_API_KEY", ""):
        raw = _fetch_from_odds_api()
        matches = []
        for event in raw:
            raw_home = event.get("home_team", "")
            raw_away = event.get("away_team", "")
            home = _map_team(raw_home)
            away = _map_team(raw_away)
            odds = _extract_best_odds(event.get("bookmakers", []), raw_home, raw_away)
            if not odds:
                logger.warning(f"No odds extracted for {raw_home} vs {raw_away}, skipping")
                continue
            matches.append({
                "home": home,
                "away": away,
                "odds": odds,
                "commence_time": event.get("commence_time"),
                "source": "the_odds_api",
            })
    else:
        matches = _fallback_from_config()

    if not matches:
        logger.warning("No matches found from any source")
        return old_data or {"matches": [], "meta": {}}

    # Detect odds changes
    matches = _detect_odds_changes(old_matches, matches)

    # Run predictions
    if predictor:
        matches = _run_predictions(matches, predictor)

    # Build cache
    cache = {
        "matches": matches,
        "meta": {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "source": "the_odds_api" if os.environ.get("ODDS_API_KEY", "") else "config_fallback",
            "total_matches": len(matches),
            "has_predictions": predictor is not None,
        }
    }

    # Save to file
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    logger.info(f"Cache updated: {len(matches)} matches, source={cache['meta']['source']}")
    return cache


def get_cached_matches() -> Optional[dict]:
    """Read cached matches from JSON file."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading cache: {e}")
        return None
