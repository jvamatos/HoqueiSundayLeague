"""Download games from the hoqueipatins.pt SportsPress REST API (2021/22 onwards).

Raw JSON pages are cached under data/raw/api/ so re-runs don't hit the site again
(pass --refresh to re-download the current season).

Output:
  data/api_matches.csv        one row per game (team-level)
  data/api_player_stats.csv   one row per player per game (goals, cards, set pieces...)
"""
import argparse
import html
import json
import time
from pathlib import Path

import pandas as pd
import requests

from .common import DATA, RAW, USER_AGENT, canonical_team

API = "https://www.hoqueipatins.pt/wp-json/sportspress/v2"

# SportsPress league ids on hoqueipatins.pt
LEAGUES = {
    230: ("1div", "regular"),
    425: ("1div", "playoff"),
    231: ("2div_norte", "regular"),
    232: ("2div_sul", "regular"),
    435: ("2div", "promocao"),
    436: ("2div", "campeao"),
}
SEASONS = {225: "2021-22", 534: "2022-23", 672: "2023-24", 799: "2024-25", 1010: "2025-26", 1122: "2026-27"}
CURRENT_SEASON = 1122

FIELDS = "id,date,status,day,title,teams,results,leagues,seasons,venues,performance"

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def get(path, params, cache: Path, refresh=False):
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    for attempt in range(4):
        r = session.get(f"{API}/{path}", params=params, timeout=60)
        if r.status_code == 400 and "rest_post_invalid_page_number" in r.text:
            data = []
            break
        if r.ok:
            data = r.json()
            break
        time.sleep(2 ** attempt)
    else:
        r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False))
    time.sleep(0.5)  # be polite to the site
    return data


def fetch_events(league, season, refresh=False):
    events, page = [], 1
    while True:
        cache = RAW / "api" / f"events_{league}_{season}_p{page}.json"
        batch = get("events", {"leagues": league, "seasons": season, "per_page": 100,
                               "page": page, "_fields": FIELDS}, cache, refresh)
        events += batch
        if len(batch) < 100:
            return events
        page += 1


def fetch_team_names(team_ids, refresh=False):
    names, ids = {}, sorted(team_ids)
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        cache = RAW / "api" / f"teams_{chunk[0]}_{chunk[-1]}_{len(chunk)}.json"
        for t in get("teams", {"include": ",".join(map(str, chunk)), "per_page": 100,
                               "_fields": "id,title"}, cache, refresh):
            names[t["id"]] = t["title"]["rendered"]
    return names


def fetch_player_names(player_ids, refresh=False):
    names, ids = {}, sorted(i for i in player_ids if i)
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        cache = RAW / "api" / f"players_{chunk[0]}_{chunk[-1]}_{len(chunk)}.json"
        for t in get("players", {"include": ",".join(map(str, chunk)), "per_page": 100,
                                 "_fields": "id,title"}, cache, refresh):
            names[t["id"]] = html.unescape(t["title"]["rendered"])
    return names


def _int(x):
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def _is_gk(pos):
    # SportsPress position 223 = goalkeeper (field can be a list, int or str)
    if isinstance(pos, list):
        pos = pos[0] if pos else None
    return str(pos) == "223"


def parse_round(day):
    # e.g. "0026-Jornada 26" -> 26 ; playoff labels kept as text
    head = (day or "").split("-")[0]
    return _int(head)


def main(refresh=False):
    matches, players, team_ids = [], [], set()
    for sid, season in SEASONS.items():
        for lid, (comp, phase) in LEAGUES.items():
            evs = fetch_events(lid, sid, refresh=refresh and sid == CURRENT_SEASON)
            print(f"{season} {comp:10s} {phase:9s} {len(evs):4d} games")
            for e in evs:
                teams = e.get("teams") or []
                if len(teams) != 2:
                    continue
                home, away = teams
                team_ids.update(teams)
                res = e.get("results")
                res = res if isinstance(res, dict) else {}
                rh, ra = res.get(str(home)), res.get(str(away))
                rh, ra = (rh if isinstance(rh, dict) else {}), (ra if isinstance(ra, dict) else {})
                hg, ag = _int(rh.get("goals")), _int(ra.get("goals"))
                played = e["status"] == "publish" and hg is not None and ag is not None
                matches.append(dict(
                    source="api", event_id=e["id"], season=season, competition=comp, phase=phase,
                    round=parse_round(e.get("day")), round_label=e.get("day"),
                    date=e["date"][:10], kickoff=e["date"][11:16], home_id=home, away_id=away,
                    home_goals=hg if played else None, away_goals=ag if played else None,
                    home_ht=_int(rh.get("first")), away_ht=_int(ra.get("first")),
                    home_ot=_int(rh.get("ot")), away_ot=_int(ra.get("ot")),
                    home_pen=_int(rh.get("ppo")), away_pen=_int(ra.get("ppo")),
                    played=played, venue_id=(e.get("venues") or [None])[0],
                ))
                perf = e.get("performance")
                for tid, plist in (perf.items() if isinstance(perf, dict) else []):
                    if not isinstance(plist, dict) or _int(tid) not in (home, away):
                        continue
                    for pid, st in plist.items():
                        if not isinstance(st, dict) or _int(pid) in (None, 0):
                            continue
                        row = {k: v for k, v in st.items() if k not in ("position", "status")}
                        row.update(event_id=e["id"], team_id=_int(tid), player_id=_int(pid),
                                   role=st.get("status"),
                                   goalkeeper=_is_gk(st.get("position")))
                        players.append(row)

    names = fetch_team_names(team_ids)
    m = pd.DataFrame(matches).drop_duplicates("event_id")
    m["home_name"], m["away_name"] = m.home_id.map(names), m.away_id.map(names)
    m["home"] = m.home_name.map(canonical_team)
    m["away"] = m.away_name.map(canonical_team)
    m.to_csv(DATA / "api_matches.csv", index=False)
    p = pd.DataFrame(players)
    p.to_csv(DATA / "api_player_stats.csv", index=False)
    pn = fetch_player_names(set(p.player_id.dropna().astype(int)))
    pd.DataFrame({"player_id": list(pn), "name": list(pn.values())}).to_csv(DATA / "players.csv", index=False)
    print(f"saved {len(m)} games ({m.played.sum()} played), {len(p)} player-game rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download the current season")
    main(ap.parse_args().refresh)
