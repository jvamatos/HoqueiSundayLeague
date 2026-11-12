"""Cups, European competitions and foreign leagues from the hoqueipatins.pt API (2021/22 on).

Used only to sharpen team ratings (and for rest days between games): Portuguese teams play
Spanish/Italian/French teams in Europe, so those leagues are included to rate the opponents.

Output: data/extra_matches.csv (same columns as api_matches.csv, plus `neutral`)
"""
import argparse
import html

import pandas as pd

from .common import DATA, RAW, canonical_team
from .scrape_api import SEASONS, CURRENT_SEASON, get, fetch_team_names, _int

# league id -> (competition, phase, neutral venue?)
EXTRA = {
    233: ("taca", "cup", False), 599: ("taca", "cup", False), 600: ("taca", "cup", False),
    535: ("elite_cup", "cup", True), 538: ("supertaca", "cup", True), 426: ("europe", "final4", True),
    351: ("europe", "group", False), 352: ("europe", "group", False), 559: ("europe", "group", False),
    560: ("europe", "group", False), 616: ("europe", "group", False), 617: ("europe", "group", False),
    618: ("europe", "group", False), 619: ("europe", "group", False), 626: ("europe", "group", False),
    627: ("europe", "group", False), 628: ("europe", "group", False), 629: ("europe", "group", False),
    850: ("europe", "group", False), 851: ("europe", "group", False), 852: ("europe", "group", False),
    853: ("europe", "group", False), 1123: ("europe", "group", False), 653: ("europe", "final8", True),
    553: ("europe", "cup", True), 410: ("europe", "cup", False), 353: ("europe", "cup", False),
    354: ("europe", "cup", False), 355: ("europe", "cup", False), 356: ("europe", "cup", False),
    620: ("europe", "cup", False), 857: ("europe", "cup", False), 358: ("europe", "intercontinental", True),
    235: ("esp", "regular", False), 424: ("esp", "playoff", False),
    236: ("ita", "regular", False), 413: ("ita", "playoff", False),
    284: ("fra", "regular", False),
}
FIELDS = "id,date,status,day,teams,results,leagues,seasons"


def fetch(league, season, refresh=False):
    events, page = [], 1
    while True:
        cache = RAW / "api_extra" / f"events_{league}_{season}_p{page}.json"
        batch = get("events", {"leagues": league, "seasons": season, "per_page": 100, "page": page,
                               "_fields": FIELDS}, cache, refresh)
        events += batch
        if len(batch) < 100:
            return events
        page += 1


def main(refresh=False):
    rows, team_ids = [], set()
    for sid, season in SEASONS.items():
        n = 0
        for lid, (comp, phase, neutral) in EXTRA.items():
            for e in fetch(lid, sid, refresh=refresh and sid == CURRENT_SEASON):
                teams = e.get("teams") or []
                if len(teams) != 2:
                    continue
                home, away = teams
                res = e.get("results") if isinstance(e.get("results"), dict) else {}
                rh = res.get(str(home)) if isinstance(res.get(str(home)), dict) else {}
                ra = res.get(str(away)) if isinstance(res.get(str(away)), dict) else {}
                hg, ag = _int(rh.get("goals")), _int(ra.get("goals"))
                played = e["status"] == "publish" and hg is not None and ag is not None
                team_ids.update(teams)
                rows.append(dict(source="api_extra", event_id=e["id"], season=season, competition=comp,
                                 phase=phase, round_label=e.get("day"), date=e["date"][:10],
                                 home_id=home, away_id=away,
                                 home_goals=hg if played else None, away_goals=ag if played else None,
                                 home_ot=_int(rh.get("ot")), away_ot=_int(ra.get("ot")),
                                 home_pen=_int(rh.get("ppo")), away_pen=_int(ra.get("ppo")),
                                 played=played, neutral=neutral))
                n += 1
        print(f"{season}: {n} games")
    m = pd.DataFrame(rows).drop_duplicates("event_id")
    names = {k: html.unescape(v) for k, v in fetch_team_names(team_ids).items()}
    # Portuguese teams keep the key they have in the domestic leagues (same team id);
    # foreign teams get a country-free key prefixed with "x:" so names never collide.
    dom = pd.read_csv(DATA / "api_matches.csv")
    key_of = dict(zip(dom.home_id, dom.home)) | dict(zip(dom.away_id, dom.away))
    for side in ("home", "away"):
        ids = m[f"{side}_id"]
        m[f"{side}_name"] = ids.map(names)
        m[side] = [key_of.get(i) or "x:" + str(canonical_team(names.get(i, str(i)))) for i in ids]
    m.to_csv(DATA / "extra_matches.csv", index=False)
    print(f"saved {len(m)} games ({m.played.sum()} played)")
    print(m[m.played].groupby("competition").size())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    main(ap.parse_args().refresh)
