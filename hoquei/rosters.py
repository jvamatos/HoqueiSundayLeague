"""Pre-season squads from the hoqueipatins.pt transfers pages.

Each page (e.g. /transferencias-placard-2026-27/) lists, per club: PERMANÊNCIAS (players who
stay), SAÍDAS (who leaves and where to) and ENTRADAS (who arrives and from where). The squad
before round 1 = players who stay + players who arrive.

Output: data/rosters.csv  (season, league, team, player, player_id, status, other_club)
"""
import difflib
import html
import re
import time
import unicodedata

import pandas as pd
import requests

from .common import DATA, RAW, USER_AGENT, team_key

PAGES = {  # (season, league) -> slug
    ("2023-24", "1div"): "transferencias-placard-2023-24",
    ("2024-25", "1div"): "transferencias-placard-2024-25",
    ("2024-25", "2div_norte"): "transferencias-2a-divisao-norte-2024-25",
    ("2024-25", "2div_sul"): "transferencias-2a-divisao-sul-2024-25",
    ("2025-26", "1div"): "transferencias-placard-2025-26",
    ("2025-26", "2div_norte"): "transferencias-2a-divisao-norte-2025-26",
    ("2025-26", "2div_sul"): "transferencias-2a-divisao-sul-2025-26",
    ("2026-27", "1div"): "transferencias-placard-2026-27",
    ("2026-27", "2div_norte"): "transferencias-2a-divisao-norte-2026-27",
    ("2026-27", "2div_sul"): "transferencias-2a-divisao-sul-2026-27",
}
SECTIONS = {"PERMANÊNCIAS": "stay", "PERMANENCIAS": "stay", "SAÍDAS": "out", "SAIDAS": "out",
            "ENTRADAS": "in", "TREINADOR": None}

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def fetch(slug):
    f = RAW / "rosters" / f"{slug}.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    r = session.get(f"https://www.hoqueipatins.pt/{slug}/", timeout=60)
    if not r.ok:
        return ""
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(r.text, encoding="utf-8")
    time.sleep(0.5)
    return r.text


def text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def no_nick(s):
    """Drop nicknames in quotes: 'Xavier “Xavi” Malián' -> 'xavier malian'."""
    return plain(re.sub(r"[“\"][^”\"]*[”\"]", " ", str(s)))


def plain(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[\"'“”‘’®()]", " ", s)
    return " ".join(s.split())


TOKEN = re.compile(r"<a[^>]*>\s*(?:<strong>)?\s*<img[^>]*>\s*(?:</strong>)?\s*<br[^>]*>(.*?)</a>|<table[^>]*>(.*?)</table>", re.S)


def parse(page):
    rows, club, section = [], None, None
    for m in TOKEN.finditer(page):
        if m.group(1) is not None:
            name = text(m.group(1))
            if "TRANSFER" not in name.upper():
                club, section = name, None
            continue
        cells = [[text(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
                 for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(2), re.S)]
        flat = " ".join(c for row in cells for c in row if c).strip()
        if flat.upper() in SECTIONS:
            section = SECTIONS[flat.upper()]
            continue
        if not club or not section:
            continue
        for row in cells:
            if not row or not row[0]:
                continue
            player = re.sub(r"^\d+\s*[–-]\s*", "", row[0]).strip()
            if not player or player.upper().startswith(("SAÍDA", "ENTRADA")):
                continue
            rows.append(dict(club=club, player=player, status=section,
                             other_club=row[1] if len(row) > 1 else "",
                             goalkeeper="®" in player))
    return rows


def match_club(club, candidates):
    """Full club name -> our team key (e.g. 'Sport Lisboa e Benfica' -> 'benfica')."""
    full = team_key(club).replace(" ", "")
    hits = [c for c in candidates if c.replace(" ", "") in full]
    if hits:
        # 'benfica b' must not match the A team page and vice versa
        b = re.search(r"\bb$|“b”|\"b\"", club.lower())
        hits = [h for h in hits if h.endswith(" b") == bool(b)] or hits
        # "Sporting Clube de Tomar": prefer the key that matches at the end (the town)
        return max(hits, key=lambda h: (full.rfind(h.replace(" ", "")) + len(h.replace(" ", "")), len(h)))
    best = difflib.get_close_matches(full, [c.replace(" ", "") for c in candidates], n=1, cutoff=0.6)
    return next((c for c in candidates if c.replace(" ", "") == best[0]), None) if best else None


def main():
    m = pd.read_csv(DATA / "matches.csv")
    players = pd.read_csv(DATA / "players.csv")
    players["key"] = players.name.map(plain)
    by_key = players.groupby("key").player_id.apply(list).to_dict()
    players["nn"] = players.name.map(no_nick)
    by_nn = players.groupby("nn").player_id.apply(list).to_dict()
    # which players appeared for which team in which season (to tell apart players with the same name)
    st = pd.read_csv(DATA / "api_player_stats.csv")[["event_id", "team_id", "player_id"]]
    am = pd.read_csv(DATA / "api_matches.csv")[["event_id", "season", "home_id", "away_id", "home", "away"]]
    st = st.merge(am, on="event_id")
    st["team"] = st.home.where(st.team_id == st.home_id, st.away)
    played_for = st.groupby(["player_id"]).apply(lambda g: set(zip(g.season, g.team))).to_dict()
    all_teams = set(am.home) | set(am.away)
    out = []
    for (season, league), slug in PAGES.items():
        page = fetch(slug)
        rows = parse(page)
        teams = set(m[(m.season == season) & (m.competition == league)].home)
        for r in rows:
            r.update(season=season, league=league, team=match_club(r["club"], teams))
            k = plain(r["player"])
            ids = by_key.get(k)
            if not ids:  # all words of the listed name inside a known name (nicknames, order)
                words = set(k.split())
                ids = [pid for kk, pids in by_key.items() if words and words <= set(kk.split()) for pid in pids]
            if not ids or len(ids) > 1:  # same name without nicknames
                ids = by_nn.get(no_nick(r["player"])) or ids
            if ids and len(ids) > 1:  # same name: who played for this club (stay) / the club he came from (in)
                prev = f"{int(season[:4]) - 1}-{season[2:4]}"
                club_from = match_club(r["other_club"] or "", all_teams) if r["status"] == "in" else r["team"]
                fit = [i for i in ids if (prev, club_from) in played_for.get(i, set())
                       or (prev, r["team"]) in played_for.get(i, set())]
                ids = fit if len(fit) == 1 else ids
            r["player_id"] = ids[0] if ids and len(ids) == 1 else None
        out += rows
        n_team = len({r["club"] for r in rows})
        unmatched = sorted({r["club"] for r in rows if not r["team"]})
        print(f"{season} {league:10s} {len(rows):4d} entries, {n_team} clubs"
              + (f", clubs not matched: {unmatched}" if unmatched else ""))
    df = pd.DataFrame(out)
    df.to_csv(DATA / "rosters.csv", index=False)
    print(f"player names matched to ids: {df.player_id.notna().mean():.0%} "
          f"(stay {df[df.status == 'stay'].player_id.notna().mean():.0%}, "
          f"in {df[df.status == 'in'].player_id.notna().mean():.0%})")


if __name__ == "__main__":
    main()
