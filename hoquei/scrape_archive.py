"""Download regular-season results from arquivo.hoqueipatins.pt (2015/16 - 2020/21).

Each season has a generated calendar page per league (hp/CAMN_<code>.HTM) that lists
every round with both legs, e.g.

    1ª JORNADA 14ª
    Sábado, 3 de Outubro de 2015   Sábado, 30 de Janeiro de 2016
    UD Oliveirense   3-2   1-4   A Juventude Viana

The second score is printed in the same left/right order, but the game was played at
the right-hand team's rink (checked against the round pages).

Output: data/archive_matches.csv
"""
import html
import re
import time

import pandas as pd
import requests

from .common import DATA, RAW, USER_AGENT, canonical_team

BASE = "https://arquivo.hoqueipatins.pt"
LEAGUE_CODES = {"1": "1div", "2N": "2div_norte", "2S": "2div_sul"}
# 2021-22 onwards comes from the API (scrape_api.py)
SEASONS = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]

MONTHS = {m: i + 1 for i, m in enumerate(
    ["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto",
     "setembro", "outubro", "novembro", "dezembro"])}

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def season_url(season):
    start = int(season[:4])
    return f"{BASE}/{season}" if start >= 2016 else f"{BASE}/anos_anteriores/{season}"


def fetch(url, cache):
    if cache.exists():
        return cache.read_text(encoding="latin-1")
    r = session.get(url, timeout=60)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    time.sleep(0.5)
    return r.content.decode("latin-1")


def cell_text(s):
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def parse_date(text):
    t = text.lower().replace("ç", "c")
    m = re.search(r"(\d{1,2}) de (\w+) de (\d{4})", t)
    if not m or m.group(2) not in MONTHS:
        return None
    return f"{int(m.group(3)):04d}-{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}"


SCORE = re.compile(r"(\d+)\s*-\s*(\d+)")


def parse_calendar(page):
    rows = []
    leg_rounds, leg_dates = (None, None), (None, None)
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S | re.I):
        tds = [cell_text(td) for td in re.findall(r"<td[^>]*>(.*?)(?=<td|$)", tr, flags=re.S | re.I)]
        joined = " ".join(tds)
        hdr = re.search(r"(\d+)ª\s+JORNADA\s+(\d+)ª", joined)
        if hdr:
            leg_rounds = (int(hdr.group(1)), int(hdr.group(2)))
            continue
        dates = re.findall(r"\d{1,2} de \w+ de \d{4}", joined)
        if len(dates) == 2 and len(tds) <= 2:
            leg_dates = (parse_date(dates[0]), parse_date(dates[1]))
            continue
        tds = [t for t in tds if t]
        if len(tds) < 3 or leg_rounds[0] is None:
            continue
        left, right, middle = tds[0], tds[-1], " ".join(tds[1:-1])
        scores = SCORE.findall(middle)
        if SCORE.fullmatch(left) or SCORE.fullmatch(right):
            continue
        # leg 1: left team at home
        s1 = scores[0] if len(scores) >= 1 else None
        s2 = scores[1] if len(scores) >= 2 else None
        rows.append(dict(round=leg_rounds[0], date=leg_dates[0], home=left, away=right,
                         home_goals=int(s1[0]) if s1 else None, away_goals=int(s1[1]) if s1 else None))
        # leg 2: right team at home, score printed left-right
        rows.append(dict(round=leg_rounds[1], date=leg_dates[1], home=right, away=left,
                         home_goals=int(s2[1]) if s2 else None, away_goals=int(s2[0]) if s2 else None))
    return rows


# league page and round-page file name, used when a calendar page was overwritten
LEAGUE_PAGES = {"1div": ("1div.htm", "1.html"), "2div_norte": ("2div_norte.htm", "2_n.html"),
                "2div_sul": ("2div_sul.htm", "2_s.html")}
LOGO_PAIR = re.compile(r"logos/([a-z0-9_]+)_50\.gif(.*?)logos/([a-z0-9_]+)_50\.gif", re.S)


def parse_round_pages(season, comp):
    """Fallback: read each round page ('jornadas/<date>/<n>.html').

    On these pages every game shows the home logo, the score, then the away logo."""
    league_page, round_file = LEAGUE_PAGES[comp]
    base = season_url(season)
    index = fetch(f"{base}/act_nacional/{league_page}", RAW / "archive" / season / league_page)
    folders = sorted(set(re.findall(r"jornadas/([^/\"]+)/" + re.escape(round_file), index)))
    rows = []
    for rnd, folder in enumerate(folders, start=1):
        page = fetch(f"{base}/jornadas/{folder}/{round_file}",
                     RAW / "archive" / season / "jornadas" / folder / round_file)
        date = re.match(r"(\d{4})-(\d{2})_\w+-(\d{2})", folder)
        date = f"{date.group(1)}-{date.group(2)}-{date.group(3)}" if date else None
        for home, middle, away in LOGO_PAIR.findall(page):
            sc = SCORE.search(cell_text(middle))
            rows.append(dict(round=rnd, date=date, home=home.replace("_", " "),
                             away=away.replace("_", " "),
                             home_goals=int(sc.group(1)) if sc else None,
                             away_goals=int(sc.group(2)) if sc else None))
    return rows


def main():
    out = []
    for season in SEASONS:
        for code, comp in LEAGUE_CODES.items():
            page = fetch(f"{season_url(season)}/hp/CAMN_{code}.HTM",
                         RAW / "archive" / season / f"CAMN_{code}.HTM")
            rows = parse_calendar(page)
            # Some calendar pages were later overwritten by the next season's empty calendar
            if not any(r["home_goals"] is not None for r in rows) or \
                    (rows and min(r["date"] or "9999" for r in rows)[:4] > season[:4]):
                rows = parse_round_pages(season, comp)
            for r in rows:
                r.update(source="archive", season=season, competition=comp, phase="regular",
                         played=r["home_goals"] is not None)
            out += rows
            played = sum(r["played"] for r in rows)
            print(f"{season} {comp:10s} {len(rows):4d} games ({played} with result)")
    m = pd.DataFrame(out)
    m["home_name"], m["away_name"] = m.home, m.away
    m["home"] = m.home.map(canonical_team)
    m["away"] = m.away.map(canonical_team)
    m.to_csv(DATA / "archive_matches.csv", index=False)
    print(f"saved {len(m)} games")


if __name__ == "__main__":
    main()
