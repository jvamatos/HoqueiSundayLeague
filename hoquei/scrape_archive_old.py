"""Regular-season results 2007/08 - 2014/15 from arquivo.hoqueipatins.pt.

Two layouts:
  * 2007/08 - 2012/13: one "all rounds" page per league (<league>_jornadas_<yy-yy>.htm),
    with rows   [ "1ª DIVISÃO - 1ª Jornada" ] [ "29 de Setembro de 2007" ] [home | 3-1 | away]
  * 2013/14 - 2014/15: one page per round (jornadas/<date>/<n>.html) with
    [home logo] score [away logo], parsed like the 2016/17 fallback.

Output: data/archive_old_matches.csv
"""
import re

import pandas as pd

from .common import DATA, RAW, canonical_team, team_key
from .scrape_archive import BASE, SCORE, cell_text, fetch, parse_date, LOGO_PAIR

LEAGUES = {"1div": "1div", "2div_norte": "2div_norte", "2div_sul": "2div_sul"}
ROUND_FILES = {"1div": "1.html", "2div_norte": "2_n.html", "2div_sul": "2_s.html"}
SEASONS_A = ["2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13"]
SEASONS_B = ["2013-14", "2014-15"]


def base(season):
    return f"{BASE}/anos_anteriores/{season}"


NOT_TEAM = re.compile(r"^(\d{1,2}[:h]\d{2}|\d{1,2} \w{3}\.?|Aç|Ma|Md|TV|\(.*\)|-|\d+)$", re.I)


def _team_cell(t):
    return bool(t) and not NOT_TEAM.match(t) and not parse_date(t) and not SCORE.fullmatch(t)


# "3-1", "21-Out 3 -1", "12-2 Benfica TV"
SCORE_CELL = re.compile(r"^(?:(\d{1,2})[-/ ]([A-Za-zç]{3})\.?\s+)?(\d+)\s*-\s*(\d+)(?:\s+[^\d\s].*)?$")
DAY_MON = re.compile(r"^(\d{1,2})[ -]([A-Za-zç]{3})\.?$")


def _resched(day, mon, round_date):
    """'21-Out' printed next to a score = the game was actually played that day."""
    mm = MONTH_ABBR.get(mon.lower()[:3].replace("ç", "c"))
    if not mm or not round_date:
        return round_date
    y = int(round_date[:4])
    # pick the year that puts the date closest to the scheduled round date
    best = min((abs((pd.Timestamp(yy, mm, min(int(day), 28)) - pd.Timestamp(round_date)).days), yy)
               for yy in (y - 1, y, y + 1))[1]
    try:
        return str(pd.Timestamp(best, mm, int(day)).date())
    except ValueError:
        return round_date


def parse_all_rounds(page):
    """Rows look like [time] [home] [3-1] [away] (time/notes optional)."""
    rows, rnd, date = [], None, None
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S | re.I):
        tds = [cell_text(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S | re.I)]
        joined = " ".join(tds)
        hdr = re.search(r"(\d+)ª\s*Jornada", joined, flags=re.I)
        scores = [i for i, t in enumerate(tds) if SCORE_CELL.match(t.strip())]
        if hdr and not scores:
            rnd = int(hdr.group(1))
            date = parse_date(joined) or date
            continue
        if not scores:
            if parse_date(joined) and len(joined) < 60:
                date = parse_date(joined)
            continue
        if rnd is None or len(scores) != 1:
            continue
        k = scores[0]
        left = [t for t in tds[:k] if _team_cell(t)]
        right = [t for t in tds[k + 1:] if _team_cell(t)]
        if not left or not right:
            continue
        sc = SCORE_CELL.match(tds[k].strip())
        played_on = _resched(sc.group(1), sc.group(2), date) if sc.group(1) else date
        for t in tds[:k]:  # separate "25 Jan" cell = actual date of a rescheduled game
            dm = DAY_MON.match(t.strip())
            if dm:
                played_on = _resched(dm.group(1), dm.group(2), date)
        rows.append(dict(round=rnd, date=played_on, home=left[-1], away=right[0],
                         home_goals=int(sc.group(3)), away_goals=int(sc.group(4))))
    return rows


def season_a(season, comp):
    yy = season[2:4] + "-" + season[5:7]
    sub = "" if season == "2007-08" else "act_nacional/"
    url = f"{base(season)}/{sub}{comp}_jornadas_{yy}.htm"
    page = fetch(url, RAW / "archive_old" / season / f"{comp}_jornadas.htm")
    return parse_all_rounds(page)


MONTH_ABBR = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6, "jul": 7, "ago": 8,
              "set": 9, "out": 10, "nov": 11, "dez": 12}


def folder_date(folder):
    # "2013_10_out_26" or "2014-10_out-04"
    m = re.match(r"(\d{4})[_-](\d{2})[_-][a-z]{3}[_-](\d{2})", folder)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def season_b(season, comp):
    index = fetch(f"{base(season)}/act_nacional/{comp}.htm", RAW / "archive_old" / season / f"{comp}.htm")
    rf = ROUND_FILES[comp]
    folders = sorted(set(re.findall(r"jornadas/([^/\"]+)/" + re.escape(rf), index)),
                     key=lambda f: folder_date(f) or f)
    rows = []
    for rnd, folder in enumerate(folders, start=1):
        try:
            page = fetch(f"{base(season)}/jornadas/{folder}/{rf}",
                         RAW / "archive_old" / season / "jornadas" / folder / rf)
        except Exception as e:
            print(f"  missing round page {folder}/{rf}: {e.__class__.__name__}")
            continue
        for home, middle, away in LOGO_PAIR.findall(page):
            sc = SCORE.search(cell_text(middle))
            rows.append(dict(round=rnd, date=folder_date(folder), home=home.replace("_", " "),
                             away=away.replace("_", " "),
                             home_goals=int(sc.group(1)) if sc else None,
                             away_goals=int(sc.group(2)) if sc else None))
    # Round pages name teams by logo, and B teams share the main club's logo, so the
    # results matrix (text names) is authoritative; round pages only provide dates.
    return fill_from_matrix(season, comp, rows, folders)


QRE_CODES = {"1div": "1", "2div_norte": "2N", "2div_sul": "2S"}


def fill_from_matrix(season, comp, rows, folders):
    """hp/_QRE_<code>.HTM is a home(row) x away(column) results grid without dates.
    Dates: known from the round pages, otherwise the return fixture's round + (n-1)."""
    page = fetch(f"{base(season)}/hp/_QRE_{QRE_CODES[comp]}.HTM",
                 RAW / "archive_old" / season / f"_QRE_{QRE_CODES[comp]}.HTM")
    grid = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S | re.I):
        tds = [cell_text(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S | re.I)]
        if len(tds) > 3 and "EQUIPAS" not in tds[0]:
            grid.append(tds)
    teams = [r[0] for r in grid]
    n = len(teams)
    # map logo-based names from the round pages onto the matrix names
    import difflib
    mkeys = [team_key(t) for t in teams]
    def to_matrix(name):
        k = team_key(name)
        if k in mkeys:
            return k
        best = difflib.get_close_matches(k, [x for x in mkeys if not x.endswith(" b")], n=1, cutoff=0.5)
        return best[0] if best else k
    known = {(to_matrix(r["home"]), to_matrix(r["away"])): r for r in rows}
    out = []
    for i, row in enumerate(grid):
        for j, cell in enumerate(row[1:n + 1]):
            sc = SCORE.fullmatch(cell.replace(" ", ""))
            if i == j or not sc:
                continue
            h, a = teams[i], teams[j]
            k = (team_key(h), team_key(a))
            kk = k if k in known else tuple(re.sub(r" b$", "", x) for x in k)
            if kk in known:
                rnd, date = known[kk]["round"], known[kk]["date"]
            else:
                rev = known.get((kk[1], kk[0]))
                rnd = rev["round"] + n - 1 if rev else None
                date = folder_date(folders[rnd - 1]) if rnd and rnd <= len(folders) else None
            out.append(dict(round=rnd, date=date, home=h, away=a,
                            home_goals=int(sc.group(1)), away_goals=int(sc.group(2))))
    # games whose round is unknown: spread them over the rounds after the last known date
    dated = sorted(r["date"] for r in out if r["date"])
    later = [d for d in (folder_date(f) for f in folders) if d and dated and d > dated[-1]] or dated[-1:]
    undated = [r for r in out if not r["date"]]
    for i, r in enumerate(undated):
        r["date"] = later[int(i * len(later) / max(len(undated), 1))]
    print(f"  {season} {comp}: filled from results matrix ({len(rows)} -> {len(out)} games, "
          f"{len(undated)} with approximate date)")
    return out


def main():
    out = []
    for season in SEASONS_A + SEASONS_B:
        for comp in LEAGUES:
            try:
                rows = season_a(season, comp) if season in SEASONS_A else season_b(season, comp)
            except Exception as e:  # keep going; report what failed
                print(f"{season} {comp}: FAILED {e}")
                continue
            # the same game can appear twice (e.g. a postponed-game note); keep the first
            seen, uniq = set(), []
            for r in rows:
                k = (r["home"], r["away"], r["round"])
                if k not in seen:
                    seen.add(k); uniq.append(r)
            for r in uniq:
                r.update(source="archive_old", season=season, competition=comp, phase="regular",
                         played=r["home_goals"] is not None)
            out += uniq
            teams = {r["home"] for r in uniq} | {r["away"] for r in uniq}
            print(f"{season} {comp:10s} {len(uniq):4d} games, {sum(r['played'] for r in uniq)} with result, "
                  f"{len(teams)} teams")
    m = pd.DataFrame(out)
    m["home_name"], m["away_name"] = m.home, m.away
    m["home"] = m.home.map(canonical_team)
    m["away"] = m.away.map(canonical_team)
    m.to_csv(DATA / "archive_old_matches.csv", index=False)
    print(f"saved {len(m)} games")


if __name__ == "__main__":
    main()
