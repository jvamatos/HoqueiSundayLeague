"""Download the play-by-play ("Jogada a jogada") and referees of every played game
from hoqueipatins.pt match pages (/evento/<id>/), for games in data/api_matches.csv.

Only the play-by-play and referee part of each page is cached (data/raw/events/<id>.html.gz).
Then it is parsed into:
  data/events.csv    one row per action (goal, blue/red card, timeout, 10th foul, ...)
  data/referees.csv  one row per game and official
"""
import argparse
import gzip
import html
import re
import time

import pandas as pd
import requests

from .common import DATA, RAW, USER_AGENT

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT
CACHE = RAW / "events"


def fetch(event_id):
    f = CACHE / f"{event_id}.html.gz"
    if f.exists():
        return gzip.decompress(f.read_bytes()).decode("utf-8")
    for attempt in range(4):
        try:
            r = session.get(f"https://www.hoqueipatins.pt/evento/{event_id}/", timeout=60)
            if r.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    else:
        return None
    page = r.text
    i = page.find('id="painel-lance"')
    j = page.find("Outros jogos da jornada", i)
    part = page[i:j if j > 0 else i + 300000] if i >= 0 else ""
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(gzip.compress(part.encode("utf-8")))
    time.sleep(0.3)
    return part


def _text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


TIME = re.compile(r"(\d+)'(?:\s*(\d+)'')?\s*(?:(\d)ªP|(TE|Pr))", re.I)
TOKEN = re.compile(r'class="tl-side tl-(home|away)"|<span class="time">([^<]*)</span>|'
                   r'<div class="tl-item"[^>]*>(.*?)</div>', re.S)


def parse_time(t):
    m = TIME.search(t or "")
    if not m:
        return None, None, None
    period = int(m.group(3)) if m.group(3) else m.group(4).upper()
    return period, int(m.group(1)), int(m.group(2)) if m.group(2) else 0


def parse(event_id, part):
    """Timeline rows are: [home side items] [time] [away side items]."""
    actions, side, pending, last_time = [], None, [], None
    for tok in TOKEN.finditer(part):
        if tok.group(1):
            if tok.group(1) == "home":
                side, pending = "home", []
            else:
                side = "away"
        elif tok.group(2) is not None:
            last_time = tok.group(2)
            for a in pending:  # home items get the time printed after them
                a["period"], a["minute"], a["second"] = parse_time(last_time)
            pending = []
        else:
            item = tok.group(3)
            icon = re.search(r"actions/([a-z0-9_]+)\.png", item)
            if not icon or icon.group(1).startswith("inicial") or icon.group(1) == "info":
                continue
            a = dict(event_id=event_id, icon=icon.group(1), side=side, period=None, minute=None,
                     second=None, text=_text(item))
            if side == "away":
                a["period"], a["minute"], a["second"] = parse_time(last_time)
            elif side == "home":
                pending.append(a)
            actions.append(a)
    refs = []
    arb = part.find("Arbitragem")
    if arb >= 0:
        t = _text(part[arb:arb + 6000])
        for role, nm in re.findall(r"(Principal|Mesa|Auxiliar)\s+(.+?)(?=\s+(?:Principal|Mesa|Auxiliar)\b|$)", t):
            refs.append(dict(event_id=event_id, role=role, name=nm.strip()))
    return actions, refs


def main(limit=None, parse_only=False):
    m = pd.read_csv(DATA / "api_matches.csv")
    ids = m[m.played.astype(bool)].event_id.astype(int).tolist()[:limit]
    if parse_only:  # only games already downloaded
        ids = [i for i in ids if (CACHE / f"{i}.html.gz").exists()]
    from concurrent.futures import ThreadPoolExecutor
    acts, refs = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:  # a few parallel requests, still polite
        for n, (eid, part) in enumerate(zip(ids, pool.map(fetch, ids)), 1):
            if part:
                a, r = parse(eid, part)
                acts += a; refs += r
            if n % 100 == 0:
                print(f"{n}/{len(ids)}", flush=True)
    pd.DataFrame(acts).to_csv(DATA / "events.csv", index=False)
    pd.DataFrame(refs).to_csv(DATA / "referees.csv", index=False)
    print(f"saved {len(acts)} actions, {len(refs)} referee rows for {len(ids)} games")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--parse-only", action="store_true", help="parse cached pages, download nothing")
    a = ap.parse_args()
    main(a.limit, a.parse_only)
