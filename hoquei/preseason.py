"""Pre-season squad strength from the transfers pages (data/rosters.csv).

For each team and season with a transfers page, before the first game:
  squad = players who stay + players who arrive (data/rosters.csv)
  sq_pre = log(scoring power of that squad / scoring power of last season's squad)
Scoring power = sum of each outfield player's goals per game (shrunk; 2nd division goals x0.7,
using only games before the season) weighted by his expected share of playing time.
Players we have never seen (arrivals from abroad or youth teams) count as average newcomers.

Output: data/preseason_squads.csv (season, team, sq_pre, arrivals, departures)
Used by predict.py --preseason and by the evaluation in main().
"""
import numpy as np
import pandas as pd

from .common import DATA
from . import impact, transfers

NEW_PLAYER_RATE = 0.25


def build():
    m, p = impact.load_participation()
    ros = pd.read_csv(DATA / "rosters.csv")
    ros = ros[ros.team.notna()]
    p_team = p.merge(m[["event_id", "season", "date", "home_id", "away_id", "home", "away"]], on="event_id")
    p_team["team"] = p_team.home.where(p_team.team_id == p_team.home_id, p_team.away)
    rows = []
    for (season, team), r in ros.groupby(["season", "team"]):
        games = m[m.season == season]
        start = games.date.min() if len(games) else pd.Timestamp(f"{season[:4]}-09-15")
        prev = f"{int(season[:4]) - 1}-{season[2:4]}"
        rates = transfers.goal_rates(m, p, start)
        old_games = m[(m.season == prev) & ((m.home == team) | (m.away == team))]
        old = transfers.squad_weights(p_team[(p_team.season == prev) & (p_team.team == team)], len(old_games))
        if not old:
            continue
        # expected weight of each roster player: his share last season (at any club), else 0.5
        share = {}
        for (pid, t), g in p_team[(p_team.season == prev) & p_team.player_id.notna()].groupby(["player_id", "team"]):
            n = len(m[(m.season == prev) & ((m.home == t) | (m.away == t))])
            share[pid] = max(share.get(pid, 0), len(g) / max(n, 1) * g.w.mean())
        squad = r[r.status.isin(["stay", "in"]) & ~r.goalkeeper.astype(bool)]
        new_w, new_rates = [], []
        for pid in squad.player_id:
            known = pd.notna(pid)
            new_w.append(share.get(pid, 0.5) if known else 0.5)
            new_rates.append(rates.get(pid, NEW_PLAYER_RATE) if known else NEW_PLAYER_RATE)
        new_w = np.array(new_w); new_rates = np.array(new_rates)
        if not len(new_w):
            continue
        new_power = float((new_w * 4 / new_w.sum()) @ new_rates)
        old_sk = {pid: w for pid, w in old.items() if pid in set(p[~p.goalkeeper].player_id)}
        old_power = sum(w * rates.get(pid, NEW_PLAYER_RATE) for pid, w in old_sk.items())
        rows.append(dict(season=season, team=team, sq_pre=np.log(max(new_power, .1) / max(old_power, .1)),
                         arrivals=int((r.status == "in").sum()), departures=int((r.status == "out").sum())))
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "preseason_squads.csv", index=False)
    return out


def evaluate_preseason():
    """Predict each team's first 6 league games as if in August: no squad info vs. rosters."""
    from .backtest import evaluate, load_matches, EVAL_LEAGUES
    from .model import Params
    pre = pd.read_csv(DATA / "preseason_squads.csv").set_index(["season", "team"]).sq_pre
    m = load_matches(("features_players.csv",))
    m["team_game_h"] = m.groupby(["season", "competition", "home"]).cumcount()  # rough count
    order = m.sort_values("date")
    seen = {}
    early = []
    for i, r in order.iterrows():
        k_h = seen.get((r.season, r.home), 0); k_a = seen.get((r.season, r.away), 0)
        early.append((i, k_h < 6 or k_a < 6))
        seen[(r.season, r.home)] = k_h + 1; seen[(r.season, r.away)] = k_a + 1
    early = pd.Series(dict(early))
    d = pd.to_datetime(m.date)
    mask = (d >= "2023-08-01") & (d < "2026-08-01") & m.competition.isin(EVAL_LEAGUES) & early.reindex(m.index).values
    mask &= m.apply(lambda r: (r.season, r.home) in pre.index or (r.season, r.away) in pre.index, axis=1)
    res = {}
    for name in ["no squad info", "squad from transfers page", "actual squad lists"]:
        mm = m.copy()
        if name != "actual squad lists":
            mm.loc[mask, ["sq_known_h", "sq_known_a", "gk_known_h", "gk_known_a"]] = 0.0
        if name == "squad from transfers page":
            mm.loc[mask, "sq_known_h"] = [pre.get((s, t), 0.0) for s, t in zip(mm.loc[mask, "season"], mm.loc[mask, "home"])]
            mm.loc[mask, "sq_known_a"] = [pre.get((s, t), 0.0) for s, t in zip(mm.loc[mask, "season"], mm.loc[mask, "away"])]
        r, _ = evaluate(mm, mask, Params(), verbose=False)
        res[name] = r.loc["model", ["n", "logloss", "accuracy"]]
    out = pd.DataFrame(res).T
    print(out.round(4).to_string())
    return out


def main():
    out = build()
    print(out[out.season == out.season.max()].sort_values("sq_pre").round(3).to_string())
    evaluate_preseason()


if __name__ == "__main__":
    main()
