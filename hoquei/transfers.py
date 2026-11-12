"""Season-over-season squad changes (transfers) as a feature for the match model.

For every team and season (2022/23 onwards, so that a previous season with line-ups exists):
  * player ratings are fitted once at the start of the season, on earlier games only
    (attack/defence impact from hoquei/impact.py, and goals per game)
  * last season's squad: each player weighted by how much he played for the team
    (his share of the team's games x his usual weight in a game)
  * this season's squad, at each game: the players called up for the team so far this
    season, this game's squad list included (published before the game), weighted the same way
  * transfer effect = strength(this season's squad) - strength(last season's squad)
The effect fades as the team plays: x exp(-games_played_this_season / FADE_GAMES), because
the team's own new results take over in its rating.

Output: data/features_transfers.csv
  tr_att_h/a, tr_def_h/a   impact-based attack / defence change
  tr_gls_h/a               goals-per-game-based change (log ratio of squad scoring power)
  team_game_h/a            how many games the team had played this season before this one
and data/transfers_summary.csv: biggest squad changes per season (for reports).
"""
import numpy as np
import pandas as pd

from .common import DATA
from . import impact

FADE_GAMES = 8.0
DIV2_GOAL_VALUE = 0.7


def season_start(m, season):
    return m[m.season == season].date.min()


def squad_weights(rows, n_games):
    """rows: player participation of one team in a set of games -> {player: expected weight}.
    weight = (games called up / team games) x mean weight when he played; outfield sums to 4."""
    if n_games == 0 or rows.empty:
        return {}
    g = rows.groupby(["player_id", "goalkeeper"]).agg(n=("event_id", "nunique"), w=("w", "mean")).reset_index()
    g["exp_w"] = g.n / n_games * g.w.where(g.w > 0, 0.3)
    out = {}
    sk = g[~g.goalkeeper]
    tot = sk.exp_w.sum()
    if tot > 0:
        out.update(dict(zip(sk.player_id, sk.exp_w * 4 / tot)))
    gk = g[g.goalkeeper]
    tot = gk.exp_w.sum()
    if tot > 0:
        out.update(dict(zip(gk.player_id, gk.exp_w / tot)))
    return out


def goal_rates(m, p, as_of):
    """Shrunk goals per game of every outfield player before as_of (2nd division goals x 0.7)."""
    pp = p[(p.w > 0) & ~p.goalkeeper].merge(m[["event_id", "date", "competition"]], on="event_id")
    pp = pp[(pp.date < as_of) & (pp.date >= as_of - pd.Timedelta(days=400))]
    goals = pd.to_numeric(pp.golosmarcadostotal, errors="coerce").fillna(0)
    goals = goals * np.where(pp.competition == "1div", 1.0, DIV2_GOAL_VALUE)
    agg = pd.DataFrame({"pid": pp.player_id, "g": goals}).groupby("pid").g.agg(["sum", "count"])
    return ((agg["sum"] + 6 * 0.25) / (agg["count"] + 6)).to_dict()


REGULAR = 0.3   # expected weight per game to count as a regular (outfield sum is 4, goalkeeper 1)


def summarise(season, team, old, new, d_att, d_def, d_goals, ratings):
    """Squad change of one team so far this season: effects and the main players in / out."""
    names = _names()
    imp = lambda pid: sum(ratings.get(pid, (0, 0)))
    ins = sorted([pid for pid, w in new.items() if w >= REGULAR and old.get(pid, 0) < REGULAR / 3], key=imp, reverse=True)
    outs = sorted([pid for pid, w in old.items() if w >= REGULAR and new.get(pid, 0) < REGULAR / 3], key=imp, reverse=True)
    nm = lambda ids: ", ".join(names.get(i, str(i)) for i in ids[:3])
    return dict(season=season, team=team, d_att=d_att, d_def=d_def, d_goals=d_goals,
                arrivals=len(ins), departures=len(outs), main_arrivals=nm(ins), main_departures=nm(outs))


_NAMES = {}


def _names():
    if not _NAMES:
        n = pd.read_csv(DATA / "players.csv")
        _NAMES.update(dict(zip(n.player_id, n.name.str.replace("®", "", regex=False).str.strip())))
    return _NAMES


def main():
    m, p = impact.load_participation()
    seasons = sorted(m.season.unique())
    rows, summary = [], []
    for prev, season in zip(seasons, seasons[1:]):
        start = season_start(m, season)
        ratings, _ = impact.fit(m, p, start)
        rates = goal_rates(m, p, start)
        prev_games = m[m.season == prev]
        cur = m[m.season == season].sort_values(["date", "event_id"])
        pprev = p[p.event_id.isin(prev_games.event_id)]
        called = p[p.convocado > 0]
        old = {}
        for tid in set(prev_games.home_id) | set(prev_games.away_id):
            n = ((prev_games.home_id == tid) | (prev_games.away_id == tid)).sum()
            old[tid] = squad_weights(pprev[pprev.team_id == tid], n)

        def strength(sq):
            att = sum(w * ratings.get(pid, (0, 0))[0] for pid, w in sq.items())
            dfn = sum(w * ratings.get(pid, (0, 0))[1] for pid, w in sq.items())
            gls = sum(w * rates.get(pid, 0.25) for pid, w in sq.items())  # goalkeepers count ~0
            return att, dfn, gls

        seen = {}   # team -> list of event ids this season so far
        latest = {}
        for r in cur.itertuples():
            row = {"event_id": r.event_id}
            for side, tid in (("h", r.home_id), ("a", r.away_id)):
                evs = seen.get(tid, []) + [r.event_id]
                k = len(evs) - 1
                row[f"team_game_{side}"] = k
                if tid in old and old[tid]:
                    new = squad_weights(called[(called.team_id == tid) & called.event_id.isin(evs)], len(evs))
                    if new:
                        oa, od, og = strength(old[tid])
                        na, nd, ng = strength(new)
                        fade = np.exp(-k / FADE_GAMES)
                        row[f"tr_att_{side}"] = (na - oa) * fade
                        row[f"tr_def_{side}"] = (nd - od) * fade
                        row[f"tr_gls_{side}"] = np.log(max(ng, 0.1) / max(og, 0.1)) * fade
                        # latest state of the season for this team (overwritten game by game)
                        latest[tid] = (season, r.home if side == "h" else r.away, old[tid], new,
                                       na - oa, nd - od, ng - og)
                seen[tid] = evs
            rows.append(row)
        for season_, team, o, n, da, dd, dg in latest.values():
            summary.append(summarise(season_, team, o, n, da, dd, dg, ratings))
    f = pd.DataFrame(rows)
    f.to_csv(DATA / "features_transfers.csv", index=False)
    s = pd.DataFrame(summary)
    s["d_total"] = s.d_att + s.d_def
    s.sort_values(["season", "d_total"]).to_csv(DATA / "transfers_summary.csv", index=False)
    print(f.describe().round(3).T[["count", "mean", "std", "min", "max"]])
    return f, s


if __name__ == "__main__":
    main()
