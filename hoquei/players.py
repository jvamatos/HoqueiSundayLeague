"""Player-based features for each game (2021/22 onwards), computed without looking ahead.

The games are processed in date order. Every feature for a game uses only earlier games.

Player ratings (they follow the player across clubs, so transfers are captured):
  * skater scoring rate = recent goals per game, shrunk toward a modest prior; 2nd-division
    goals count less (DIV2_GOAL_VALUE).
  * goalkeeper "goals prevented" per game = (goals the opponent usually scores - goals
    conceded), shrunk toward 0. "Usually scores" = the opponent's recent goals per game.

Team features per game, for both "known" squads (the called-up players, *convocados*, of this
game; usually published before the game) and "expected" squads (the players of the team's
previous game, the information available without the squad list):
  * sq_*   = log(squad scoring power / the team's usual squad scoring power)
             squad scoring power = sum of the scoring rates of the best 8 skaters in the squad
  * gk_*   = starting goalkeeper's goals prevented - the team's usual goalkeeper's value

"usual" = exponentially weighted average over the team's previous games, so the features
measure changes (injuries, suspensions, transfers, a reserve goalkeeper) that the team
rating has not absorbed yet.

Output: data/features_players.csv (event_id, sq_known_h, sq_known_a, gk_known_h, ...)
        data/player_ratings.csv   (latest rating of every player)
"""
import json
import unicodedata

import numpy as np
import pandas as pd

from .common import DATA, team_key

HALF_LIFE = 365.0           # days, for player ratings
TEAM_HALF_LIFE = 120.0      # days, for a team's "usual" squad
PRIOR_GAMES = 6             # shrinkage strength (games)
PRIOR_RATE = 0.25           # goals/game assumed for an unknown skater
DIV2_GOAL_VALUE = 0.7       # a 2nd-division goal is worth this many 1st-division goals
TOP_SKATERS = 8


class Decayed:
    """Exponentially decayed sums: value(t) = sum x_i * 0.5 ** ((t - t_i) / half_life)."""

    def __init__(self, half_life):
        self.hl, self.v, self.t = half_life, {}, {}

    def get(self, key, t, default=0.0):
        if key not in self.v:
            return default
        return self.v[key] * 0.5 ** ((t - self.t[key]) / self.hl)

    def add(self, key, t, x):
        self.v[key] = self.get(key, t) + x
        self.t[key] = t


def build():
    m = pd.read_csv(DATA / "api_matches.csv")
    m = m[m.played.astype(bool)].copy()
    m["day"] = (pd.to_datetime(m.date) - pd.Timestamp("2000-01-01")).dt.days
    p = pd.read_csv(DATA / "api_player_stats.csv")
    p = p[p.event_id.isin(m.event_id)]
    for c in ["convocado", "jogou", "cincoinicial", "golosmarcadostotal", "golossofridostotal"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    by_game = {k: g for k, g in p.groupby(["event_id", "team_id"])}

    goals, games = Decayed(HALF_LIFE), Decayed(HALF_LIFE)           # skaters
    gk_prev, gk_games = Decayed(HALF_LIFE), Decayed(HALF_LIFE)      # goalkeepers
    team_gf, team_gp = Decayed(TEAM_HALF_LIFE), Decayed(TEAM_HALF_LIFE)
    team_sq, team_gk, team_n = Decayed(TEAM_HALF_LIFE), Decayed(TEAM_HALF_LIFE), Decayed(TEAM_HALF_LIFE)
    last_squad = {}  # team_id -> (skaters, starting gk) of its previous game

    def skater_rate(pid, t):
        return (goals.get(pid, t) + PRIOR_GAMES * PRIOR_RATE) / (games.get(pid, t) + PRIOR_GAMES)

    def gk_value(pid, t):
        return gk_prev.get(pid, t) / (gk_games.get(pid, t) + PRIOR_GAMES)

    def squad_power(skaters, t):
        rates = sorted((skater_rate(s, t) for s in skaters), reverse=True)[:TOP_SKATERS]
        rates += [PRIOR_RATE * 0.5] * (TOP_SKATERS - len(rates))  # fewer than 8 players: weak fill-ins
        return sum(rates)

    def squad_of(g):
        if g is None:
            return [], None
        called = g[g.convocado > 0]
        skaters = called[~called.goalkeeper.astype(bool)].player_id.tolist()
        gks = called[called.goalkeeper.astype(bool)]
        start = gks[gks.cincoinicial > 0]
        gk = (start if len(start) else gks).player_id.tolist()
        return skaters, (gk[0] if gk else None)

    rows = []
    for r in m.sort_values(["day", "event_id"]).itertuples():
        t = r.day
        row = {"event_id": r.event_id}
        for side, tid, opp in (("h", r.home_id, r.away_id), ("a", r.away_id, r.home_id)):
            n = team_n.get(tid, t)
            usual_sq = team_sq.get(tid, t) / n if n > 0 else None
            usual_gk = team_gk.get(tid, t) / n if n > 0 else None
            for mode, (sk, gk) in (("known", squad_of(by_game.get((r.event_id, tid)))),
                                   ("exp", last_squad.get(tid, ([], None)))):
                if not sk or usual_sq is None:
                    row[f"sq_{mode}_{side}"] = 0.0
                    row[f"gk_{mode}_{side}"] = 0.0
                    continue
                row[f"sq_{mode}_{side}"] = np.log(squad_power(sk, t) / usual_sq)
                row[f"gk_{mode}_{side}"] = (gk_value(gk, t) if gk else 0.0) - usual_gk
        rows.append(row)

        # ---------- update with this game's results (after computing its features)
        div_w = 1.0 if r.competition == "1div" else DIV2_GOAL_VALUE
        for tid, opp, conceded in ((r.home_id, r.away_id, r.away_goals), (r.away_id, r.home_id, r.home_goals)):
            g = by_game.get((r.event_id, tid))
            sk, gk = squad_of(g)
            if g is not None:
                played = g[(g.jogou > 0) & ~g.goalkeeper.astype(bool)]
                for pid, gl in zip(played.player_id, played.golosmarcadostotal):
                    goals.add(pid, t, div_w * gl)
                    games.add(pid, t, 1.0)
            opp_usual = team_gf.get(opp, t) / team_gp.get(opp, t) if team_gp.get(opp, t) > 0 else None
            if gk and opp_usual is not None:
                gk_prev.add(gk, t, div_w * (opp_usual - conceded))
                gk_games.add(gk, t, 1.0)
            if sk:
                team_sq.add(tid, t, squad_power(sk, t))
                team_gk.add(tid, t, gk_value(gk, t) if gk else 0.0)
                team_n.add(tid, t, 1.0)
                last_squad[tid] = (sk, gk)
        for tid, gf in ((r.home_id, r.home_goals), (r.away_id, r.away_goals)):
            team_gf.add(tid, t, gf)
            team_gp.add(tid, t, 1.0)

    feats = pd.DataFrame(rows)
    feats.to_csv(DATA / "features_players.csv", index=False)

    # latest player ratings (for reports)
    t = m.day.max() + 1
    names = p.drop_duplicates("player_id", keep="last").set_index("player_id")
    pr = pd.DataFrame({"player_id": list(games.v)})
    pr["games_w"] = [games.get(i, t) for i in pr.player_id]
    pr["goals_per_game"] = [skater_rate(i, t) for i in pr.player_id]
    gk_ids = list(gk_games.v)
    gkr = pd.DataFrame({"player_id": gk_ids, "gk_goals_prevented_per_game": [gk_value(i, t) for i in gk_ids],
                        "gk_games_w": [gk_games.get(i, t) for i in gk_ids]})
    pr = pr.merge(gkr, on="player_id", how="outer")
    pr["last_team_id"] = pr.player_id.map(names.team_id)
    keys = dict(zip(m.home_id, m.home)) | dict(zip(m.away_id, m.away))
    pr["last_team"] = pr.last_team_id.map(keys)
    if (DATA / "players.csv").exists():
        pr = pr.merge(pd.read_csv(DATA / "players.csv"), on="player_id", how="left")
    pr.to_csv(DATA / "player_ratings.csv", index=False)

    # latest squad of each team, used to adjust predictions for known absences
    squads = {}
    for tid, (sk, gk) in last_squad.items():
        g = by_game.get(max((k for k in by_game if k[1] == tid), key=lambda k: k[0]))
        gks = g[g.goalkeeper.astype(bool) & (g.convocado > 0)].player_id.tolist() if g is not None else []
        squads[keys.get(tid, str(tid))] = {"skaters": [int(x) for x in sk], "gk": int(gk) if gk else None,
                                          "gks": [int(x) for x in gks]}
    (DATA / "team_squads.json").write_text(json.dumps(squads, indent=1))
    return feats


def _plain(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(w for w in s.replace('"', " ").replace("“", " ").replace("”", " ").split() if w != "®")


def absence_features(team, absent_names):
    """Features for a team that misses some players, relative to its latest squad.

    Returns (sq, gk, matched names). sq = log(squad power without them / with them);
    gk = backup goalkeeper value - starting goalkeeper value (if the starter is out).
    """
    squads = json.loads((DATA / "team_squads.json").read_text())
    pr = pd.read_csv(DATA / "player_ratings.csv").set_index("player_id")
    sq = squads.get(team_key(team))
    if sq is None:
        raise ValueError(f"no squad known for {team}")
    names = {pid: _plain(pr.name.get(pid, "")) for pid in sq["skaters"] + sq["gks"]}
    out = set()
    for a in absent_names:
        a = _plain(a)
        hits = [pid for pid, n in names.items() if a and (a in n or all(w in n for w in a.split()))]
        if not hits:
            raise ValueError(f"'{a}' not found in {team}'s latest squad: "
                             + ", ".join(pr.name.get(p, str(p)) for p in names))
        out.update(hits)

    def power(skaters):
        rates = sorted((pr.goals_per_game.get(s, PRIOR_RATE) for s in skaters), reverse=True)[:TOP_SKATERS]
        return sum(rates + [PRIOR_RATE * 0.5] * (TOP_SKATERS - len(rates)))

    full = power(sq["skaters"])
    sqf = np.log(power([s for s in sq["skaters"] if s not in out]) / full)
    gkf = 0.0
    if sq["gk"] in out:
        backups = [g for g in sq["gks"] if g not in out]
        val = lambda g: pr.gk_goals_prevented_per_game.get(g, 0.0) if g is not None else -0.5
        gkf = (val(backups[0]) if backups else -0.5) - val(sq["gk"])
        gkf = 0.0 if np.isnan(gkf) else gkf
    return sqf, gkf, [pr.name.get(p, str(p)) for p in out]


def main():
    f = build()
    print(f"{len(f)} games with player features")
    print(f.describe().round(3).T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
