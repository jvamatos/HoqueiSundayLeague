"""Power plays, short-handed play, blue cards and referees (play-by-play from 2022/23).

1. Per game and team (data/game_discipline.csv): blue/red cards, power-play (PP)
   opportunities, PP minutes, PP goals, short-handed goals, penalties, direct free hits.
2. Summaries (data/discipline_*.csv): league, team-season (PP%, PK%, cards) and referees.
3. Leak-free features per game (data/features_discipline.csv), using only earlier games:
   * ppx_h / ppx_a: expected extra goals from power plays
       = expected PP chances (opponent's blue-card rate x referee strictness)
         x expected conversion (own PP skill x opponent's penalty-kill weakness)
         - the league average of that product
   * ref_h = ref_a: log(referee pair's blue cards per game / league average)
"""
import numpy as np
import pandas as pd

from .common import DATA
from .players import Decayed

HALF_LIFE = 365.0
K_GAMES = 8        # shrinkage (games) for card rates
K_PP = 15          # shrinkage (power plays) for conversion rates
K_REF = 10         # shrinkage (games) for referees

BLUE = {"cartao_azul", "cartao_azul_jogo_parado"}
RED = {"cartao_vermelho", "cartao_vermelho_jogo_parado"}


def clock(period, minute, second):
    """Elapsed seconds in the game (25-minute halves; overtime after 50')."""
    if pd.isna(minute):
        return np.nan
    try:
        base = {1: 0, 2: 1500}[int(float(period))]
    except (ValueError, TypeError, KeyError):
        base = 3000  # overtime
    return base + 60 * float(minute) + (0 if pd.isna(second) else float(second))


def per_game():
    ev = pd.read_csv(DATA / "events.csv")
    ev["t"] = [clock(p, m, s) for p, m, s in zip(ev.period, ev.minute, ev.second)]
    ev["goal"] = ev.icon.str.startswith("golo")
    has_pbp = set(ev[ev.icon.isin(BLUE | RED | {"inicio_power_play"}) | ev.goal].event_id)
    rows = []
    for eid, g in ev[ev.event_id.isin(has_pbp)].groupby("event_id"):
        windows = {"home": [], "away": []}
        marked = (g.icon == "inicio_power_play").any()
        if marked:  # explicit power-play markers (recent seasons): start -> matching end
            for side in ("home", "away"):
                starts = g[(g.icon == "inicio_power_play") & (g.side == side)].t.dropna().tolist()
                ends = sorted(g[(g.icon == "fim_power_play") & (g.side == side)].t.dropna().tolist())
                for s in starts:
                    e = next((x for x in ends if x >= s), s + 120)
                    if e in ends:
                        ends.remove(e)
                    windows[side].append((s, e if e > s else s + 120))
        else:  # infer from cards: blue = 2 min, red = 4 min for the opponent; same-time cards cancel
            cards = g[g.icon.isin(BLUE | RED) & g.t.notna()]
            for t0, c in cards.groupby("t"):
                n = {sd: ((c.side == sd) & c.icon.isin(BLUE)).sum() * 120 +
                         ((c.side == sd) & c.icon.isin(RED)).sum() * 240 for sd in ("home", "away")}
                diff = n["home"] - n["away"]
                if diff > 0:
                    windows["away"].append((t0, t0 + diff))
                elif diff < 0:
                    windows["home"].append((t0, t0 - diff))
        rec = {"event_id": eid, "pp_marked": marked}
        for side, opp in (("home", "away"), ("away", "home")):
            mine = g[g.side == side]
            goals = mine[mine.goal]
            in_pp = goals.t.apply(lambda t: any(a <= t <= b for a, b in windows[side]))
            in_sh = goals.t.apply(lambda t: any(a <= t <= b for a, b in windows[opp]))
            s = side[0]
            rec.update({
                f"blue_{s}": mine.icon.isin(BLUE).sum(),
                f"red_{s}": mine.icon.isin(RED).sum(),
                f"pp_opps_{s}": len(windows[side]),
                f"pp_minutes_{s}": sum(b - a for a, b in windows[side]) / 60,
                f"pp_goals_{s}": int((goals.icon.str.contains("power_play") | in_pp).sum()),
                f"sh_goals_{s}": int((in_sh & ~goals.icon.str.contains("power_play")).sum()),
                f"goals_pbp_{s}": len(goals),
                f"pen_goals_{s}": goals.icon.str.contains("penalti").sum(),
                f"pen_att_{s}": mine.icon.str.contains("penalti").sum(),
                f"fh_goals_{s}": goals.icon.str.contains("livre").sum(),
                f"fh_att_{s}": mine.icon.str.contains("livre").sum(),
            })
        rows.append(rec)
    gd = pd.DataFrame(rows)
    m = pd.read_csv(DATA / "api_matches.csv")[["event_id", "season", "competition", "date", "home", "away",
                                                "home_goals", "away_goals"]]
    gd = m.merge(gd, on="event_id")
    gd.to_csv(DATA / "game_discipline.csv", index=False)
    return gd


def summaries(gd):
    long = pd.concat([
        gd.assign(team=gd.home, opp=gd.away, **{c[:-2]: gd[c] for c in gd if c.endswith("_h")},
                  pp_goals_against=gd.pp_goals_a, pp_opps_against=gd.pp_opps_a, sh_goals_against=gd.sh_goals_a),
        gd.assign(team=gd.away, opp=gd.home, **{c[:-2]: gd[c] for c in gd if c.endswith("_a")},
                  pp_goals_against=gd.pp_goals_h, pp_opps_against=gd.pp_opps_h, sh_goals_against=gd.sh_goals_h),
    ])
    league = long.groupby(["season", "competition"]).agg(
        games=("event_id", "nunique"), blue_per_team_game=("blue", "mean"),
        pp_per_team_game=("pp_opps", "mean"), pp_minutes=("pp_minutes", "sum"),
        pp_goals=("pp_goals", "sum"), pp_opps=("pp_opps", "sum"), sh_goals=("sh_goals", "sum"),
        goals=("goals_pbp", "sum"))
    league["pp_conversion"] = league.pp_goals / league.pp_opps
    league["goals_per_min_pp"] = league.pp_goals / league.pp_minutes
    even_min = league.games * 2 * 50 - league.pp_minutes  # team-minutes not on PP (approx.)
    league["goals_per_min_other"] = (league.goals - league.pp_goals) / even_min
    league["pp_goal_rate_x"] = league.goals_per_min_pp / league.goals_per_min_other
    league.round(3).to_csv(DATA / "discipline_league.csv")

    team = long.groupby(["season", "competition", "team"]).agg(
        games=("event_id", "nunique"), blue=("blue", "sum"), red=("red", "sum"),
        pp_opps=("pp_opps", "sum"), pp_goals=("pp_goals", "sum"), sh_goals=("sh_goals", "sum"),
        pk_opps=("pp_opps_against", "sum"), pp_goals_against=("pp_goals_against", "sum"),
        sh_goals_against=("sh_goals_against", "sum"))
    team["blue_per_game"] = team.blue / team.games
    team["pp_pct"] = team.pp_goals / team.pp_opps
    team["pk_pct"] = 1 - team.pp_goals_against / team.pk_opps
    team.round(3).to_csv(DATA / "discipline_teams.csv")

    refs = pd.read_csv(DATA / "referees.csv")
    refs = refs[refs.role == "Principal"].merge(gd[["event_id", "season", "blue_h", "blue_a", "pp_opps_h",
                                                     "pp_opps_a", "home_goals", "away_goals"]], on="event_id")
    refs["blue"] = refs.blue_h + refs.blue_a
    refs["goals"] = refs.home_goals + refs.away_goals
    rt = refs.groupby("name").agg(games=("event_id", "nunique"), blue_per_game=("blue", "mean"),
                                  goals_per_game=("goals", "mean")).sort_values("games", ascending=False)
    rt.round(2).to_csv(DATA / "discipline_referees.csv")
    return league, team, rt


def features(gd):
    """Leak-free power-play / referee features, processed in date order."""
    refs = pd.read_csv(DATA / "referees.csv")
    refs = refs[refs.role == "Principal"].groupby("event_id").name.apply(lambda s: tuple(sorted(s))).to_dict()
    m = pd.read_csv(DATA / "api_matches.csv")
    m = m[m.played.astype(bool)].copy()
    m["day"] = (pd.to_datetime(m.date) - pd.Timestamp("2000-01-01")).dt.days
    gd = gd.set_index("event_id")
    blue, games = Decayed(HALF_LIFE), Decayed(HALF_LIFE)
    ppg, ppo = Decayed(HALF_LIFE), Decayed(HALF_LIFE)          # own PP goals / chances
    pkg, pko = Decayed(HALF_LIFE), Decayed(HALF_LIFE)          # PP goals conceded / chances conceded
    rblue, rgames = Decayed(HALF_LIFE), Decayed(HALF_LIFE)     # per referee
    lg = Decayed(HALF_LIFE)                                    # league totals (keys)

    rows = []
    for r in m.sort_values(["day", "event_id"]).itertuples():
        t, eid = r.day, r.event_id
        n_g = lg.get("games", t)
        row = {"event_id": eid}
        if n_g > 20:
            avg_blue = lg.get("blue", t) / n_g                # blue cards per team-game
            avg_conv = lg.get("ppg", t) / max(lg.get("ppo", t), 1)
            avg_opps = lg.get("ppo", t) / n_g                  # PP chances per team-game
            ref = refs.get(eid, ())
            rf = [(rblue.get(x, t) + K_REF * 2 * avg_blue) / (rgames.get(x, t) + K_REF) for x in ref]
            ref_factor = (np.mean(rf) / (2 * avg_blue)) if rf else 1.0

            def disc(team):  # expected blue cards of team, relative to average
                return (blue.get(team, t) + K_GAMES * avg_blue) / (games.get(team, t) + K_GAMES) / avg_blue

            def conv(team):  # PP conversion relative to average
                return (ppg.get(team, t) + K_PP * avg_conv) / (ppo.get(team, t) + K_PP) / avg_conv

            def pk_weak(team):  # PP goals allowed per chance, relative to average
                return (pkg.get(team, t) + K_PP * avg_conv) / (pko.get(team, t) + K_PP) / avg_conv

            base = avg_opps * avg_conv
            for s, team, opp in (("h", r.home, r.away), ("a", r.away, r.home)):
                exp_pp_goals = avg_opps * disc(opp) * ref_factor * avg_conv * conv(team) * pk_weak(opp)
                row[f"ppx_{s}"] = exp_pp_goals - base
                row[f"ref_{s}"] = np.log(ref_factor)
        rows.append(row)

        if eid not in gd.index:
            continue
        x = gd.loc[eid]
        for s, team, o in (("h", r.home, "a"), ("a", r.away, "h")):
            blue.add(team, t, x[f"blue_{s}"]); games.add(team, t, 1)
            ppg.add(team, t, x[f"pp_goals_{s}"]); ppo.add(team, t, x[f"pp_opps_{s}"])
            pkg.add(team, t, x[f"pp_goals_{o}"]); pko.add(team, t, x[f"pp_opps_{o}"])
        lg.add("games", t, 2); lg.add("blue", t, x.blue_h + x.blue_a)
        lg.add("ppg", t, x.pp_goals_h + x.pp_goals_a); lg.add("ppo", t, x.pp_opps_h + x.pp_opps_a)
        for ref in refs.get(eid, ()):
            rblue.add(ref, t, x.blue_h + x.blue_a); rgames.add(ref, t, 1)
    f = pd.DataFrame(rows)
    f.to_csv(DATA / "features_discipline.csv", index=False)
    return f


def main():
    gd = per_game()
    league, team, refs = summaries(gd)
    f = features(gd)
    print(f"{len(gd)} games with play-by-play")
    print(league.round(3).to_string())
    print(f.describe().round(3).T[["count", "mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
