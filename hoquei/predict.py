"""Predict upcoming games, rank teams and simulate the rest of the season.

    python -m hoquei.predict                # as of today
    python -m hoquei.predict --as-of 2026-11-12 --days 10 --sims 20000

Writes data/predictions/<as_of>/{upcoming.csv, ratings.csv, season_<league>.csv, report.md}
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd

from .common import DATA, team_key
from .players import absence_features
from .model import Params, TeamModel, division

LEAGUES = {"1div": "1ª Divisão", "2div_norte": "2ª Divisão Norte", "2div_sul": "2ª Divisão Sul"}
# Regular-season zones (based on recent seasons' rules; adjust if the format changes)
ZONES = {"1div": {"playoff_top8": 8, "relegation_bottom3": -3},
         "2div_norte": {"top2": 2, "bottom3": -3},
         "2div_sul": {"top2": 2, "bottom3": -3}}


_NAMES = {}


def name(key):
    if not _NAMES and (DATA / "teams.csv").exists():
        _NAMES.update(pd.read_csv(DATA / "teams.csv").set_index("team").display.to_dict())
    return _NAMES.get(key, str(key).title())


def simulate_season(model, games, comp, sims, rng):
    played = games[games.played]
    todo = games[~games.played]
    teams = sorted(set(games.home) | set(games.away))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    pts = np.zeros(T); gd = np.zeros(T)
    for r in played.itertuples():
        h, a = idx[r.home], idx[r.away]
        pts[h] += 3 if r.home_goals > r.away_goals else 1 if r.home_goals == r.away_goals else 0
        pts[a] += 3 if r.away_goals > r.home_goals else 1 if r.home_goals == r.away_goals else 0
        gd[h] += r.home_goals - r.away_goals; gd[a] += r.away_goals - r.home_goals
    P = np.tile(pts, (sims, 1)); G = np.tile(gd, (sims, 1))
    for r in todo.itertuples():
        lh, la = model.expected_goals(r.home, r.away, comp)
        hg, ag = rng.poisson(lh, sims), rng.poisson(la, sims)
        h, a = idx[r.home], idx[r.away]
        P[:, h] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
        P[:, a] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
        G[:, h] += hg - ag; G[:, a] += ag - hg
    # rank by points, then goal difference, then random (head-to-head not modelled)
    key = P * 1e6 + G * 1e2 + rng.random(P.shape)
    pos = (-key).argsort(1).argsort(1) + 1  # 1 = first
    out = pd.DataFrame({"team": teams, "points_now": pts.astype(int),
                        "played": [((played.home == t) | (played.away == t)).sum() for t in teams],
                        "exp_points": P.mean(0).round(1), "p_champion": (pos == 1).mean(0)})
    for zname, n in ZONES[comp].items():
        out[f"p_{zname}"] = (pos <= n).mean(0) if n > 0 else (pos > T + n).mean(0)
    out["avg_position"] = pos.mean(0).round(1)
    return out.sort_values(["exp_points", "p_champion"], ascending=False).reset_index(drop=True)


FEATURE_FILES = ("features_players.csv", "features_discipline.csv")


def apply_absences(up, spec):
    """Set squad/goalkeeper features of upcoming games for teams with known absences."""
    notes = []
    for part in filter(None, (x.strip() for x in spec.split(";"))):
        team, _, players = part.partition(":")
        sq, gk, found = absence_features(team.strip(), [p.strip() for p in players.split(",") if p.strip()])
        key = team_key(team)
        games = up[(up.home == key) | (up.away == key)].sort_values("date")
        if games.empty:
            continue
        nxt = games.index[0]  # absences apply to the team's next game only
        side = "h" if up.at[nxt, "home"] == key else "a"
        up.loc[nxt, f"sq_known_{side}"] = sq
        up.loc[nxt, f"gk_known_{side}"] = gk
        notes.append(f"{name(key)} without {', '.join(found)}: squad power {100 * (np.exp(sq) - 1):+.0f}%"
                     + (f", goalkeeper change {gk:+.2f} goals/game" if gk else ""))
        print(notes[-1])
    return notes


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=str(date.today()))
    ap.add_argument("--days", type=int, default=8, help="predict games in the next N days")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--absent", default="",
                    help='known absences for each team\'s NEXT game, e.g. "Benfica: Pau Bargallo, Zé Miranda; Porto: Malian"')
    args = ap.parse_args()

    m = pd.read_csv(DATA / "matches.csv")
    m["played"] = m.played.astype(bool)
    for f in FEATURE_FILES:  # per-game features (squad strength, goalkeeper, ...)
        if (DATA / f).exists():
            m = m.merge(pd.read_csv(DATA / f), on="event_id", how="left")
    as_of = pd.Timestamp(args.as_of)
    # Only use results known before the as_of date
    m.loc[pd.to_datetime(m.date) >= as_of, "played"] = False
    season = m[pd.to_datetime(m.date) <= as_of + pd.Timedelta(days=200)].season.max()

    model = TeamModel(Params()).fit(m, as_of=as_of)
    out = DATA / "predictions" / str(as_of.date())
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # ---- upcoming games
    d = pd.to_datetime(m.date)
    up = m[~m.played & (d >= as_of) & (d < as_of + pd.Timedelta(days=args.days))
           & m.competition.isin(LEAGUES) & (m.phase == "regular")]
    up = up.copy()
    notes = apply_absences(up, args.absent)
    preds = model.predict_df(up)
    up = up[["date", "competition", "round"]].join(preds)
    up.to_csv(out / "upcoming.csv", index=False)

    # ---- ratings (teams in this season's leagues)
    cur = m[(m.season == season) & m.competition.isin(LEAGUES)]
    league_of = pd.concat([cur[["home", "competition"]].rename(columns={"home": "team"}),
                           cur[["away", "competition"]].rename(columns={"away": "team"})]).drop_duplicates("team")
    rat = model.ratings().merge(league_of, on="team")
    for comp in LEAGUES:  # express ratings as goals scored/conceded vs an average team of the league
        mask = rat.competition == comp
        lvl = model.level[division(comp)]
        mean_att, mean_def = rat[mask].attack.mean(), rat[mask].defence.mean()
        rat.loc[mask, "goals_for_per_game"] = np.exp(lvl + rat[mask].attack - mean_def)
        rat.loc[mask, "goals_against_per_game"] = np.exp(lvl + mean_att - rat[mask].defence)
    rat.to_csv(out / "ratings.csv", index=False)

    # ---- season simulations + report
    lines = [f"# Previsões Hóquei em Patins — {as_of.date()}", "",
             f"Model fitted on all games before {as_of.date()}, recent games weighted more "
             f"(half-life {model.params.half_life_days:.0f} days). Probabilities are for the "
             f"regulation-time result (no overtime / penalties).", ""]
    if notes:
        lines += ["**Ausências consideradas:** " + "; ".join(notes), ""]
    for comp, label in LEAGUES.items():
        g = cur[(cur.competition == comp) & (cur.phase == "regular")]
        if g.empty:
            continue
        sim = simulate_season(model, g, comp, args.sims, rng)
        sim.to_csv(out / f"season_{comp}.csv", index=False)
        lines += [f"## {label}", ""]
        u = up[up.competition == comp]
        if len(u):
            lines += ["**Próximos jogos**", "", "| Data | Jogo | Casa | Empate | Fora | Golos esperados | Resultado + provável |",
                      "|---|---|---|---|---|---|---|"]
            for r in u.sort_values("date").itertuples():
                lines.append(f"| {r.date} | {name(r.home)} vs {name(r.away)} | {pct(r.p_home)} | {pct(r.p_draw)} | "
                             f"{pct(r.p_away)} | {r.xg_home:.1f} – {r.xg_away:.1f} | {r.likely_score} |")
            lines.append("")
        zcols = [c for c in sim.columns if c.startswith("p_") and c != "p_champion"]
        lines += [f"**Simulação da fase regular** ({args.sims:,} simulações)", "",
                  "| # | Equipa | J | Pts | Pts esperados | 1º lugar | " + " | ".join(c[2:] for c in zcols) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(zcols)]
        for i, r in enumerate(sim.itertuples(), 1):
            z = " | ".join(pct(getattr(r, c)) for c in zcols)
            lines.append(f"| {i} | {name(r.team)} | {r.played} | {r.points_now} | {r.exp_points} | {pct(r.p_champion)} | {z} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
