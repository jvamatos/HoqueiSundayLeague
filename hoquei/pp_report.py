"""Write data/analysis/power_play.md: power-play, penalty-kill, cards and referee tables."""
import pandas as pd

from .common import DATA

LEAGUES = {"1div": "1ª Divisão", "2div_norte": "2ª Divisão Norte", "2div_sul": "2ª Divisão Sul"}


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    names = pd.read_csv(DATA / "teams.csv").set_index("team").display.to_dict()
    league = pd.read_csv(DATA / "discipline_league.csv")
    team = pd.read_csv(DATA / "discipline_teams.csv")
    refs = pd.read_csv(DATA / "referees.csv")
    gd = pd.read_csv(DATA / "game_discipline.csv")
    out = DATA / "analysis"
    out.mkdir(exist_ok=True)

    L = ["# Power play, cards and referees", "",
         "Source: hoqueipatins.pt play-by-play (2021/22 onwards). A power play (PP) starts with a "
         "blue card (2 min) or red card (4 min). From 2025/26 the site marks PP start/end; before "
         "that, PP windows are inferred from card times. Goals within the window, including the "
         "direct free hit/penalty that often comes with a blue card, count as PP goals. Times are "
         "to the minute, so a few goals in the same minute as a card may be misassigned. A team can "
         "score more than once in one power play, so 'goals per PP' can exceed 1. Team tables "
         "include play-off games.", "",
         "## League averages", "",
         "| Season | League | PP per team-game | goals per PP | PP goal rate vs. even strength | short-handed goals per 100 PP |",
         "|---|---|---|---|---|---|"]
    for r in league[league.competition.isin(LEAGUES) & (league.games >= 100)].itertuples():
        L.append(f"| {r.season} | {LEAGUES[r.competition]} | {r.pp_per_team_game:.2f} | {r.pp_conversion:.2f} | "
                 f"{r.pp_goal_rate_x:.1f}x | {100 * r.sh_goals / r.pp_opps:.0f} |")

    for season in ["2025-26"]:
        for comp, label in LEAGUES.items():
            t = team[(team.season == season) & (team.competition == comp)].copy()
            if t.empty:
                continue
            t["name"] = t.team.map(names).fillna(t.team)
            L += ["", f"## {label} {season}: teams", "",
                  "| Team | Blue cards/game | PP | PP goals | goals per PP | PK | goals conceded per PK | Short-handed goals for / against |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in t.sort_values("pp_pct", ascending=False).itertuples():
                L.append(f"| {r.name} | {r.blue_per_game:.2f} | {r.pp_opps} | {r.pp_goals} | {r.pp_pct:.2f} | "
                         f"{r.pk_opps} | {1 - r.pk_pct:.2f} | {r.sh_goals} / {r.sh_goals_against} |")

    # referees: last two seasons, 1st division, at least 15 games
    r = refs[refs.role == "Principal"].merge(gd, on="event_id")
    r = r[(r.season >= "2024-25") & (r.competition == "1div")]
    r["blue"] = r.blue_h + r.blue_a
    r["goals"] = r.home_goals + r.away_goals
    avg_b, avg_g = r.drop_duplicates("event_id").blue.mean(), r.drop_duplicates("event_id").goals.mean()
    rt = r.groupby("name").agg(games=("event_id", "nunique"), blue=("blue", "mean"), goals=("goals", "mean"))
    rt = rt[rt.games >= 15].sort_values("blue", ascending=False)
    L += ["", "## Referees, 1ª Divisão 2024/25–now (≥15 games)", "",
          f"League average: {avg_b:.2f} blue cards and {avg_g:.1f} goals per game.", "",
          "| Referee | Games | Blue cards/game | Goals/game |", "|---|---|---|---|"]
    for x in rt.itertuples():
        L.append(f"| {x.Index} | {x.games} | {x.blue:.2f} | {x.goals:.1f} |")
    L += ["", "Note: referees work in pairs, so each game counts for both referees. In the model, "
          "a stricter referee pair means more goals for *both* teams (about +0.35 × log(strictness)); "
          "it does not change who is favoured."]
    (out / "power_play.md").write_text("\n".join(L))
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
