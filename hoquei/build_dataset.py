"""Merge archive (2007/08-2020/21) and API (2021/22-) games into data/matches.csv."""
import html

import pandas as pd

from .common import DATA, team_key

COLS = ["season", "competition", "phase", "round", "date", "home", "away",
        "home_goals", "away_goals", "home_ot", "away_ot", "home_pen", "away_pen",
        "played", "source", "event_id", "round_label"]


def load():
    o = pd.read_csv(DATA / "archive_old_matches.csv")  # 2007/08 - 2014/15
    a = pd.read_csv(DATA / "archive_matches.csv")      # 2015/16 - 2020/21
    b = pd.read_csv(DATA / "api_matches.csv")          # 2021/22 -
    m = pd.concat([o, a, b], ignore_index=True)
    m["neutral"] = False
    if (DATA / "extra_matches.csv").exists():  # cups, Europe, foreign leagues (2021/22 on)
        x = pd.read_csv(DATA / "extra_matches.csv")
        known = set(m.home) | set(m.away)
        # Taça games against lower-division teams the model has never seen are left out
        x = x[(x.competition != "taca") | (x.home.isin(known) & x.away.isin(known))]
        m = pd.concat([m, x], ignore_index=True)
    # re-apply the alias table so name fixes don't need a re-scrape
    m["home"] = m.home.map(team_key)
    m["away"] = m.away.map(team_key)
    for c in COLS:
        if c not in m:
            m[c] = None
    m = m[COLS + ["home_name", "away_name", "neutral"]].sort_values(["date", "competition", "home"]).reset_index(drop=True)
    m["played"] = m.played.astype(bool) & m.home_goals.notna() & m.away_goals.notna()
    m["neutral"] = m.neutral.fillna(False).astype(bool)
    add_rest_days(m)
    return m


def add_rest_days(m):
    """Days since each team's previous game in any competition (capped at 14)."""
    d = pd.to_datetime(m.date)
    long = pd.concat([pd.DataFrame({"i": m.index, "team": m.home, "d": d, "side": "h"}),
                      pd.DataFrame({"i": m.index, "team": m.away, "d": d, "side": "a"})]).sort_values(["team", "d"])
    long["rest"] = long.groupby("team").d.diff().dt.days.clip(upper=14).fillna(14)
    for side in "ha":
        r = long[long.side == side].set_index("i").rest
        m[f"rest_{side}"] = r.reindex(m.index).values
        m[f"tired_{side}"] = (m[f"rest_{side}"] <= 3).astype(float)


# Display fixes for names the API spells without accents/apostrophes
DISPLAY_FIX = {"riba dave": "Riba d'Ave", "candelaria": "Candelária", "povoa": "Póvoa",
               "grandola": "Grândola", "academica": "Académica", "valenca": "Valença",
               "fisica": "Física", "paco arcos": "Paço de Arcos"}


def team_names(m):
    """Latest name each team was listed with (API names are short: 'Sporting', 'Valongo')."""
    rows = pd.concat([m[["date", "home", "home_name"]].set_axis(["date", "team", "display"], axis=1),
                      m[["date", "away", "away_name"]].set_axis(["date", "team", "display"], axis=1)])
    rows = rows.dropna().sort_values("date")
    rows["display"] = rows.display.map(html.unescape)
    out = rows.drop_duplicates("team", keep="last")[["team", "display"]].sort_values("team")
    out["display"] = [DISPLAY_FIX.get(t, d) for t, d in zip(out.team, out.display)]
    return out


def main():
    m = load()
    team_names(m).to_csv(DATA / "teams.csv", index=False)
    m.to_csv(DATA / "matches.csv", index=False)
    p = m[m.played]
    print(f"{len(m)} games, {len(p)} played, {p.date.min()} -> {p.date.max()}")
    print(p.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
