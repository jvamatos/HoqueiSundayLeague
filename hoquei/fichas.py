"""Game cards ("fichas de jogo") for the 2ª Divisão Sul: context for the group's own picks.

For each upcoming game (next N days): table position, last 5 results, the home team's home
record and the away team's away record (since 2025/26), each team's "home factor" over the
last three seasons (home goal difference minus away goal difference, per game), the last 5
head-to-head games, top scorers this season and last, the main goalkeeper's saves, known
absences (data/absences.csv) and, last, the model's probabilities.

    python -m hoquei.fichas --as-of 2026-11-12 --days 10   -> data/site/fichas.csv
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd

from .common import DATA
from .model import regulation_scores

COMP = "2div_sul"
SEASON = "2026-27"


def res_letter(gf, ga):
    return "V" if gf > ga else "E" if gf == ga else "D"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=str(date.today()))
    ap.add_argument("--days", type=int, default=10)
    args = ap.parse_args()
    as_of = pd.Timestamp(args.as_of)
    names = pd.read_csv(DATA / "teams.csv").set_index("team").display.to_dict()
    nm = lambda t: names.get(t, str(t))

    m = pd.read_csv(DATA / "matches.csv")
    m["date"] = pd.to_datetime(m.date)
    m["played"] = m.played.astype(bool) & (m.date < as_of)
    hg, ag = regulation_scores(m)
    m["hg"], m["ag"] = m.home_goals, m.away_goals  # final scores for records / form
    lg = m[m.competition.isin(["1div", "2div_norte", "2div_sul", "2div"])]

    # current table
    cur = lg[(lg.season == SEASON) & (lg.competition == COMP)]
    pl = cur[cur.played]
    table = {}
    for r in pl.itertuples():
        for t, gf, ga in ((r.home, r.hg, r.ag), (r.away, r.ag, r.hg)):
            x = table.setdefault(t, [0, 0, 0, 0])  # pts, games, gf, ga
            x[0] += 3 if gf > ga else 1 if gf == ga else 0; x[1] += 1; x[2] += gf; x[3] += ga
    for t in set(cur.home) | set(cur.away):
        table.setdefault(t, [0, 0, 0, 0])
    order = sorted(table, key=lambda t: (-table[t][0], -(table[t][2] - table[t][3]), -table[t][2]))
    pos = {t: i + 1 for i, t in enumerate(order)}

    played = lg[lg.played].sort_values("date")
    def form(t):
        g = played[(played.home == t) | (played.away == t)].tail(5)
        out = []
        for r in g.itertuples():
            home = r.home == t
            gf, ga = (r.hg, r.ag) if home else (r.ag, r.hg)
            out.append(f"{res_letter(gf, ga)} {int(gf)}-{int(ga)} {'vs' if home else '@'} {nm(r.away if home else r.home)}")
        return " · ".join(reversed(out))

    def record(t, home):
        g = played[(played.date >= "2025-08-01") & ((played.home == t) if home else (played.away == t))]
        gf = g.hg if home else g.ag; ga = g.ag if home else g.hg
        w = (gf > ga).sum(); d = (gf == ga).sum(); l = (gf < ga).sum()
        return f"{w}V {d}E {l}D · {gf.mean():.1f}-{ga.mean():.1f} golos/jogo".replace(".", ",") if len(g) else "—"

    def home_factor(t):
        g = played[played.date >= as_of - pd.Timedelta(days=3 * 365)]
        h = g[g.home == t]; a = g[g.away == t]
        if len(h) < 8 or len(a) < 8:
            return None
        return round(float((h.hg - h.ag).mean() - (a.ag - a.hg).mean()), 2)

    allm = m[m.played].sort_values("date")
    def h2h(a, b):
        g = allm[((allm.home == a) & (allm.away == b)) | ((allm.home == b) & (allm.away == a))].tail(5)
        return " · ".join(f"{r.date:%d/%m/%y} {nm(r.home)} {int(r.hg)}-{int(r.ag)} {nm(r.away)}"
                          for r in g.iloc[::-1].itertuples()) or "Sem jogos entre as duas equipas nos nossos dados."

    ps = pd.read_csv(DATA / "site" / "jogadores_stats.csv")
    gks = pd.read_csv(DATA / "site" / "gr_stats.csv")
    def scorers(t):
        x = ps[(ps.periodo == "2026/27") & (ps.equipa == nm(t))].sort_values("golos", ascending=False).head(3)
        y = ps[(ps.periodo == "2025/26") & (ps.equipa == nm(t))].sort_values("golos", ascending=False).head(3)
        now = ", ".join(f"{r.jogador} {int(r.golos)}" for r in x.itertuples() if r.golos > 0) or "—"
        last = ", ".join(f"{r.jogador} {int(r.golos)}" for r in y.itertuples() if r.golos > 0) or "—"
        return now, last
    def keeper(t):
        x = gks[(gks.periodo == "desde 2023/24") & (gks.equipa == nm(t))].sort_values("jogos", ascending=False).head(1)
        if x.empty:
            return "—"
        r = x.iloc[0]
        return (f"{r.jogador}: {str(r.sofridos_por_jogo).replace('.', ',')} golos sofridos/jogo, penáltis "
                f"{r.penaltis_defendidos} e livres {r.livres_defendidos} defendidos (desde 2023/24)")

    absn = pd.read_csv(DATA / "absences.csv") if (DATA / "absences.csv").exists() else pd.DataFrame(columns=["team", "player", "reason"])
    def absences(t):
        x = absn[absn.team == t]
        return "; ".join(f"{r.player} ({r.reason})" for r in x.itertuples()) or "Nenhuma conhecida"

    up = pd.read_csv(sorted((DATA / "predictions").glob("*/upcoming.csv"))[-1])
    up["date"] = pd.to_datetime(up.date)
    games = cur[(~cur.played) & (cur.date >= as_of) & (cur.date < as_of + pd.Timedelta(days=args.days))].sort_values("date")
    rows = []
    for r in games.itertuples():
        pr = up[(up.home == r.home) & (up.away == r.away)]
        p = pr.iloc[0] if len(pr) else None
        s_h, s_h_last = scorers(r.home); s_a, s_a_last = scorers(r.away)
        rows.append({
            "jogo": f"{r.date:%Y-%m-%d}_{r.home}_{r.away}".replace(" ", "-"), "data": f"{r.date:%Y-%m-%d}",
            "jornada": int(r.round) if pd.notna(r.round) else None, "casa": nm(r.home), "fora": nm(r.away),
            "pos_casa": pos.get(r.home), "pts_casa": table[r.home][0], "j_casa": table[r.home][1],
            "pos_fora": pos.get(r.away), "pts_fora": table[r.away][0], "j_fora": table[r.away][1],
            "forma_casa": form(r.home), "forma_fora": form(r.away),
            "casa_em_casa": record(r.home, True), "fora_fora": record(r.away, False),
            "fator_casa_casa": home_factor(r.home), "fator_casa_fora": home_factor(r.away),
            "h2h": h2h(r.home, r.away),
            "marcadores_casa": s_h, "marcadores_casa_ant": s_h_last,
            "marcadores_fora": s_a, "marcadores_fora_ant": s_a_last,
            "gr_casa": keeper(r.home), "gr_fora": keeper(r.away),
            "ausencias_casa": absences(r.home), "ausencias_fora": absences(r.away),
            "p_casa": int(round(100 * p.p_home)) if p is not None else None,
            "p_empate": int(round(100 * p.p_draw)) if p is not None else None,
            "p_fora": int(round(100 * p.p_away)) if p is not None else None,
            "golos_esp_casa": round(p.xg_home, 1) if p is not None else None,
            "golos_esp_fora": round(p.xg_away, 1) if p is not None else None,
        })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "site" / "fichas.csv", index=False)
    print(f"{len(out)} game cards")
    return out


if __name__ == "__main__":
    o = main()
    if len(o):
        print(o.iloc[0].to_string())
