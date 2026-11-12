"""Deeper team statistics for the dashboard ("Estatísticas" tab).

League games (regular season + play-offs) of this season's teams, for two periods:
  * "2025/26"        the last full season
  * "desde 2023/24"  2023/24 to now (play-by-play and the model's backtest cover it fully)

Outputs (data/site/):
  estatisticas.csv  one row per team and period, many metrics, each with its league rank
  tempos.csv        goals scored / conceded per 5-minute slice, per team and period
"""
import numpy as np
import pandas as pd

from .common import DATA
from .model import regulation_scores

LEAGUES = {"1div": "1ª Divisão", "2div_norte": "2ª Divisão Norte", "2div_sul": "2ª Divisão Sul"}
PERIODS = {"2025/26": ("2025-08-01", "2026-08-01"), "desde 2023/24": ("2023-08-01", "2100-01-01")}
SEASON = "2026-27"
# metric -> True if higher is better (for ranks)
BETTER = {
    "pontos_por_jogo": True, "golos_marcados_jogo": True, "golos_sofridos_jogo": False,
    "pontos_vs_esperado": True, "vitorias_marca_primeiro": True, "pontos_sofre_primeiro": True,
    "vitorias_ganha_intervalo": True, "nao_perde_perde_intervalo": True, "pontos_jogos_1_golo": True,
    "saldo_1a_parte": True, "saldo_2a_parte": True,
    "pct_bola_corrida": None, "pct_power_play": None, "pct_inferioridade": None, "pct_penalti": None,
    "pct_livre_direto": None, "pct_recarga": None,
    "penaltis_convertidos": True, "livres_convertidos": True,
    "penaltis_sofridos_convertidos": False, "livres_sofridos_convertidos": False,
    "pct_melhor_marcador": None, "marcadores_diferentes": None,
}


def goal_type(icon):
    if "under_play" in icon:
        return "inferioridade"
    if "power_play" in icon:
        return "power_play"
    if "penalti" in icon:
        return "penalti"
    if "livre" in icon:
        return "livre_direto"
    if "recarga" in icon:
        return "recarga"
    return "bola_corrida"


def main():
    names = pd.read_csv(DATA / "teams.csv").set_index("team").display.to_dict()
    am = pd.read_csv(DATA / "api_matches.csv")
    am = am[am.played.astype(bool) & am.competition.isin(list(LEAGUES) + ["2div"])].copy()
    am["hg"], am["ag"] = regulation_scores(am)
    am["date"] = pd.to_datetime(am.date)
    cur = am[am.season == SEASON]
    allm = pd.read_csv(DATA / "matches.csv")
    cur_all = allm[(allm.season == SEASON) & allm.competition.isin(LEAGUES)]
    league_of = dict(zip(cur_all.home, cur_all.competition)) | dict(zip(cur_all.away, cur_all.competition))

    # team-game rows
    tg = pd.concat([
        pd.DataFrame({"event_id": am.event_id, "date": am.date, "comp": am.competition, "team": am.home, "opp": am.away, "side": "home",
                      "gf": am.hg, "ga": am.ag, "ht_f": am.home_ht, "ht_a": am.away_ht}),
        pd.DataFrame({"event_id": am.event_id, "date": am.date, "comp": am.competition, "team": am.away, "opp": am.home, "side": "away",
                      "gf": am.ag, "ga": am.hg, "ht_f": am.away_ht, "ht_a": am.home_ht})], ignore_index=True)
    tg["pts"] = np.select([tg.gf > tg.ga, tg.gf == tg.ga], [3, 1], 0)

    # expected points from the walk-forward backtest (predictions made before each game)
    bt = pd.read_csv(DATA / "backtest_predictions.csv")
    bt["date"] = pd.to_datetime(bt.date)
    xp = pd.concat([
        pd.DataFrame({"date": bt.date, "team": bt.home, "xpts": 3 * bt.p_home + bt.p_draw}),
        pd.DataFrame({"date": bt.date, "team": bt.away, "xpts": 3 * bt.p_away + bt.p_draw})])
    tg = tg.merge(xp.drop_duplicates(["date", "team"]), on=["date", "team"], how="left")

    # play-by-play goals: first goal, types, timing
    ev = pd.read_csv(DATA / "events.csv")
    goals = ev[ev.icon.str.startswith("golo") & ev.side.isin(["home", "away"])].copy()
    goals["order"] = range(len(goals))
    goals = goals.sort_values(["event_id", "period", "minute", "order"])
    goals["type"] = goals.icon.map(goal_type)
    first = goals.groupby("event_id").side.first()
    tg["scored_first"] = tg.event_id.map(first)
    tg["scored_first"] = np.where(tg.scored_first.isna(), np.nan, (tg.scored_first == tg.side).astype(float))
    gk = goals.merge(am[["event_id", "home", "away"]], on="event_id")
    gk["team"] = gk.home.where(gk.side == "home", gk.away)
    gk["opp"] = gk.away.where(gk.side == "home", gk.home)
    gk = gk.merge(am[["event_id", "date"]], on="event_id")
    per = pd.to_numeric(gk.period, errors="coerce")
    gk["slice"] = np.where(per.isin([1, 2]), (per - 1) * 25 + (gk.minute.clip(upper=24) // 5) * 5, np.nan)

    # set pieces from player stats (attempts = scored + missed, saved included in missed)
    ps = pd.read_csv(DATA / "api_player_stats.csv")
    for c in ["golomarcadodepenalty", "penaltyfalhado", "golomarcadodelivredireto", "livrediretofalhado", "golosmarcadostotal"]:
        ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0)
    ps = ps.merge(am[["event_id", "date", "home_id", "home", "away"]], on="event_id")
    ps["team"] = ps.home.where(ps.team_id == ps.home_id, ps.away)
    ps["opp"] = ps.away.where(ps.team_id == ps.home_id, ps.home)

    rows, slices = [], []
    for pname, (a, b) in PERIODS.items():
        t = tg[(tg.date >= a) & (tg.date < b)]
        g = gk[(gk.date >= a) & (gk.date < b)]
        p = ps[(ps.date >= a) & (ps.date < b)]
        for team, comp in league_of.items():
            x = t[t.team == team]
            if len(x) < 5:
                continue
            r = {"id": f"{pname}_{team}".replace(" ", "-").replace("/", "-"), "periodo": pname,
                 "liga": LEAGUES[comp], "equipa": names.get(team, team), "jogos": len(x),
                 # league where most of these games were played (ranks are within it)
                 "liga_periodo": LEAGUES.get(x.comp.mode().iloc[0], "2ª Divisão")}
            r["pontos_por_jogo"] = x.pts.mean()
            r["golos_marcados_jogo"] = x.gf.mean()
            r["golos_sofridos_jogo"] = x.ga.mean()
            xx = x[x.xpts.notna()]
            r["pontos_vs_esperado"] = (xx.pts - xx.xpts).sum() if len(xx) >= 5 else np.nan
            sf = x[x.scored_first == 1]; sa = x[x.scored_first == 0]
            r["vitorias_marca_primeiro"] = (sf.gf > sf.ga).mean() if len(sf) >= 3 else np.nan
            r["pontos_sofre_primeiro"] = sa.pts.mean() if len(sa) >= 3 else np.nan
            ht = x[x.ht_f.notna()]
            lead = ht[ht.ht_f > ht.ht_a]; trail = ht[ht.ht_f < ht.ht_a]
            r["vitorias_ganha_intervalo"] = (lead.gf > lead.ga).mean() if len(lead) >= 3 else np.nan
            r["nao_perde_perde_intervalo"] = (trail.gf >= trail.ga).mean() if len(trail) >= 3 else np.nan
            one = x[(x.gf - x.ga).abs() <= 1]
            r["jogos_1_golo"] = len(one)
            r["registo_1_golo"] = f"{(one.gf > one.ga).sum()}V {(one.gf == one.ga).sum()}E {(one.gf < one.ga).sum()}D"
            r["pontos_jogos_1_golo"] = one.pts.mean() if len(one) >= 3 else np.nan
            r["saldo_1a_parte"] = (ht.ht_f - ht.ht_a).mean() if len(ht) else np.nan
            r["saldo_2a_parte"] = ((ht.gf - ht.ht_f) - (ht.ga - ht.ht_a)).mean() if len(ht) else np.nan
            mine = g[g.team == team]
            for typ in ["bola_corrida", "power_play", "inferioridade", "penalti", "livre_direto", "recarga"]:
                r[f"pct_{typ}"] = (mine.type == typ).mean() * 100 if len(mine) >= 20 else np.nan
            mp = p[p.team == team]; op = p[p.opp == team]
            att = mp.golomarcadodepenalty.sum() + mp.penaltyfalhado.sum()
            r["penaltis"] = f"{int(mp.golomarcadodepenalty.sum())}/{int(att)}"
            r["penaltis_convertidos"] = mp.golomarcadodepenalty.sum() / att * 100 if att >= 5 else np.nan
            att = mp.golomarcadodelivredireto.sum() + mp.livrediretofalhado.sum()
            r["livres"] = f"{int(mp.golomarcadodelivredireto.sum())}/{int(att)}"
            r["livres_convertidos"] = mp.golomarcadodelivredireto.sum() / att * 100 if att >= 5 else np.nan
            att = op.golomarcadodepenalty.sum() + op.penaltyfalhado.sum()
            r["penaltis_sofridos_convertidos"] = op.golomarcadodepenalty.sum() / att * 100 if att >= 5 else np.nan
            att = op.golomarcadodelivredireto.sum() + op.livrediretofalhado.sum()
            r["livres_sofridos_convertidos"] = op.golomarcadodelivredireto.sum() / att * 100 if att >= 5 else np.nan
            sc = mp.groupby("player_id").golosmarcadostotal.sum()
            sc = sc[sc > 0]
            r["pct_melhor_marcador"] = sc.max() / sc.sum() * 100 if sc.sum() >= 10 else np.nan
            r["marcadores_diferentes"] = int((sc > 0).sum())
            rows.append(r)
            gf = g[g.team == team].slice.value_counts(); ga = g[g.opp == team].slice.value_counts()
            for s in range(0, 50, 5):
                slices.append({"id": f"{r['id']}_{s}", "periodo": pname, "equipa": r["equipa"], "liga": r["liga"],
                               "minuto": f"{s}-{s + 5}'", "ordem": s // 5,
                               "golos_marcados": int(gf.get(s, 0)), "golos_sofridos": int(ga.get(s, 0))})
    out = pd.DataFrame(rows)
    for c in ["vitorias_marca_primeiro", "vitorias_ganha_intervalo", "nao_perde_perde_intervalo"]:
        out[c] = out[c] * 100  # stored as percentages, as shown on the page
    # league ranks within period
    for m, hi in BETTER.items():
        if hi is None:
            continue
        out[f"rank_{m}"] = out.groupby(["periodo", "liga_periodo"])[m].rank(ascending=not hi, method="min")
    out["equipas_liga"] = out.groupby(["periodo", "liga_periodo"]).equipa.transform("count")
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(2)
    pct_cols = [c for c in out if c.startswith(("pct_", "vitorias_", "nao_perde_")) or c.endswith("_convertidos")]
    out[pct_cols] = out[pct_cols].round(0)  # whole percentages, as shown
    (DATA / "site").mkdir(exist_ok=True)
    out.to_csv(DATA / "site" / "estatisticas.csv", index=False)
    pd.DataFrame(slices).to_csv(DATA / "site" / "tempos.csv", index=False)
    print(f"{len(out)} team-period rows")
    return out


if __name__ == "__main__":
    o = main()
    cols = ["equipa", "jogos", "pontos_por_jogo", "pontos_vs_esperado", "vitorias_marca_primeiro", "pontos_sofre_primeiro",
            "vitorias_ganha_intervalo", "registo_1_golo", "pct_power_play", "penaltis", "livres", "pct_melhor_marcador"]
    print(o[(o.periodo == "2025/26") & (o.liga == "1ª Divisão")][cols].to_string())
