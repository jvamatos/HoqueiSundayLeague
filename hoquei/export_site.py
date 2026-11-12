"""Export the tables shown on the private dashboard (data/site/*.csv).

    python -m hoquei.export_site --as-of 2026-11-12

Reads the latest prediction run (data/predictions/<as_of>/) and the analysis files.
Percentages are whole numbers (92 = 92%) so the page shows exactly the stored value.
"""
import argparse
from datetime import date

import numpy as np
import pandas as pd

from .common import DATA

LEAGUES = {"1div": "1ª Divisão", "2div_norte": "2ª Divisão Norte", "2div_sul": "2ª Divisão Sul"}
OUT = DATA / "site"


def names():
    return pd.read_csv(DATA / "teams.csv").set_index("team").display.to_dict()


def pct(x):
    return np.round(100 * x).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=str(date.today()))
    as_of = ap.parse_args().as_of
    run = DATA / "predictions" / as_of
    nm = names()
    OUT.mkdir(exist_ok=True)

    # 1. upcoming games
    up = pd.read_csv(run / "upcoming.csv")
    jogos = pd.DataFrame({
        "jogo": [f"{d}_{h}_{a}".replace(" ", "-") for d, h, a in zip(up.date, up.home, up.away)],
        "data": up.date, "liga": up.competition.map(LEAGUES), "jornada": up["round"].astype("Int64"),
        "casa": up.home.map(nm), "fora": up.away.map(nm),
        "p_casa": pct(up.p_home), "p_empate": pct(up.p_draw), "p_fora": pct(up.p_away),
        "golos_casa": up.xg_home.round(1), "golos_fora": up.xg_away.round(1),
        "resultado_provavel": up.likely_score})
    jogos.sort_values(["data", "liga", "casa"]).to_csv(OUT / "jogos.csv", index=False)

    # 2. season simulation
    rows = []
    for comp, label in LEAGUES.items():
        f = run / f"season_{comp}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        top = [c for c in s.columns if c.startswith("p_") and ("top" in c or "playoff" in c)][0]
        bottom = [c for c in s.columns if c.startswith("p_") and "bottom" in c][0]
        for r in s.itertuples():
            rows.append({"id": f"{comp}_{r.team}".replace(" ", "-"), "liga": label, "equipa": nm.get(r.team, r.team),
                         "jogos": r.played, "pontos": r.points_now, "pontos_esperados": round(r.exp_points, 1),
                         "p_campeao": int(round(100 * r.p_champion)), "p_zona_alta": int(round(100 * getattr(r, top))),
                         "p_descida": int(round(100 * getattr(r, bottom))), "posicao_media": r.avg_position})
    pd.DataFrame(rows).to_csv(OUT / "classificacao.csv", index=False)

    # 3. team ratings (goals for/against per game vs an average team of the league)
    rat = pd.read_csv(run / "ratings.csv")
    rat = rat[rat.competition.isin(LEAGUES)].copy()
    rat["rank"] = rat.groupby("competition").overall.rank(ascending=False, method="first").astype(int)
    pd.DataFrame({
        "id": (rat.competition + "_" + rat.team).str.replace(" ", "-"), "liga": rat.competition.map(LEAGUES),
        "posicao": rat["rank"], "equipa": rat.team.map(nm),
        "golos_marcados": rat.goals_for_per_game.round(2), "golos_sofridos": rat.goals_against_per_game.round(2),
        "saldo": (rat.goals_for_per_game - rat.goals_against_per_game).round(2),
    }).sort_values(["liga", "posicao"]).to_csv(OUT / "equipas.csv", index=False)

    # 4. players: best scorers and goalkeepers of teams in this season's leagues
    pr = pd.read_csv(DATA / "player_ratings.csv")
    league_of = dict(zip(rat.team, rat.competition))
    pr = pr[pr.last_team.isin(league_of)].copy()
    pr["liga"] = pr.last_team.map(league_of).map(LEAGUES)
    pr["equipa"] = pr.last_team.map(nm)
    pr["jogador"] = pr.name.fillna("").str.replace("®", "", regex=False).str.strip()
    sk = pr[(pr.games_w >= 8) & pr.gk_goals_prevented_per_game.isna()]
    sk = sk.sort_values("goals_per_game", ascending=False).groupby("liga").head(25)
    pd.DataFrame({"id": sk.player_id, "liga": sk.liga, "jogador": sk.jogador, "equipa": sk.equipa,
                  "golos_por_jogo": sk.goals_per_game.round(2), "jogos_recentes": sk.games_w.round(0).astype(int)}
                 ).to_csv(OUT / "marcadores.csv", index=False)
    gk = pr[(pr.gk_games_w >= 8)].sort_values("gk_goals_prevented_per_game", ascending=False).groupby("liga").head(15)
    pd.DataFrame({"id": gk.player_id, "liga": gk.liga, "jogador": gk.jogador, "equipa": gk.equipa,
                  "golos_evitados_por_jogo": gk.gk_goals_prevented_per_game.round(2),
                  "jogos_recentes": gk.gk_games_w.round(0).astype(int)}).to_csv(OUT / "guarda_redes.csv", index=False)

    # 5. power play per team, last full season and the current one
    t = pd.read_csv(DATA / "discipline_teams.csv")
    seasons = sorted(t.season.unique())[-2:]
    t = t[t.season.isin(seasons) & t.competition.isin(LEAGUES)].copy()
    pd.DataFrame({"id": (t.season + "_" + t.competition + "_" + t.team).str.replace(" ", "-"),
                  "epoca": t.season, "liga": t.competition.map(LEAGUES), "equipa": t.team.map(nm).fillna(t.team),
                  "jogos": t.games, "azuis_por_jogo": t.blue_per_game.round(2), "power_plays": t.pp_opps,
                  "golos_pp": t.pp_goals, "golos_por_pp": t.pp_pct.round(2), "inferioridades": t.pk_opps,
                  "golos_sofridos_por_inferioridade": (1 - t.pk_pct).round(2),
                  "golos_em_inferioridade": t.sh_goals}).to_csv(OUT / "power_play.csv", index=False)

    # 6. last results vs what the model said before the game
    bt = pd.read_csv(DATA / "backtest_predictions.csv")
    bt = bt[pd.to_datetime(bt.date) >= pd.Timestamp(as_of) - pd.Timedelta(days=8)]
    res = np.select([bt.home_goals > bt.away_goals, bt.home_goals == bt.away_goals], ["casa", "empate"], "fora")
    fav = np.array(["casa", "empate", "fora"])[bt[["p_home", "p_draw", "p_away"]].values.argmax(1)]
    pd.DataFrame({"jogo": [f"{d}_{h}_{a}".replace(" ", "-") for d, h, a in zip(bt.date, bt.home, bt.away)],
                  "data": bt.date, "liga": bt.competition.map(LEAGUES), "casa": bt.home.map(nm), "fora": bt.away.map(nm),
                  "resultado": bt.home_goals.astype(int).astype(str) + "-" + bt.away_goals.astype(int).astype(str),
                  "p_casa": pct(bt.p_home), "p_empate": pct(bt.p_draw), "p_fora": pct(bt.p_away),
                  "favorito": fav, "acertou": np.where(fav == res, "sim", "não")}
                 ).sort_values(["data", "liga"]).to_csv(OUT / "ultimos_resultados.csv", index=False)

    # 8. transfers: squad change vs last season, as % change in goals scored / conceded
    tr = pd.read_csv(DATA / "transfers_summary.csv")
    season_now = tr.season.max()
    tr = tr[(tr.season == season_now) & tr.team.isin(league_of)].copy()
    pd.DataFrame({
        "id": tr.team.str.replace(" ", "-"), "liga": tr.team.map(league_of).map(LEAGUES), "equipa": tr.team.map(nm),
        "ataque_pct": np.round(100 * (np.exp(tr.d_att) - 1)).astype(int),
        "defesa_pct": np.round(100 * (np.exp(-tr.d_def) - 1)).astype(int),
        "saldo": np.round(100 * (np.exp(tr.d_att) - 1) - 100 * (np.exp(-tr.d_def) - 1)).astype(int),
        "entradas": tr.arrivals, "saidas": tr.departures,
        "principais_entradas": tr.main_arrivals.fillna(""), "principais_saidas": tr.main_departures.fillna(""),
    }).sort_values("saldo", ascending=False).to_csv(OUT / "mercado.csv", index=False)

    # 9. player impact: % change in the team's goals scored / conceded per full game played
    im = pd.read_csv(DATA / "player_impact.csv")
    im = im[im.team.isin(league_of) & (im.games >= 15)].copy()
    im["liga"] = im.team.map(league_of).map(LEAGUES)
    im = im.sort_values("impact", ascending=False).groupby("liga").head(25)
    pd.DataFrame({
        "id": im.player_id, "liga": im.liga, "jogador": im.name.str.replace("®", "", regex=False).str.strip(),
        "equipa": im.team.map(nm), "posicao": np.where(im.goalkeeper, "GR", "Campo"),
        "ataque_pct": np.round(100 * (np.exp(im.attack) - 1)).astype(int),
        "defesa_pct": np.round(100 * (np.exp(-im.defence) - 1)).astype(int),
        "impacto": np.round(100 * im.impact).astype(int), "jogos": im.games.astype(int),
    }).to_csv(OUT / "impacto.csv", index=False)

    # 7. model accuracy (walk-forward backtest, test period 2023/24 -> Nov 2026)
    pd.DataFrame([
        {"modelo": "Frequências base (sempre o mesmo palpite)", "acerto": 51.7, "log_loss": 0.977},
        {"modelo": "Elo", "acerto": 64.5, "log_loss": 0.851},
        {"modelo": "Este modelo", "acerto": 65.9, "log_loss": 0.814},
    ]).to_csv(OUT / "precisao.csv", index=False)
    print(f"exported to {OUT}: " + ", ".join(sorted(p.name for p in OUT.glob("*.csv"))))


if __name__ == "__main__":
    main()
