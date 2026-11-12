"""Fun facts per team, computed from our own data (so each one can be checked).

Facts for every team in this season's three leagues:
  * seasons in the 1ª Divisão since 2007/08 (results archive)
  * biggest win in our data (all competitions since 2007/08 domestic, 2021/22 cups/Europe)
  * longest unbeaten run in the league
  * top scorer for the club since 2021/22, and the most goals by one player in a game
  * goalkeeper with most saved penalties + direct free hits since 2021/22
Plus, when the team is top or bottom of its league since 2022/23 (play-by-play):
  * comebacks: wins after being 2+ goals down
  * share of its goals scored in the last 5 minutes
  * home win rate, blue cards per game

Output: data/site/factos.csv (id, liga, equipa, ordem, categoria, facto)
"""
import numpy as np
import pandas as pd

from .common import DATA
from .model import regulation_scores

LEAGUES = {"1div": "1ª Divisão", "2div_norte": "2ª Divisão Norte", "2div_sul": "2ª Divisão Sul"}
SEASON = "2026-27"


def fmt_date(d):
    return pd.Timestamp(d).strftime("%d/%m/%Y")


def season_label(s):
    return f"{s[:4]}/{s[5:]}"


def main():
    names = pd.read_csv(DATA / "teams.csv").set_index("team").display.to_dict()
    nm = lambda t: names.get(t, str(t).title())
    m = pd.read_csv(DATA / "matches.csv")
    m["played"] = m.played.astype(bool)
    played = m[m.played].copy()
    played["date"] = pd.to_datetime(played.date)
    played = played[played.date.notna()]
    cur = m[(m.season == SEASON) & m.competition.isin(LEAGUES)]
    league_of = dict(zip(cur.home, cur.competition)) | dict(zip(cur.away, cur.competition))

    long = pd.concat([
        pd.DataFrame({"team": played.home, "opp": played.away, "gf": played.home_goals, "ga": played.away_goals,
                      "date": played.date, "season": played.season, "comp": played.competition, "home": True,
                      "event_id": played.event_id}),
        pd.DataFrame({"team": played.away, "opp": played.home, "gf": played.away_goals, "ga": played.home_goals,
                      "date": played.date, "season": played.season, "comp": played.competition, "home": False,
                      "event_id": played.event_id})])
    long["margin"] = long.gf - long.ga

    # ---- player data (2021/22 on)
    ps = pd.read_csv(DATA / "api_player_stats.csv")
    am = pd.read_csv(DATA / "api_matches.csv")
    ps = ps.merge(am[["event_id", "date", "home_id", "home", "away", "home_name", "away_name", "competition"]], on="event_id")
    ps["team"] = ps.home.where(ps.team_id == ps.home_id, ps.away)
    ps["opp"] = ps.away.where(ps.team_id == ps.home_id, ps.home)
    for c in ["golosmarcadostotal", "penaltydefendido", "livrediretodefendido"]:
        ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0)
    pn = pd.read_csv(DATA / "players.csv").set_index("player_id").name.str.replace("®", "", regex=False).str.strip()

    # ---- play-by-play (2022/23 on): comebacks and late goals
    ev = pd.read_csv(DATA / "events.csv")
    goals = ev[ev.icon.str.startswith("golo") & ev.side.isin(["home", "away"])].copy()
    goals["order"] = range(len(goals))
    goals = goals.sort_values(["event_id", "period", "minute", "order"])
    comeback_rows, late_rows = [], []
    eid2 = am.set_index("event_id")
    for eid, g in goals.groupby("event_id"):
        if eid not in eid2.index:
            continue
        r = eid2.loc[eid]
        h = a = 0; worst = {"home": 0, "away": 0}
        for side, per, mi in zip(g.side, g.period, g.minute):
            h += side == "home"; a += side == "away"
            worst["home"] = min(worst["home"], h - a); worst["away"] = min(worst["away"], a - h)
            if str(per) in ("2", "2.0") and pd.notna(mi) and mi >= 20:
                late_rows.append(dict(team=r.home if side == "home" else r.away, season=r.season))
        if pd.notna(r.home_goals):
            for side, team, opp, gf, ga in (("home", r.home, r.away, r.home_goals, r.away_goals),
                                            ("away", r.away, r.home, r.away_goals, r.home_goals)):
                if gf > ga and worst[side] <= -2:
                    comeback_rows.append(dict(team=team, opp=opp, deficit=-worst[side], date=r.date,
                                              score=f"{int(gf)}-{int(ga)}", season=r.season))
    comebacks = pd.DataFrame(comeback_rows)
    late = pd.DataFrame(late_rows)
    all_goals = goals.merge(am[["event_id", "home", "away", "season"]], on="event_id")
    all_goals["team"] = all_goals.home.where(all_goals.side == "home", all_goals.away)

    rows = []
    for team, comp in sorted(league_of.items()):
        facts = []
        tl = long[long.team == team]
        dom = tl[tl.comp.isin(["1div", "2div_norte", "2div_sul", "2div"])]

        # seasons in the 1st division
        s1 = sorted(dom[dom.comp == "1div"].season.unique())
        if s1:
            facts.append(("História", f"{len(s1)} épocas na 1ª Divisão desde 2007/08 (a primeira nos nossos dados: {season_label(s1[0])})."))
        else:
            facts.append(("História", "Nunca jogou na 1ª Divisão desde 2007/08."))

        # biggest win
        if len(tl):
            b = tl.sort_values(["margin", "gf", "date"], ascending=[False, False, True]).iloc[0]
            if b["margin"] > 0:
                facts.append(("Recorde", f"Maior vitória: {int(b.gf)}-{int(b.ga)} {'em casa' if b.home else 'fora'} contra o "
                                         f"{nm(b.opp)}, a {fmt_date(b.date)}."))

        # longest unbeaten run in the league
        if len(dom):
            res = dom.sort_values("date").margin.values >= 0
            best = run = 0; end_i = 0
            for i, ok in enumerate(res):
                run = run + 1 if ok else 0
                if run > best:
                    best, end_i = run, i
            if best >= 5:
                end = dom.sort_values("date").iloc[end_i]
                facts.append(("Recorde", f"Maior série sem perder no campeonato: {best} jogos (até {fmt_date(end.date)})."))

        # players
        tp = ps[ps.team == team]
        if len(tp):
            top = tp.groupby("player_id").golosmarcadostotal.sum().sort_values(ascending=False)
            if len(top) and top.iloc[0] > 0:
                facts.append(("Jogadores", f"Melhor marcador do clube desde 2021/22: {pn.get(top.index[0], '?')}, com "
                                           f"{int(top.iloc[0])} golos em todas as competições."))
            mx = tp.sort_values("golosmarcadostotal", ascending=False).iloc[0]
            if mx.golosmarcadostotal >= 4:
                facts.append(("Jogadores", f"Mais golos num só jogo: {int(mx.golosmarcadostotal)}, por "
                                           f"{pn.get(mx.player_id, '?')} contra o {nm(mx.opp)} ({fmt_date(mx.date)})."))
            gk = tp[tp.goalkeeper.astype(bool)].groupby("player_id")[["penaltydefendido", "livrediretodefendido"]].sum()
            gk["tot"] = gk.sum(axis=1)
            gk = gk.sort_values("tot", ascending=False)
            if len(gk) and gk.tot.iloc[0] >= 5:
                x = gk.iloc[0]
                facts.append(("Guarda-redes", f"{pn.get(gk.index[0], '?')} defendeu {int(x.penaltydefendido)} penáltis e "
                                              f"{int(x.livrediretodefendido)} livres diretos desde 2021/22."))

        # league-relative facts (2022/23 on, this team's league)
        peers = [t for t, c in league_of.items() if c == comp]
        def rank(series, t, high=True):
            s = series.reindex(peers).dropna().sort_values(ascending=not high)
            return (list(s.index).index(t) + 1, len(s), s.get(t)) if t in s.index else (None, len(s), None)

        if len(comebacks):
            cb = comebacks.groupby("team").size()
            pos, n, v = rank(cb.reindex(peers).fillna(0), team)
            if v and v >= 3 and pos and pos <= 2:
                best_cb = comebacks[comebacks.team == team].sort_values("deficit", ascending=False).iloc[0]
                facts.append(("Estilo", f"Equipa das reviravoltas: {int(v)} vitórias depois de estar a perder por 2 ou mais "
                                        f"golos desde 2022/23 ({pos}º na liga). A maior: recuperou de {int(best_cb.deficit)} golos "
                                        f"de desvantagem e ganhou {best_cb.score} ao {nm(best_cb.opp)}."))
        if len(late):
            share = late.groupby("team").size() / all_goals.groupby("team").size()
            pos, n, v = rank(share, team)
            if v is not None and pos and (pos <= 2):
                facts.append(("Estilo", f"Gosta de decidir no fim: {round(100 * v)}% dos seus golos (desde 2022/23) são marcados "
                                        f"nos últimos 5 minutos, a maior percentagem da liga."
                              if pos == 1 else f"{round(100 * v)}% dos seus golos (desde 2022/23) chegam nos últimos 5 minutos, "
                                               f"a {pos}ª maior percentagem da liga."))
        recent = long[(long.date >= "2022-08-01") & long.comp.isin(["1div", "2div_norte", "2div_sul"]) & long.home]
        hw = recent.assign(w=recent.margin > 0).groupby("team").w.mean()
        pos, n, v = rank(hw, team)
        if v is not None and pos == 1:
            facts.append(("Casa", f"Fortaleza: ganha {round(100 * v)}% dos jogos em casa no campeonato desde 2022/23, o melhor registo da liga."))
        disc = pd.read_csv(DATA / "game_discipline.csv")
        dl = pd.concat([disc.assign(team=disc.home, blue=disc.blue_h), disc.assign(team=disc.away, blue=disc.blue_a)])
        bl = dl[dl.season >= "2023-24"].groupby("team").blue.mean()
        pos, n, v = rank(bl, team, high=False)
        if v is not None and pos == 1:
            facts.append(("Disciplina", f"A mais disciplinada da liga: {str(round(v, 2)).replace('.', ',')} cartões azuis por jogo desde 2023/24."))
        pos, n, v = rank(bl, team, high=True)
        if v is not None and pos == 1:
            facts.append(("Disciplina", f"A que mais cartões azuis vê na liga: {str(round(v, 2)).replace('.', ',')} por jogo desde 2023/24."))

        for i, (cat, txt) in enumerate(facts, 1):
            rows.append(dict(id=f"{team}_{i}".replace(" ", "-"), liga=LEAGUES[comp], equipa=nm(team), ordem=i,
                             categoria=cat, facto=txt))
    out = pd.DataFrame(rows)
    (DATA / "site").mkdir(exist_ok=True)
    out.to_csv(DATA / "site" / "factos.csv", index=False)
    print(f"{len(out)} facts for {out.equipa.nunique()} teams")
    return out


if __name__ == "__main__":
    f = main()
    for t in ["Porto", "Benfica", "Riba d'Ave"]:
        print("\n".join(f"{t}: [{r.categoria}] {r.facto}" for r in f[f.equipa == t].itertuples()))
