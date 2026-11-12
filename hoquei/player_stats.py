"""Player statistics for the dashboard: goals by type, assists, set pieces, cards, goalkeepers.

League games (regular season + play-offs) for three periods: 2026/27, 2025/26, since 2023/24.
Only players whose last team is in this season's three leagues.

Sources:
  * per-player game stats (API): games, starts, goals, penalties and direct free hits
    scored / missed, cards; goalkeepers: goals conceded, penalties / free hits saved
  * play-by-play (2022/23 on): type of each goal (open play, power play, short-handed,
    penalty, direct free hit, rebound) and the assist. The scorer's and assister's names
    are matched to players of that game's squad lists.
  * Assists are only recorded in the 1ª Divisão, from 2025/26.

Output: data/site/jogadores_stats.csv, data/site/gr_stats.csv
"""
import re

import numpy as np
import pandas as pd

from .common import DATA
from .rosters import no_nick, plain
from .team_stats import goal_type

LEAGUES = {"1div": "1ª Divisão", "2div_norte": "2ª Divisão Norte", "2div_sul": "2ª Divisão Sul"}
PERIODS = {"2026/27": ("2026-08-01", "2100-01-01"), "2025/26": ("2025-08-01", "2026-08-01"),
           "desde 2023/24": ("2023-08-01", "2100-01-01")}
SEASON = "2026-27"
NUM = ["jogou", "cincoinicial", "golosmarcadostotal", "golomarcadodepenalty", "penaltyfalhado",
       "golomarcadodelivredireto", "livrediretofalhado", "cartesazuis", "cartesvermelhos",
       "golossofridostotal", "penaltydefendido", "livrediretodefendido", "golosofridodepenalty",
       "golosofridodelivredireto"]


def split_goal_text(t):
    """'6-1 Rafael "Rafa" Bessa, com assistência de Gonzalo "Nolito" Romero' -> (scorer, assister)."""
    t = re.sub(r"^\d+\s*-\s*\d+\s+", "", str(t)).strip()
    t = re.sub(r"^Golo( de [a-zçãé ]+)? de ", "", t)  # 2021/22 format: "Golo de X, Team"
    scorer, _, assist = t.partition(", com assistência de ")
    scorer = re.sub(r"\s*\([^)]*\)\s*$", "", scorer).strip()
    assist = re.sub(r"\s*\([^)]*\)\s*$", "", assist).strip()
    return scorer, assist or None


def main():
    names = pd.read_csv(DATA / "teams.csv").set_index("team").display.to_dict()
    pn = pd.read_csv(DATA / "players.csv").set_index("player_id").name
    am = pd.read_csv(DATA / "api_matches.csv")
    am = am[am.played.astype(bool) & am.competition.isin(list(LEAGUES) + ["2div"])].copy()
    am["date"] = pd.to_datetime(am.date)
    cur = pd.read_csv(DATA / "matches.csv")
    cur = cur[(cur.season == SEASON) & cur.competition.isin(LEAGUES)]
    league_of = dict(zip(cur.home, cur.competition)) | dict(zip(cur.away, cur.competition))

    ps = pd.read_csv(DATA / "api_player_stats.csv")
    for c in NUM:
        ps[c] = pd.to_numeric(ps[c], errors="coerce").fillna(0)
    ps = ps.merge(am[["event_id", "date", "competition", "home_id", "home", "away"]], on="event_id")
    ps["team"] = ps.home.where(ps.team_id == ps.home_id, ps.away)
    ps["goalkeeper"] = ps.goalkeeper.astype(bool)
    ps["key"] = ps.player_id.map(pn).map(plain)
    ps["nn"] = ps.player_id.map(pn).map(no_nick)

    # ---- play-by-play goals -> scorer / assister ids, by matching names in that game's squads
    ev = pd.read_csv(DATA / "events.csv")
    goals = ev[ev.icon.str.startswith("golo") & ev.side.isin(["home", "away"])].merge(
        am[["event_id", "date", "home_id", "away_id"]], on="event_id")
    squads = {k: g for k, g in ps.groupby(["event_id", "team_id"])}

    def find(name, event_id, team_id):
        if not name:
            return None
        g = squads.get((event_id, team_id))
        if g is None:
            return None
        k, nn = plain(name), no_nick(name)
        for col, val in (("key", k), ("nn", nn)):
            hit = g[g[col] == val]
            if len(hit) == 1:
                return hit.player_id.iloc[0]
        words = set(nn.split())
        hit = g[g.nn.map(lambda s: bool(words) and words <= set(str(s).split()))]
        return hit.player_id.iloc[0] if len(hit) == 1 else None

    rows = []
    for r in goals.itertuples():
        tid = r.home_id if r.side == "home" else r.away_id
        sc, ast = split_goal_text(r.text)
        rows.append(dict(event_id=r.event_id, date=r.date, type=goal_type(r.icon),
                         scorer=find(sc, r.event_id, tid), assister=find(ast, r.event_id, tid)))
    gl = pd.DataFrame(rows)
    print(f"play-by-play goals matched to a scorer: {gl.scorer.notna().mean():.0%}")

    last_team = ps.sort_values("date").drop_duplicates("player_id", keep="last").set_index("player_id").team
    out, gks = [], []
    for pname, (a, b) in PERIODS.items():
        p = ps[(ps.date >= a) & (ps.date < b) & (ps.jogou > 0)]
        g = gl[(gl.date >= a) & (gl.date < b)]
        types = g[g.scorer.notna()].pivot_table(index="scorer", columns="type", values="event_id",
                                                aggfunc="count", fill_value=0)
        assists = g[g.assister.notna()].groupby("assister").size()
        tot = p.groupby("player_id")[NUM].sum()
        games = p.groupby("player_id").event_id.nunique()
        team_now = p.sort_values("date").drop_duplicates("player_id", keep="last").set_index("player_id").team
        is_gk = p.groupby("player_id").goalkeeper.max()
        a1 = p[p.competition == "1div"].groupby("player_id").event_id.nunique()  # games where assists exist
        for pid, t in tot.iterrows():
            team = last_team.get(pid)
            if team not in league_of:
                continue
            base = {"id": f"{pname}_{int(pid)}".replace("/", "-").replace(" ", "-"), "periodo": pname,
                    "liga": LEAGUES[league_of[team]], "jogador": str(pn.get(pid, pid)).replace("®", "").strip(),
                    "equipa": names.get(team_now.get(pid), team_now.get(pid)), "jogos": int(games.get(pid, 0))}
            if is_gk.get(pid):
                faced_p = t.penaltydefendido + t.golosofridodepenalty
                faced_f = t.livrediretodefendido + t.golosofridodelivredireto
                gks.append({**base, "golos_sofridos": int(t.golossofridostotal),
                            "sofridos_por_jogo": round(t.golossofridostotal / max(games.get(pid, 1), 1), 2),
                            "penaltis_defendidos": f"{int(t.penaltydefendido)}/{int(faced_p)}",
                            "pct_penaltis_defendidos": round(100 * t.penaltydefendido / faced_p) if faced_p >= 3 else None,
                            "livres_defendidos": f"{int(t.livrediretodefendido)}/{int(faced_f)}",
                            "pct_livres_defendidos": round(100 * t.livrediretodefendido / faced_f) if faced_f >= 3 else None,
                            "azuis": int(t.cartesazuis)})
                continue
            ty = types.loc[pid] if pid in types.index else pd.Series(dtype=float)
            has_ast = pname != "desde 2023/24" and a1.get(pid, 0) > 0 and pname in ("2025/26", "2026/27")
            ast = int(assists.get(pid, 0)) if has_ast else None
            pen_att = t.golomarcadodepenalty + t.penaltyfalhado
            fh_att = t.golomarcadodelivredireto + t.livrediretofalhado
            out.append({**base, "titular": int(t.cincoinicial), "golos": int(t.golosmarcadostotal),
                        "golos_por_jogo": round(t.golosmarcadostotal / max(games.get(pid, 1), 1), 2),
                        "assistencias": ast,
                        "golos_mais_assist": int(t.golosmarcadostotal) + ast if ast is not None else None,
                        "bola_corrida": int(ty.get("bola_corrida", 0)), "power_play": int(ty.get("power_play", 0)),
                        "inferioridade": int(ty.get("inferioridade", 0)), "recarga": int(ty.get("recarga", 0)),
                        "penaltis": f"{int(t.golomarcadodepenalty)}/{int(pen_att)}",
                        "penaltis_marcados": int(t.golomarcadodepenalty),
                        "livres": f"{int(t.golomarcadodelivredireto)}/{int(fh_att)}",
                        "livres_marcados": int(t.golomarcadodelivredireto),
                        "azuis": int(t.cartesazuis), "vermelhos": int(t.cartesvermelhos)})
    o = pd.DataFrame(out).sort_values(["periodo", "golos"], ascending=[True, False])
    o.to_csv(DATA / "site" / "jogadores_stats.csv", index=False)
    gk = pd.DataFrame(gks).sort_values(["periodo", "jogos"], ascending=[True, False])
    gk.to_csv(DATA / "site" / "gr_stats.csv", index=False)
    print(f"{len(o)} player rows, {len(gk)} goalkeeper rows")
    return o, gk


if __name__ == "__main__":
    o, gk = main()
    x = o[(o.periodo == "2025/26") & (o.liga == "1ª Divisão")].sort_values("golos", ascending=False).head(10)
    print(x[["jogador", "equipa", "jogos", "golos", "assistencias", "bola_corrida", "power_play", "penaltis", "livres", "recarga"]].to_string())
    print(o[(o.periodo == "2025/26") & (o.liga == "1ª Divisão")].sort_values("assistencias", ascending=False).head(5)[["jogador", "equipa", "assistencias", "golos"]].to_string())
