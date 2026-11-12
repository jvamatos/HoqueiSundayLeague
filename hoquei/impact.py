"""Player impact ratings (adjusted plus-minus) and leak-free impact features.

Data: who played in each game (2021/22 onwards, hoqueipatins.pt line-ups). The play-by-play
logs goalkeeper changes but almost never outfield substitutions, so on-rink time per goal
can't be rebuilt; we use game-level participation instead:
  * outfield players who played: starters weight 1.0, players from the bench 0.6, rescaled
    so the outfield weights of a team in a game sum to 4 (four outfield players on the rink)
  * goalkeepers who played: the starter 1.0, or split 0.75 / 0.25 if both played

Model (Poisson, regulation-time goals of each team in each game):
    log(goals of T vs O) = level[div] + home_adv*[T at home]
                           + sum_p w_p * attack_p  (players of T)
                           - sum_q w_q * defence_q (players of O)
Every player's attack and defence are fitted together (so each gets credit only for what
he adds beyond teammates and opponents), with a ridge penalty that pulls players with
little data toward 0, and recent games weighted more.

Features for the match model (leak-free: ratings are refitted every REFIT_DAYS using only
earlier games; a game's squad weights use only the players' earlier games):
  * imp_att_h/a = expected attack impact of the called-up squad - the team's usual squad's
  * imp_def_h/a = same for defence (goalkeeper included)

Outputs: data/player_impact.csv (latest ratings), data/features_impact.csv
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from .common import DATA
from .model import division, regulation_scores

HALF_LIFE = 365.0
L2 = 25.0
BENCH_W = 0.6
REFIT_DAYS = 28
USUAL_GAMES = 8


def load_participation():
    m = pd.read_csv(DATA / "api_matches.csv")
    m = m[m.played.astype(bool)].copy()
    m["hg"], m["ag"] = regulation_scores(m)
    m["date"] = pd.to_datetime(m.date)
    p = pd.read_csv(DATA / "api_player_stats.csv")
    for c in ["convocado", "jogou", "cincoinicial", "entrouemjogodobanco"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    p["goalkeeper"] = p.goalkeeper.astype(bool)
    p = p[p.event_id.isin(m.event_id) & p.player_id.notna()]
    p = p.drop_duplicates(["event_id", "team_id", "player_id"])

    # actual weights of who played
    w = np.where(p.jogou > 0, np.where(p.cincoinicial > 0, 1.0, BENCH_W), 0.0)
    p["w"] = w
    sk = ~p.goalkeeper
    tot = p[sk].groupby(["event_id", "team_id"]).w.transform("sum")
    p.loc[sk, "w"] = np.where(tot > 0, p.loc[sk, "w"] * 4 / tot, 0)
    gk = p.goalkeeper & (p.jogou > 0)
    n_gk = p[gk].groupby(["event_id", "team_id"]).player_id.transform("count")
    p.loc[gk, "w"] = np.where(n_gk > 1, np.where(p.loc[gk, "cincoinicial"] > 0, 0.75, 0.25), 1.0)
    p.loc[p.goalkeeper & (p.jogou <= 0), "w"] = 0
    return m, p


def team_games(m):
    """Two rows per game: team T (goals for) against opponent O."""
    a = pd.DataFrame({"event_id": m.event_id, "date": m.date, "comp": m.competition,
                      "team": m.home_id, "opp": m.away_id, "home": 1, "goals": m.hg})
    b = pd.DataFrame({"event_id": m.event_id, "date": m.date, "comp": m.competition,
                      "team": m.away_id, "opp": m.home_id, "home": 0, "goals": m.ag})
    return pd.concat([a, b], ignore_index=True)


def fit(m, p, as_of, l2=L2, half_life=HALF_LIFE):
    """Fit player attack/defence on games before as_of. Returns dict player -> (att, def)."""
    mm = m[m.date < as_of]
    if len(mm) < 200:
        return {}, {}
    tg = team_games(mm)
    tg["wt"] = 0.5 ** ((as_of - tg.date).dt.days / half_life)
    pp = p[p.event_id.isin(mm.event_id) & (p.w > 0)]
    players = np.sort(pp.player_id.unique())
    idx = {pid: i for i, pid in enumerate(players)}
    P = len(players)
    row_of = {(e, t): i for i, (e, t) in enumerate(zip(tg.event_id, tg.team))}
    # X_for[r, p] = weight of p if p plays for row's team; X_against[r, p] if p plays for row's opponent
    rows_f, rows_a, cols, vals = [], [], [], []
    for e, t, pid, w in zip(pp.event_id, pp.team_id, pp.player_id, pp.w):
        rows_f.append(row_of.get((e, t), -1)); cols.append(idx[pid]); vals.append(w)
    rows_f = np.array(rows_f); cols = np.array(cols); vals = np.array(vals)
    ok = rows_f >= 0
    R = len(tg)
    Xf = sparse.csr_matrix((vals[ok], (rows_f[ok], cols[ok])), shape=(R, P))
    # the opponent's row of the same game is the "other" row
    other = np.array([row_of[(e, o)] for e, o in zip(tg.event_id, tg.opp)])
    Xo = Xf[other]
    g = (tg.comp.map(division) == "2div").astype(int).values
    h = tg.home.values.astype(float)
    y = tg.goals.values.astype(float)
    wt = tg.wt.values

    def nll(x):
        lev, ha, att, dfn = x[0:2], x[2], x[3:3 + P], x[3 + P:]
        eta = lev[g] + ha * h + Xf @ att - Xo @ dfn
        lam = np.exp(eta)
        r = wt * (lam - y)
        f = np.sum(wt * (lam - y * eta)) + l2 * (att @ att + dfn @ dfn)
        grad = np.concatenate([np.bincount(g, r, 2), [r @ h], Xf.T @ r + 2 * l2 * att, -(Xo.T @ r) + 2 * l2 * dfn])
        return f, grad

    x0 = np.zeros(3 + 2 * P)
    x0[0:2] = np.log(np.average(y, weights=wt))
    res = minimize(nll, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    att, dfn = res.x[3:3 + P], res.x[3 + P:]
    games = pp.groupby("player_id").w.size()
    return dict(zip(players, zip(att, dfn))), games.to_dict()


def features(m, p, start="2022-08-01"):
    """Leak-free impact features for every API game from `start` on."""
    m = m.sort_values(["date", "event_id"])
    called = p[p.convocado > 0]
    by_game = {k: g for k, g in called.groupby(["event_id", "team_id"])}
    # expected weight of each player = mean of his earlier actual weights (decayed), else defaults
    hist = {}   # pid -> list of (date, w, is_gk_start)

    def exp_weights(g, d):
        out = []
        for pid, isgk in zip(g.player_id, g.goalkeeper):
            h = [w for (dd, w) in hist.get(pid, []) if dd < d][-10:]
            out.append((pid, isgk, np.mean(h) if h else (0.3 if isgk else 0.5)))
        sk = [(pid, w) for pid, isgk, w in out if not isgk]
        gk = sorted([(pid, w) for pid, isgk, w in out if isgk], key=lambda x: -x[1])
        tot = sum(w for _, w in sk) or 1
        res = [(pid, w * 4 / tot) for pid, w in sk]
        if gk:
            res.append((gk[0][0], 1.0))  # most likely starter
        return res

    team_hist = {}  # team_id -> list of (date, [(pid, w), ...]) of earlier games (expected squads)
    rows = []
    snap, ratings = None, {}
    for r in m.itertuples():
        if r.date >= pd.Timestamp(start) and (snap is None or r.date >= snap + pd.Timedelta(days=REFIT_DAYS)):
            snap = r.date
            ratings, _ = fit(m, p, snap)
        row = {"event_id": r.event_id}
        for side, tid in (("h", r.home_id), ("a", r.away_id)):
            g = by_game.get((r.event_id, tid))
            squad = exp_weights(g, r.date) if g is not None else []
            prev = [s for d, s in team_hist.get(tid, []) if d < r.date][-USUAL_GAMES:]
            if r.date >= pd.Timestamp(start) and ratings and squad and prev:
                strength = lambda sq, k: sum(w * ratings.get(pid, (0.0, 0.0))[k] for pid, w in sq)
                row[f"imp_att_{side}"] = strength(squad, 0) - np.mean([strength(s, 0) for s in prev])
                row[f"imp_def_{side}"] = strength(squad, 1) - np.mean([strength(s, 1) for s in prev])
            if squad:
                team_hist.setdefault(tid, []).append((r.date, squad))
        rows.append(row)
        # update player histories with ACTUAL weights of this game (used only for later games)
        for pid, w in zip(*_actual(p_by_event, r.event_id)):
            hist.setdefault(pid, []).append((r.date, w))
    return pd.DataFrame(rows)


p_by_event = {}


def _actual(index, event_id):
    g = index.get(event_id)
    if g is None:
        return [], []
    return g.player_id.tolist(), g.w.tolist()


def main():
    global p_by_event
    m, p = load_participation()
    p_by_event = {e: g[g.w > 0] for e, g in p.groupby("event_id")}
    f = features(m, p)
    f.to_csv(DATA / "features_impact.csv", index=False)
    print(f.describe().round(3).T[["count", "mean", "std", "min", "max"]])

    # latest ratings for reports
    ratings, games = fit(m, p, m.date.max() + pd.Timedelta(days=1))
    names = pd.read_csv(DATA / "players.csv").set_index("player_id").name
    last = p[p.w > 0].merge(m[["event_id", "date", "home_id", "home", "away"]], on="event_id").sort_values("date")
    last["team"] = np.where(last.team_id == last.home_id, last.home, last.away)
    last = last.drop_duplicates("player_id", keep="last").set_index("player_id")
    out = pd.DataFrame([{"player_id": pid, "name": names.get(pid, str(pid)), "team": last.team.get(pid),
                         "goalkeeper": bool(last.goalkeeper.get(pid, False)), "games": games.get(pid, 0),
                         "attack": a, "defence": d, "impact": a + d} for pid, (a, d) in ratings.items()])
    out.sort_values("impact", ascending=False).to_csv(DATA / "player_impact.csv", index=False)
    print(out.sort_values("impact", ascending=False).head(15).round(3).to_string())


if __name__ == "__main__":
    main()
