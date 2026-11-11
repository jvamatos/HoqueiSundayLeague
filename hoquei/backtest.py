"""Walk-forward backtest: every week, refit on games played *before* that week and
predict that week's games. No future information leaks into a prediction.

Compares the Poisson team model against two baselines:
  * base rates  - always predict the historical home/draw/away frequencies
  * Elo         - goal-difference Elo with home advantage

Metrics (lower is better): log loss and RPS on the regulation-time result (H/D/A),
plus accuracy of the most likely outcome and of the exact most likely score.
"""
import argparse
import itertools

import numpy as np
import pandas as pd

from .common import DATA
from .model import Params, TeamModel, division, regulation_scores


def outcomes(m):
    hg, ag = regulation_scores(m)
    return np.select([hg > ag, hg == ag], [0, 1], 2), hg, ag


def metrics(P, y):
    P = np.clip(np.asarray(P, float), 1e-9, 1)
    P = P / P.sum(1, keepdims=True)
    onehot = np.eye(3)[y]
    logloss = -np.mean(np.log(P[np.arange(len(y)), y]))
    cum_p, cum_o = np.cumsum(P, 1)[:, :2], np.cumsum(onehot, 1)[:, :2]
    rps = np.mean(np.sum((cum_p - cum_o) ** 2, 1) / 2)
    acc = np.mean(P.argmax(1) == y)
    return dict(n=len(y), logloss=logloss, rps=rps, accuracy=acc)


# ----------------------------------------------------------------------------- Elo
class Elo:
    def __init__(self, k=24, home=60, new_team=1350, draw=0.12, regress=0.25):
        self.k, self.home, self.new, self.draw, self.regress = k, home, new_team, draw, regress
        self.r, self.season = {}, None

    def probs(self, h, a):
        rh, ra = self.r.get(h, self.new), self.r.get(a, self.new)
        e = 1 / (1 + 10 ** (-(rh + self.home - ra) / 400))
        d = self.draw * 4 * e * (1 - e)  # draws likelier between close teams
        return np.array([(1 - d) * e, d, (1 - d) * (1 - e)])

    def update(self, h, a, hg, ag, season):
        if season != self.season:  # pull ratings toward the mean between seasons
            mean = np.mean(list(self.r.values())) if self.r else 1500
            self.r = {t: v + self.regress * (mean - v) for t, v in self.r.items()}
            self.season = season
        rh, ra = self.r.get(h, self.new), self.r.get(a, self.new)
        e = 1 / (1 + 10 ** (-(rh + self.home - ra) / 400))
        s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        mult = np.log1p(abs(hg - ag))
        self.r[h] = rh + self.k * mult * (s - e)
        self.r[a] = ra - self.k * mult * (s - e)


def run_elo(m, test_mask):
    elo, P = Elo(), {}
    for r in m[m.played].itertuples():
        if test_mask.loc[r.Index]:
            P[r.Index] = elo.probs(r.home, r.away)
        hg = r.home_goals - np.nan_to_num(r.home_ot) - np.nan_to_num(r.home_pen)
        ag = r.away_goals - np.nan_to_num(r.away_ot) - np.nan_to_num(r.away_pen)
        elo.update(r.home, r.away, hg, ag, r.season)
    return pd.DataFrame(P, index=["p_home", "p_draw", "p_away"]).T


# ------------------------------------------------------------------- Poisson model
def run_model(m, test_mask, params, refit_days=7):
    test = m[test_mask]
    dates = pd.to_datetime(test.date)
    week_start = dates.min()
    preds = []
    while week_start <= dates.max():
        week_end = week_start + pd.Timedelta(days=refit_days)
        chunk = test[(dates >= week_start) & (dates < week_end)]
        if len(chunk):
            model = TeamModel(params).fit(m, as_of=week_start)
            preds.append(model.predict_df(chunk))
        week_start = week_end
    return pd.concat(preds)


def evaluate(m, test_mask, params, label=None, verbose=True):
    test = m[test_mask]
    y, hg, ag = outcomes(test)
    pm = run_model(m, test_mask, params)
    res = {"model": metrics(pm[["p_home", "p_draw", "p_away"]].loc[test.index].values, y)}
    exact = (pm.loc[test.index].likely_score == hg.astype(int).astype(str) + "-" + ag.astype(int).astype(str))
    res["model"]["exact_score"] = exact.mean()
    if verbose:
        pe = run_elo(m, test_mask)
        res["elo"] = metrics(pe.loc[test.index].values, y)
        train = m[m.played & ~test_mask & (pd.to_datetime(m.date) < pd.to_datetime(test.date).min())]
        yb, _, _ = outcomes(train)
        base = np.bincount(yb, minlength=3) / len(yb)
        res["base_rates"] = metrics(np.tile(base, (len(y), 1)), y)
    return pd.DataFrame(res).T, pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="grid-search params on 2021-23")
    args = ap.parse_args()
    m = pd.read_csv(DATA / "matches.csv")
    m = m[m.played].reset_index(drop=True)
    d = pd.to_datetime(m.date)

    if args.tune:
        val = (d >= "2021-08-01") & (d < "2023-08-01")
        rows = []
        for hl, l2, db in itertools.product([120, 180, 240, 365, 540], [1, 3, 10], [0.0, 0.1, 0.2]):
            r, _ = evaluate(m, val, Params(hl, l2, db), verbose=False)
            rows.append(dict(half_life=hl, l2=l2, draw_boost=db, **r.loc["model"].to_dict()))
            print(rows[-1])
        print(pd.DataFrame(rows).sort_values("logloss").head(10).to_string())
        return

    test = d >= "2023-08-01"  # 2023/24 onwards: never used for tuning
    params = Params()
    res, pm = evaluate(m, test, params)
    print(f"Backtest {m[test].date.min()} -> {m[test].date.max()}  ({test.sum()} games)  params={params}")
    print(res.round(4).to_string())
    print("\nBy competition (model):")
    y, _, _ = outcomes(m[test])
    for comp in ["1div", "2div_norte", "2div_sul"]:
        mask = (m[test].competition == comp).values
        r = metrics(pm.loc[m[test].index][["p_home", "p_draw", "p_away"]].values[mask], y[mask])
        print(f"  {comp:10s} " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()))
    # calibration of home-win probability
    t = m[test].join(pm[["p_home", "p_draw", "p_away"]])
    t["y"] = y
    t["bin"] = pd.cut(t.p_home, [0, .2, .4, .6, .8, 1])
    cal = t.groupby("bin", observed=True).agg(predicted=("p_home", "mean"),
                                              actual=("y", lambda s: (s == 0).mean()), n=("y", "size"))
    print("\nCalibration (home win):\n" + cal.round(3).to_string())
    print(f"\nDraws: predicted {t.p_draw.mean():.3f}, actual {(t.y == 1).mean():.3f}")
    out = m[test][["date", "competition", "home", "away", "home_goals", "away_goals"]].join(pm.drop(columns=["home", "away"]))
    out.to_csv(DATA / "backtest_predictions.csv", index=False)


if __name__ == "__main__":
    main()
