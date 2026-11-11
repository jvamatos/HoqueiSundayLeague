"""Team-strength model for rink hockey scores.

Expected goals:
    log(home_xg) = level[div] + home_adv[div] + attack[home] - defence[away]
    log(away_xg) = level[div]                 + attack[away] - defence[home]

* Fitted by weighted Poisson maximum likelihood on regulation-time scores.
* Recent games weigh more: weight = exp(-days_ago * ln2 / half_life).
* Ratings are shrunk toward 0 (L2 penalty), so teams with few games stay near average.
* Ratings are on ONE scale across 1st and 2nd division: teams that move between
  divisions (and promotion games) link the two pools.

The final-score distribution uses independent Poisson goals with an optional
draw adjustment (`draw_boost`), because Poisson usually under-predicts draws.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 25


def division(competition):
    return "1div" if competition == "1div" else "2div"


def regulation_scores(m):
    """Final score minus overtime and shootout goals (those are included in 'goals')."""
    hg = m.home_goals - m.home_ot.fillna(0) - m.home_pen.fillna(0)
    ag = m.away_goals - m.away_ot.fillna(0) - m.away_pen.fillna(0)
    return hg.clip(lower=0), ag.clip(lower=0)


@dataclass
class Params:
    half_life_days: float = 240.0
    l2: float = 0.5
    draw_boost: float = 0.0  # multiplies the diagonal of the score matrix by (1 + draw_boost)


@dataclass
class TeamModel:
    params: Params = field(default_factory=Params)
    teams: list = None
    attack: np.ndarray = None
    defence: np.ndarray = None
    level: dict = None
    home_adv: dict = None
    as_of: pd.Timestamp = None
    n_games: dict = None

    # ------------------------------------------------------------------ fitting
    def fit(self, matches, as_of=None):
        p = self.params
        m = matches[matches.played].copy()
        m["date"] = pd.to_datetime(m.date)
        as_of = pd.Timestamp(as_of) if as_of is not None else m.date.max() + pd.Timedelta(days=1)
        m = m[m.date < as_of]
        m["w"] = np.exp(-(as_of - m.date).dt.days.values * np.log(2) / p.half_life_days)
        m = m[m.w > 1e-3]
        hg, ag = regulation_scores(m)

        self.teams = sorted(set(m.home) | set(m.away))
        idx = {t: i for i, t in enumerate(self.teams)}
        T = len(self.teams)
        hi, ai = m.home.map(idx).values, m.away.map(idx).values
        g = (m.competition.map(division) == "2div").astype(int).values  # 0=1div, 1=2div
        y_h, y_a, w = hg.values.astype(float), ag.values.astype(float), m.w.values

        def unpack(x):
            return x[0:2], x[2:4], x[4:4 + T], x[4 + T:4 + 2 * T]

        def nll(x):
            lev, ha, att, dfn = unpack(x)
            eta_h = lev[g] + ha[g] + att[hi] - dfn[ai]
            eta_a = lev[g] + att[ai] - dfn[hi]
            lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)
            f = np.sum(w * (lam_h - y_h * eta_h + lam_a - y_a * eta_a))
            f += p.l2 * (att @ att + dfn @ dfn)
            r_h, r_a = w * (lam_h - y_h), w * (lam_a - y_a)
            g_lev = np.bincount(g, r_h + r_a, 2)
            g_ha = np.bincount(g, r_h, 2)
            g_att = np.bincount(hi, r_h, T) + np.bincount(ai, r_a, T) + 2 * p.l2 * att
            g_dfn = -np.bincount(ai, r_h, T) - np.bincount(hi, r_a, T) + 2 * p.l2 * dfn
            return f, np.concatenate([g_lev, g_ha, g_att, g_dfn])

        x0 = np.zeros(4 + 2 * T)
        x0[0:2] = np.log(np.average((y_h + y_a) / 2, weights=w))
        res = minimize(nll, x0, jac=True, method="L-BFGS-B")
        lev, ha, att, dfn = unpack(res.x)
        self.level = {"1div": lev[0], "2div": lev[1]}
        self.home_adv = {"1div": ha[0], "2div": ha[1]}
        self.attack, self.defence = att, dfn
        self.as_of = as_of
        counts = pd.concat([m.home, m.away]).value_counts()
        self.n_games = counts.to_dict()
        return self

    # --------------------------------------------------------------- prediction
    def _team(self, t):
        if t in self.teams:
            i = self.teams.index(t)
            return self.attack[i], self.defence[i]
        return 0.0, 0.0  # unknown team -> average

    def expected_goals(self, home, away, competition, neutral=False):
        d = division(competition)
        ah, dh = self._team(home)
        aa, da = self._team(away)
        lh = np.exp(self.level[d] + (0 if neutral else self.home_adv[d]) + ah - da)
        la = np.exp(self.level[d] + aa - dh)
        return lh, la

    def score_matrix(self, lh, la):
        k = np.arange(MAX_GOALS + 1)
        mat = np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))
        if self.params.draw_boost:
            mat[np.diag_indices_from(mat)] *= 1 + self.params.draw_boost
        return mat / mat.sum()

    def predict(self, home, away, competition, neutral=False):
        lh, la = self.expected_goals(home, away, competition, neutral)
        mat = self.score_matrix(lh, la)
        i, j = np.unravel_index(mat.argmax(), mat.shape)
        return dict(home=home, away=away, xg_home=lh, xg_away=la,
                    p_home=np.tril(mat, -1).sum(), p_draw=np.trace(mat), p_away=np.triu(mat, 1).sum(),
                    likely_score=f"{i}-{j}", p_likely_score=mat[i, j])

    def predict_df(self, games):
        rows = [self.predict(r.home, r.away, r.competition) for r in games.itertuples()]
        return pd.DataFrame(rows, index=games.index)

    def ratings(self):
        df = pd.DataFrame({"team": self.teams, "attack": self.attack, "defence": self.defence})
        df["overall"] = df.attack + df.defence
        df["games"] = df.team.map(self.n_games).fillna(0).astype(int)
        return df.sort_values("overall", ascending=False).reset_index(drop=True)
