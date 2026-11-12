"""Compare model variants with the walk-forward backtest.

    python -m hoquei.experiments
"""
import pandas as pd

from .backtest import evaluate, load_matches as _load


def load_matches():
    return _load(("features_players.csv", "features_discipline.csv", "features_impact.csv"))
from .model import Params

BASE = dict(half_life_days=240, l2=2.0, prior_half_life=1095)

SQ_GK = (("sq_known", "own"), ("gk_known", "opp"))
IMP = (("imp_att", "own"), ("imp_def", "opp"))
VARIANTS = {
    "baseline (no prior)": Params(240, 0.5, prior_half_life=None, features=()),
    "long-run prior": Params(**BASE, features=()),
    "+ squad (known)": Params(**BASE, features=(("sq_known", "own"),)),
    "+ squad + gk (known)": Params(**BASE, features=(("sq_known", "own"), ("gk_known", "opp"))),
    "+ squad + gk (expected)": Params(**BASE, features=(("sq_exp", "own"), ("gk_exp", "opp"))),
    "+ squad + gk + power play": Params(**BASE, features=SQ_GK + (("ppx", "own"),)),
    "+ squad + gk + referee": Params(**BASE, features=SQ_GK + (("ref", "own"),)),
    "+ squad + gk + pp + referee": Params(**BASE, features=SQ_GK + (("ppx", "own"), ("ref", "own"))),
    "+ squad + gk + impact": Params(**BASE, features=SQ_GK + IMP),
    "impact only (no squad/gk)": Params(**BASE, features=IMP),
}


def run(variants=VARIANTS, periods=None):
    m = load_matches()
    d = pd.to_datetime(m.date)
    periods = periods or {"val 2021-23": (d >= "2021-08-01") & (d < "2023-08-01"),
                          "test 2023-26": d >= "2023-08-01"}
    out = []
    for name, params in variants.items():
        for pname, mask in periods.items():
            r, _ = evaluate(m, mask, params, verbose=False)
            out.append(dict(variant=name, period=pname, **r.loc["model"][["logloss", "rps", "accuracy"]].to_dict()))
            print(out[-1], flush=True)
    res = pd.DataFrame(out).pivot(index="variant", columns="period", values=["logloss", "accuracy"])
    print(res.round(4).to_string())
    return res


if __name__ == "__main__":
    import sys
    names = sys.argv[1:]
    run({k: v for k, v in VARIANTS.items() if not names or k in names})
