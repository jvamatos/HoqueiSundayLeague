"""Compare model variants with the walk-forward backtest.

    python -m hoquei.experiments
"""
import pandas as pd

from .backtest import evaluate, load_matches
from .model import Params

BASE = dict(half_life_days=240, l2=2.0, prior_half_life=1095)

VARIANTS = {
    "baseline (no prior)": Params(240, 0.5, prior_half_life=None, features=()),
    "long-run prior": Params(**BASE, features=()),
    "+ squad (known)": Params(**BASE, features=(("sq_known", "own"),)),
    "+ squad + gk (known)": Params(**BASE, features=(("sq_known", "own"), ("gk_known", "opp"))),
    "+ squad + gk (expected)": Params(**BASE, features=(("sq_exp", "own"), ("gk_exp", "opp"))),
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
    run()
