"""Live track record: every prediction made before a game, scored once the result is in.

predict.py appends each run's upcoming-game predictions to data/prediction_log.csv
(log_prediction below). For each game, the LAST prediction logged before the game's date counts.
This is the honest record of what was published: it starts on the first logged run, and
backtest results are never mixed in.

    python -m hoquei.track_record   -> data/track_record_games.csv, data/track_record_summary.csv
"""
import numpy as np
import pandas as pd

from .common import DATA
from .model import regulation_scores

LOG = DATA / "prediction_log.csv"
LOG_COLS = ["logged_at", "game", "date", "competition", "home", "away",
            "p_home", "p_draw", "p_away", "xg_home", "xg_away", "likely_score"]


def game_key(date, home, away):
    return f"{date}_{home}_{away}".replace(" ", "-")


def log_prediction(up, as_of):
    """Append predictions (one row per upcoming game) made on as_of."""
    rows = up.assign(logged_at=str(pd.Timestamp(as_of).date()),
                     game=[game_key(d, h, a) for d, h, a in zip(up.date, up.home, up.away)])[LOG_COLS]
    if LOG.exists():
        old = pd.read_csv(LOG)
        rows = pd.concat([old, rows], ignore_index=True)
    rows = rows.drop_duplicates(["logged_at", "game"], keep="last")
    rows.to_csv(LOG, index=False)


def score():
    if not LOG.exists():
        return pd.DataFrame(), pd.DataFrame()
    log = pd.read_csv(LOG)
    log = log[log.logged_at < log.date]  # made before the game day
    log = log.sort_values("logged_at").drop_duplicates("game", keep="last")
    m = pd.read_csv(DATA / "matches.csv")
    m = m[m.played.astype(bool)].copy()
    m["game"] = [game_key(d, h, a) for d, h, a in zip(m.date, m.home, m.away)]
    hg, ag = regulation_scores(m)
    m["result"] = hg.astype(int).astype(str) + "-" + ag.astype(int).astype(str)
    m["outcome"] = np.select([hg > ag, hg == ag], ["home", "draw"], "away")
    g = log.merge(m[["game", "season", "result", "outcome"]], on="game")
    if g.empty:
        return g, pd.DataFrame()
    P = g[["p_home", "p_draw", "p_away"]].values
    y = g.outcome.map({"home": 0, "draw": 1, "away": 2}).values
    g["favourite"] = np.array(["home", "draw", "away"])[P.argmax(1)]
    g["hit"] = g.favourite == g.outcome
    g["p_outcome"] = P[np.arange(len(g)), y]
    g["logloss"] = -np.log(np.clip(g.p_outcome, 1e-9, 1))
    g["exact"] = g.likely_score == g.result
    summ = g.groupby(["season", "competition"]).agg(games=("game", "size"), hits=("hit", "sum"),
                                                     accuracy=("hit", "mean"), logloss=("logloss", "mean"),
                                                     exact_scores=("exact", "sum")).reset_index()
    tot = pd.DataFrame([dict(season="all", competition="all", games=len(g), hits=int(g.hit.sum()),
                             accuracy=g.hit.mean(), logloss=g.logloss.mean(), exact_scores=int(g.exact.sum()))])
    summ = pd.concat([summ, tot], ignore_index=True)
    return g, summ


def main():
    g, s = score()
    if g.empty:
        print("No logged prediction has a result yet. The record starts with the first games after "
              "the first logged run.")
        return
    g.to_csv(DATA / "track_record_games.csv", index=False)
    s.to_csv(DATA / "track_record_summary.csv", index=False)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
