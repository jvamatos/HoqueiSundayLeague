# HoqueiSundayLeague

A prediction model for Portuguese rink hockey (hóquei em patins). It covers the men's
**1ª Divisão**, **2ª Divisão Norte** and **2ª Divisão Sul**.

## Quick start

```bash
pip install -r requirements.txt
python -m hoquei.scrape_api --refresh   # new results from hoqueipatins.pt (current season)
python -m hoquei.scrape_archive         # 2015/16-2020/21 from arquivo.hoqueipatins.pt (cached)
python -m hoquei.build_dataset          # -> data/matches.csv
python -m hoquei.predict                # -> data/predictions/<today>/report.md
python -m hoquei.backtest               # how good is the model?
```

## Data

| Seasons | Source | What we get |
|---|---|---|
| 2015/16 – 2020/21 | `arquivo.hoqueipatins.pt` calendar pages (`hp/CAMN_*.HTM`), plus round pages for 2016/17 1ª Div. | Regular-season results and dates |
| 2021/22 – now | `hoqueipatins.pt` SportsPress REST API (`/wp-json/sportspress/v2/events`) | Results incl. half-time, overtime and shootouts, play-offs, line-ups, starting five, goals by type (open play / penalty / direct free hit), blue cards, warnings, goalkeeper stats |

* `data/matches.csv`: about 6,700 games (6,049 played). One row per game, with team names
  normalised across sources (see `ALIASES` in `hoquei/common.py`).
* `data/api_player_stats.csv`: about 50,000 player-game rows (2021/22 onwards). Not used by
  the model yet.
* `data/raw/` holds the downloaded pages. It is a cache and is not committed. Requests are
  rate-limited to be polite to the site.

Checks: 2015/16 1ª Div. totals match the archive's own table (Benfica 187 scored, Porto 61
conceded). 2019/20 is incomplete because of COVID. 2007/08–2014/15 use older, hand-made
page layouts and are not included yet.

## Model (`hoquei/model.py`)

This is a time-weighted Poisson team-strength model, the same family as the Dixon–Coles
models used in football:

```
log(home xG) = level[div] + home_adv[div] + attack[home] - defence[away]
log(away xG) = level[div] +                 attack[away] - defence[home]
```

* It is fitted on **regulation-time** scores. Overtime and shootout goals are removed,
  because the API includes them in the final score.
* Recent games count more: the half-life is 240 days. Ratings are shrunk toward average
  (L2 = 0.5).
* The 1ª and 2ª Divisão are on **one rating scale**. They are linked by teams that are
  promoted or relegated, and by promotion games.
* Outputs: home/draw/away probabilities, expected goals, the most likely score, and a
  Monte-Carlo simulation of the regular season (title, play-off top 8, relegation).

## Backtest (`hoquei/backtest.py`)

The backtest is walk-forward. Every week the model is refitted using only earlier games,
then it predicts that week's games. Parameters were tuned on 2021/22–2022/23. The test
below uses the **unseen** seasons 2023/24 → Nov 2026 (1,762 games).

| | log loss ↓ | RPS ↓ | correct outcome |
|---|---|---|---|
| **Poisson model** | **0.819** | **0.180** | **65.4%** |
| Elo (goal-diff) | 0.851 | 0.188 | 64.4% |
| Base rates | 0.977 | 0.239 | 51.7% |

By league: 1ª Div. 70.0% correct, 2ª Norte 62.6%, 2ª Sul 63.3%. Calibration is good: when
the model gives the home team 60–80%, the home team won 72% of the time. Predicted draws
were 12.2%; actual draws were 13.2%.

## Ideas / next steps

* Use the player data: missing top scorers or goalkeepers, form, and set-piece specialists.
* Blue cards → power-play / short-handed effects.
* A play-off series simulator (best-of-3/5, with overtime and shootouts).
* Add 2007/08–2014/15 from the older archive pages.
* Compare with bookmaker odds, if available, as a tougher benchmark.
* Later: format the output for X posts.
