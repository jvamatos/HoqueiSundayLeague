# HoqueiSundayLeague

A prediction model for Portuguese rink hockey (hóquei em patins). It covers the men's
**1ª Divisão**, **2ª Divisão Norte** and **2ª Divisão Sul**, using data from 2007/08 to today.

## Quick start

```bash
pip install -r requirements.txt
python -m hoquei.scrape_api --refresh   # results, line-ups, player stats (2021/22 onwards)
python -m hoquei.scrape_archive         # 2015/16-2020/21 (cached after the first run)
python -m hoquei.scrape_archive_old     # 2007/08-2014/15 (cached after the first run)
python -m hoquei.scrape_extra --refresh # cups, European competitions, Spanish/Italian/French leagues
python -m hoquei.build_dataset          # -> data/matches.csv
python -m hoquei.players                # player ratings + squad/goalkeeper features
python -m hoquei.scrape_events          # play-by-play: goals, cards, power plays, referees (slow first time)
python -m hoquei.discipline             # power-play / card / referee stats + features
python -m hoquei.pp_report              # -> data/analysis/power_play.md
python -m hoquei.rosters                # squads from the transfers pages (pre-season)
python -m hoquei.preseason              # pre-season squad strength (+ its backtest)
python -m hoquei.predict --absent "Benfica: Pau Bargallo; Porto: Malian"   # -> data/predictions/<today>/report.md
python -m hoquei.predict --preseason    # in August: use the transfers pages for squads
python -m hoquei.track_record           # score logged predictions against results
python -m hoquei.export_site            # tables for the private dashboard (data/site/)
python -m hoquei.backtest               # how good is the model?
python -m hoquei.experiments            # compare model variants
```

## Data

| Seasons | Source | What we get |
|---|---|---|
| 2007/08 – 2012/13 | archive "all rounds" pages (`*_jornadas_<yy-yy>.htm`) | results and dates (rescheduled dates where noted) |
| 2013/14 – 2014/15 | archive results matrices (`hp/_QRE_*.HTM`) + round pages for dates | results; part of 2013/14 has approximate dates because some round pages are gone |
| 2015/16 – 2020/21 | archive calendar pages (`hp/CAMN_*.HTM`); round pages for 2016/17 1ª Div. | results and dates |
| 2021/22 – now | `hoqueipatins.pt` SportsPress REST API | results incl. half-time, overtime and shootouts, play-offs, line-ups, goals by type, cards, goalkeeper stats |
| 2021/22 – now | match pages (`/evento/<id>/`), "Jogada a jogada" | play-by-play with minutes (from 2022/23), power plays, referees |
| 2021/22 – now | same API: Taça de Portugal, Elite Cup, Supertaça, WSE European cups, OK Liga, Serie A1, N1 Élite | 3,788 extra games: rate the European opponents and the top clubs' form |
| 2023/24 – now | transfers pages (`/transferencias-placard-<season>/`, 2ª Divisão from 2024/25) | who stays, leaves and arrives at each club, before the season |

* `data/matches.csv`: about 16,000 games (**14,973 played**: 11,341 in our three leagues since
  2007/08, plus cups, Europe and foreign leagues since 2021/22), with team names normalised
  across sources (see `ALIASES` in `hoquei/common.py`). Taça games against teams we never
  rate (3rd division and below) are left out.
* `data/rosters.csv`: squads from the transfers pages; 94% of names matched to player ids
  (the rest are mostly arrivals from abroad or youth teams).
* `data/api_player_stats.csv`: about 50,000 player-game rows. `data/players.csv` has the
  names. 431 of 1,213 players appear at more than one club, and player IDs stay the same
  when they move.
* `data/events.csv`: about 79,000 play-by-play actions. `data/referees.csv` has the
  officials of 2,900 games.
* Checks: the 2007/08 and 2015/16 1ª Div. totals match the archive's own tables exactly.
  2019/20 is incomplete because of COVID. Porto Santo withdrew in 2011/12.
* `data/raw/` holds the downloaded pages. It is a cache and is not committed. Requests are
  rate-limited to be polite to the site.

## Model (`hoquei/model.py`)

This is a time-weighted Poisson team-strength model:

```
log(home xG) = level[div] + home_adv[div] + attack[home] - defence[away] + β·features(home)
log(away xG) = level[div] +                 attack[away] - defence[home] + β·features(away)
```

* It is fitted on **regulation-time** scores. Overtime and shootout goals are removed.
* Recent games count more: the half-life is 240 days.
* **Long-run prior:** ratings are shrunk toward each club's 3-year (half-life) rating,
  not toward average. This is where the 2007+ history helps.
* The 1ª and 2ª Divisão are on one rating scale, linked by promoted and relegated teams, and
  since 2021/22 by cups and European games (foreign teams get ratings from their own league).
  Scoring level and home advantage are fitted per competition type (1ª Div., 2ª Div., cup,
  Europe, each foreign league); neutral-venue games (Elite Cup, Supertaça, final fours)
  have no home advantage.
* **Game tempo:** both teams' expected goals share one random factor per game (Gamma, k=40),
  because scores correlate (+0.08) and vary a bit more than Poisson. It improves exact-score
  predictions (validation 4.202 → 4.196, test 4.043 → 4.041 negative log-likelihood) and
  leaves who-wins predictions the same.
* **Player features** (`hoquei/players.py`, computed without looking ahead):
  * `sq_known`: the squad's scoring power vs. the team's usual squad. Each player's scoring
    rate follows them across clubs, so transfers count. Learned effect: about 0.65, so a
    squad 10% weaker scores about 6% fewer goals.
  * `gk_known`: the starting goalkeeper's "goals prevented" vs. the team's usual
    goalkeeper.
  * For future games the squad isn't known, so these are 0 unless you pass absences with
    `--absent` (applied to each team's next game).

## How good is it? (walk-forward backtest)

Every week the model is refitted on earlier games only, then predicts that week's games.
Parameters were tuned on 2021/22–2022/23; the test period 2023/24 → Nov 2026 was not used
for tuning.

| Variant | log loss, test 2023–26 ↓ | correct outcome, test | log loss, val 2021–23 |
|---|---|---|---|
| Base rates | 0.977 | 51.7% | – |
| Elo (goal-diff) | 0.851 | 64.5% | – |
| Poisson model (v1) | 0.819 | 65.4% | 0.798 |
| + long-run prior | 0.816 | 65.4% | **0.795** |
| + squad + goalkeeper (known squads) | 0.814 | 65.9% | 0.801 |
| **+ cups, Europe, foreign leagues + game tempo (current)** | **0.811** | **66.2%** | 0.799 |
| + squad + gk from the *previous* game's squad | 0.816 | 65.8% | 0.803 |
| + power-play feature | 0.814 | 65.7% | 0.804 |
| + referee feature | 0.815 | 65.8% | 0.801 |

A longer test, 2012/13–2020/21 (4,937 games): model 0.794 log loss / 66.6% vs. Elo
0.829 / 65.1%; adding the long-run prior gives 0.7937.

Current model on the test period: 1ª Div. 70.1% correct, 2ª Norte 63.8%, 2ª Sul 64.2%;
Elo 0.841 / 64.7% on the same 1,750 games. Calibration: when it gives the home team
80–100%, the home team won 91% of the time; 60–80% → 72%.

What we learned:
* **More competitions** help in every period (test 0.8132 → 0.8109, validation 0.7998 →
  0.7993): European games tell the model how strong the top Portuguese clubs really are.
* **Fatigue** (≤3 days since the last game, any competition): about −1% goals, no gain.
* **A direct draw adjustment** helps on test but hurts on validation, so it is not used.
* **The long-run prior** (from the 2007+ history) helps a little in every period.
* **Squads** help when we know who plays: missing scorers and goalkeepers move the odds.
  Knowing only last week's line-up adds almost nothing beyond the team rating. Player
  history starts in 2021, so the 2021–23 period has too little history and the feature
  hurts there.
* **Power play and discipline:** a team's blue cards per game and power-play success
  barely carry over from one season to the next (correlation ≈ 0.2). They are mostly noise
  and are already inside the team ratings, so they don't improve win/draw/loss
  predictions.
* **Referees:** stricter referee pairs mean more goals for *both* teams (effect ≈ +0.35 ×
  log strictness). This slightly improves goal-total predictions (over/under 7.5 log loss
  0.668 → 0.666) but not who wins, so it is not in the default model.

* **Player impact (adjusted plus-minus, `hoquei/impact.py`):** every player gets an attack
  and a defence rating from who played in each game since 2021/22 (starters weigh more than
  bench players), fitted together so each player is credited only for what he adds beyond
  teammates and opponents. Outfield substitutions are not in the play-by-play (99.9% of the
  logged substitutions are goalkeeper changes), so on-rink plus-minus per goal is not
  possible. Result: no reliable gain. Test 2023–26 0.8144 → 0.8141 log loss, but 2022/23
  gets worse (0.821 → 0.837 even with strong shrinkage). The attack part overlaps with the
  squad-scoring feature (its effect even gets the wrong sign), and the ratings mostly
  reflect team quality because regulars almost always play together. It is not in the
  default model; the ratings are in `data/player_impact.csv` as a statistic.

* **Transfers (season over season, `hoquei/transfers.py`):** at each season start, player
  ratings (impact and goals per game, earlier games only) are applied to last season's squad
  and to the squad called up so far this season. The difference is the transfer effect,
  which fades over the first ~8 games. The first games of a season are clearly the hardest
  to predict (0.877 log loss vs 0.814 over the whole season), but the feature barely helps:
  on the first 6 games per team in 2023–26, 0.8773 → 0.8761 and 61.1% → 62.0% correct,
  which is within noise. On 2022/23 it is worse, because the model had no earlier seasons
  to learn how much a transfer matters. Part of the effect is already in the squad feature,
  which compares the squad with the team's last games (i.e. last season's at the start). Not
  in the default model; the squad changes per team are in `data/transfers_summary.csv`.

* **Pre-season squads (transfers pages, `hoquei/rosters.py`, `hoquei/preseason.py`):** predicting
  each team's first 6 league games as if in August (2023/24–2025/26, 275 games): no squad
  information 0.8472 / 62.2% correct; squads from the transfers page **0.8443 / 64.4%**;
  the actual squad lists of each game 0.8426 / 63.3%. The transfers page gives most of the
  benefit weeks before the season. Use `python -m hoquei.predict --preseason`.
* **Bookmaker odds:** no archive of historical rink hockey odds was found, so there is no
  bookmaker benchmark yet. The only route is to record current odds each week from now on.

## Track record

`predict.py` appends every run's predictions to `data/prediction_log.csv`. For each game
the last prediction logged before the game day counts, and `python -m hoquei.track_record`
scores them once results are in. The live record started on 12 Nov 2026; backtest results are
never mixed in.

Power-play statistics (goals per PP, penalty kill, short-handed goals, cards and referee
tables) are in `data/analysis/power_play.md`. Key facts: about 1.1 power plays per team
per game, 0.5–0.7 goals per power play (including the free hit that comes with a blue
card), and goals come about 4× faster on the power play.

## Ideas / next steps

* Play-off series simulator (best-of-3/5, overtime, shootouts), for real title odds.
* Fetch the squad list (*convocatórias*) and the referee nominations before games, to use
  the squad and referee features for real predictions.
* Use the FPP suspension bulletins (`fpp.pt/docs-fpp/castigos-2026-27/`) automatically.
* Record bookmaker odds weekly from now on, as a tougher benchmark.
* Format the output for X posts.
