# ShortStack Neural — heads-up short-deck PLO5 bot

A CPU-only bot for the TAMU Datathon Heads Up challenge. `agent.py` retains the
organizers' HTTP protocol and delegates decisions to `strategy.py`.
The participant identity is `jr007`.

The bot combines **PyPokerEngine's open source Monte Carlo equity approach**,
an event-specific trained neural equity predictor, and an explicit adaptive
betting policy. The neural weights were trained locally using scikit-learn;
they are not pretrained Hold'em weights or a fine-tuned language model. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and attribution.

The frozen upgrade beat the previous submitted bot on 3,000 fresh duplicate
deals: **+0.377 bb per duplicate deal**, with a whole-seed 95% interval
**[+0.139, +0.616]**. Validation covered 120,000 hands across both versions and
ten profiles, with zero errors. Improvement over the wider external field is
still uncertain. The complete results and limits are in
[docs/NEURAL_REVISION.md](docs/NEURAL_REVISION.md).
The six-hour improvement worker and its protected validation protocol are
described in [docs/OVERNIGHT_RUN.md](docs/OVERNIGHT_RUN.md).

## Run

Requires Python 3.12 (tested on Linux amd64):

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python agent.py
```

To play against jr007 in your browser:

```sh
npm install
npm run dev
```

Open `http://127.0.0.1:3000` on that machine. The npm launcher starts or reuses
the Python bot server using the virtual environment above; no additional npm
dependencies are required. In the managed cloud, use the
environment's Desktop browser. The table deals five cards, provides the four
legal action buttons, keeps a session score and reveals the bot's cards only
at showdown. Each hand resets both stacks to 80 practice chips. See
[docs/HUMAN_PLAY.md](docs/HUMAN_PLAY.md).

In separate terminals, from this directory:

```sh
.venv/bin/python baseline-bots/baseline_random.py
.venv/bin/python judge_engine.py --deals 1000 --seed 7 --json results/match.json
```

`PORT` chooses the bot's listening port; default 5008. The opponent defaults to
5009. Cold startup builds exact hand tables and JIT-compiles the equity engine;
subsequent decisions need no disk writes or network. Memory and learning reset
at every new half. One CPU thread is used for the simulations.

## Strategy

- Evaluate hands using exactly two of five hole cards and three board cards;
  flushes beat full houses and trips beat straights, including the A6789 wheel.
- Load a 1.2 MB preflop model fitted with 40.96 million simulated deals, then
  sample legal unknown hands and future boards, sharing no hidden information
  with the referee. Ties contribute half a win.
- Anchor flop/turn sampling with a 38 KB neural equity predictor trained on
  80,000 legal positions. Limit its influence using measured model error and
  Monte Carlo covariance; neural prediction accuracy alone is not a win rate.
- Learn recent opponent tendencies separately by street and betting context
  in `opponent_model.py`, with smoothed priors and uncertainty estimates.
- Credit rare value raises differently from frequent loose bets; avoid copying
  an aggressive opponent's raise frequency. Fold reads include wager price.
- Fade stale evidence as play changes. Require a clearer edge for pot raises
  and use conservative fold estimates when evaluating speculative pressure.
- Compare calls with pot odds; estimate half-pot and pot-raise value against a
  continuation range. Smooth observed folds and aggression across each half.
- Remove future-play penalties from terminal calls and bound remaining risk by
  the chips that can actually be lost. Use separate half-pot and pot margins.
- Cache simulations within a street so repeated raises do not repeat sampling.
- Keep the model compact, deterministic and offline; no language model or GPU.

## Evaluate and tune

```sh
.venv/bin/python tools/benchmark.py --deals 1000 --seeds 11 29 73 \
  --opponents random call half pot checkfold mc_loose mc_tight mc_range \
  --json results/benchmark.json
.venv/bin/python tools/tune.py --space tools/tuning_space.json --deals 150 \
  --iterations 8 --output results/tuning.json --params-output results/candidate.json
```

To reproduce the fitted preflop model (roughly 1–3 minutes on this machine):

```sh
.venv/bin/python tools/train_preflop.py --samples 2048 --seed 20261108
```

The shipped betting parameters are in `policy.json`; tuning outputs do not
replace them automatically. Keep fresh validation deals separate if retuning.

The benchmark uses duplicate cards with seats swapped, fresh memory per half,
and paired 95% confidence intervals. Its direct function timing **excludes HTTP**
and does not enforce a time bank. Use the official judge to validate deployment
and time usage. The independent MC opponents are stronger stress tests; their
own slow Python simulations need not meet the event's time limit.
Earlier Docker resource requests did not establish enforcement in this
rootless cloud; [docs/RUNTIME_LIMITS.md](docs/RUNTIME_LIMITS.md) records the
correction and the newer OS-enforced certification method.

Development tuning seeds and validation seeds are separate. Synthetic opponents
cannot establish performance against unknown tournament entries. Detailed
current comparisons and limits are in
[docs/NEURAL_REVISION.md](docs/NEURAL_REVISION.md).
[docs/ADAPTATION.md](docs/ADAPTATION.md) records the earlier adaptive version;
[EVALUATION.md](EVALUATION.md) records the original v1 results.
For the neural model's training and reproduction, see
[docs/NEURAL_MODEL.md](docs/NEURAL_MODEL.md). Additional training dependencies
in `requirements-training.txt` are offline only.

## Submission

`PARTICIPANT` in `agent.py` is set to `jr007`. Retain `agent.py` and `requirements.txt` at repository root.
With Docker running, execute the organizers' build check:

```sh
./tools/check_build.sh
```

In this managed cloud, the proxy CA must also be trusted inside Docker builds.
Use `./tools/check_build_cloud.sh` here; it supplies the existing CA as a temporary
BuildKit secret, keeps TLS verification enabled, and runs the same restricted
smoke check. The organizer Dockerfiles remain unchanged. Its generated build
recipes match those used for the final verified image.

Commit and push all required runtime files, including any model data, to a public
repository. Submit that repository URL and full 40-character commit SHA through
the event's submission form. Register your team separately as required by the
participant guide. Event registration and competition submission are separate steps; this project does not submit itself.

The bundled `game.py`, `evaluator.py`, judge, baselines, tests, Docker recipes and
server plumbing come from event bundle revision
`c83ab6eeb7660a2fd1fae7df224315d41fe985a9`; rule/referee files are unchanged.
