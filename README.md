# ShortStack — heads-up short-deck PLO5 bot

A CPU-only bot for the TAMU Datathon Heads Up challenge. `agent.py` retains the
organizers' HTTP protocol and delegates decisions to `strategy.py`.

The model adapts **PyPokerEngine's open source Monte Carlo equity policy** to
this game's exact rules. It does not use pretrained Hold'em weights: those
weights assume the wrong deck, private-card count and hand rankings. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source and MIT attribution.

## Run

Requires Python 3.12 (tested on Linux amd64):

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python agent.py
```

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
- Weight plausible opponent hands by the public action history. Frequent
  indiscriminate raises receive less credit for strength than rare raises.
- Compare calls with pot odds; estimate half-pot and pot-raise value against a
  continuation range. Smooth observed folds and aggression across each half.
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

Development tuning seeds and validation seeds are separate. Synthetic opponents
cannot establish performance against unknown tournament entries. Detailed
measured results and limitations are in [EVALUATION.md](EVALUATION.md).

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
