# Open source model provenance

The bot's equity model adapts the Monte Carlo equity approach used by
**PyPokerEngine HonestPlayer**, by ishikota. Upstream source:

- Repository: https://github.com/ishikota/PyPokerEngine
- Reference revision: `a52a048a15da276005eca4acae96fb6eeb4dc034`
- Policy: `examples/players/honest_player.py`
- Equity model: `pypokerengine/utils/card_utils.py`
- License: `LICENSE.md` (MIT), reproduced below.

This implementation replaces the original 52-card Texas Hold'em evaluator and
two-card opponent hands with the event's exact 36-card, five-card Omaha rules.
It treats ties as half a win, uses compiled sampling, and adds range-aware
decision making. It does **not** use pretrained neural-network weights or claim
that a Hold'em policy is valid for this event. Parameters are tuned locally.

The event's `evaluator.py`, `game.py`, web-server template, judge and baseline
bots originate from https://github.com/tamu-datathon-org/heads-up-bundle.

## Event-specific neural weights and research

`equity_network.npz` contains newly trained weights produced by this project
with the open-source scikit-learn `MLPRegressor` (BSD-3-Clause):
https://github.com/scikit-learn/scikit-learn . No scikit-learn package code is
vendored, and the tournament runtime does not import that training dependency.
The model's simulation labels use the event's exact rules. See
`docs/NEURAL_MODEL.md` for training and validation provenance.

The optional, disabled river abstraction is an independent implementation of
CFR+. OpenSpiel's open-source CFR implementations and RLCard's model-zoo
documentation informed the research; source links and applicability limits are
in `docs/STRENGTH_RESEARCH.md`. No pretrained betting policy from those projects
is represented as a solution to this event.

## PyPokerEngine MIT license

MIT License

Copyright (c) 2016 ishikota

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
