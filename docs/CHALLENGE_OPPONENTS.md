# Fixed additional synthetic stress opponents

`tools/challenge_opponents.py` supplies six independent implementations for
overnight candidate comparisons. These are synthetic policies, not other
teams' entries, actual ChatGPT Astra decisions, or solved poker strategies.
The source and fixed configuration are fingerprinted in
`results/challenge-suite-manifest.json` before optimization uses them.

| Profile | Samples per visible board | Purpose |
| --- | ---: | --- |
| `challenge_value` | 384 | Range-aware value bets, conservative large bets, occasional traps |
| `challenge_polarized` | 256 | Strong value and weak blocker-preferred bluffs using the same sizes |
| `challenge_checkraise` | 256 | Frequent strong OOP checks followed by value and occasional semibluff raises |
| `challenge_adaptive` | 192 | Learn public responses to legal bet sizes and pressure demonstrated excess folds |
| `challenge_station` | 128 | Broad calls and selective value bets; penalize unsupported bluff assumptions |
| `challenge_shifting` | 128–384 | Value → polarized → station → checkraise, by public hand index |

The shifting policy uses quarters of the configured half, with a minimum
20-hand phase. The other five configurations remain fixed. Each half starts
with the same profile-specific pseudorandom stream, derived with a stable hash
from the literal profile and runner-provided seed. Neither the deal seed nor
any actual opponent/future cards enter that stream.

Each policy sees only its own `Observation` and public hand-result fields.
It shares the project's exact legal Monte Carlo sampler, but imports no
production strategy, opponent model, fitted preflop lookup or neural model.
Candidate opponent holdings are ranked using a private-card texture before
the flop and current made-hand plus legal draw texture afterward. Historical
actions use only their historical visible board prefix. Betting-range
likelihoods never read sampled showdown outcomes or sampled future cards.
Showdown outcomes are used only after weighting, to estimate current equity.

Opponent action frequencies and replies to half/pot bets decay over recent
hands. Terminal calls and folds are accounted for exactly once from public
winner/showdown flags. No revealed opponent holding is needed. Value decisions
compare both actual legal wager costs; polarized mixtures deliberately retain
some bluffs even when their isolated estimated EV is negative.

The suite is heuristic. Draw descriptors are not calibrated equity models,
range likelihoods can be wrong, and performance against these probes cannot
establish a tournament win probability. Direct evaluation is untimed unless a
separate official-judge run enforces the bank. A search can overfit a fixed
suite; candidate selection therefore still requires fresh seeds, a frozen
incumbent comparison and broad holdout opponents.

The public interface is:

```python
from tools.challenge_opponents import ChallengeOpponent, PROFILES

opponent = ChallengeOpponent("challenge_value", seed=830907)
opponent.new_half(config, memory)
action = opponent.choose_action(observation, memory)
opponent.on_hand_end(public_result, memory)
```

For immutable evaluation, the runner can capture the entire source and append
a literal instance with these three callbacks. Capturing only an importing
wrapper would leave a mutable transitive dependency. Fourteen focused tests
cover actual legal games, reproducible half resets, equal decisions for equal
observations with different actual hidden deals, public terminal-reply
accounting, and fixed style-shift boundaries.
