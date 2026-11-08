# Public event-compatible bot research

Six public forks of the event bundle were inspected at fixed commits. These
are actual independently published event agents, not policies invented for
our synthetic benchmark. Their `game.py` and `evaluator.py` bytes match event
revision `c83ab6eeb7660a2fd1fae7df224315d41fe985a9`, including the 36-card deck,
five-card Omaha rules and unusual hand ranking. Compatibility does not imply
that their strength claims have been independently established.

| Public repository | Pinned commit | Findings |
| --- | --- | --- |
| [ArmaanMediratta/heads-up-bundle](https://github.com/ArmaanMediratta/heads-up-bundle) | `183342c61239b9df5db22e6fb2cf426d5e2df758` | Primary agent combines a fitted preflop table, vectorized Monte Carlo, opponent learning and EV-based sizing. A mirror detector is disabled by default. Separate Challenger and Demon agents also exist; the committed Demon asset contains expert parameters, not residual-Q or meta-Q weights. Primary agent included in the benchmark. |
| [Dushyant-Betala/heads-up-bundle](https://github.com/Dushyant-Betala/heads-up-bundle) | `ba64c267aa2e8c442e5e5f824539c400faea7a64` | `byte_me / Blueprint`: CFR abstraction and response tables with public-action opponent adaptation. Included in the benchmark. |
| [malki-aal/heads-up-bundle](https://github.com/malki-aal/heads-up-bundle) | `70ddc9555b7016898d2b9478e2f67590cc61ffaf` | `EquityNet`: compact NumPy neural equity predictor and explicit betting thresholds. Default ML mode included in the benchmark. |
| [AdityaVSM/heads-up-bundle](https://github.com/AdityaVSM/heads-up-bundle) | `22330246ecb2143b9256204b106473bdfdeaa7a1` | Honest uniform-opponent Monte Carlo with 48/64/96 samples and street-dependent call/raise margins. Inspected but not selected for the initial three-profile public suite. |
| [PayneAM/heads-up-bundle](https://github.com/PayneAM/heads-up-bundle) | `16d7fa032ccd2f616528d41d007b86a9de1414f9` | Vectorized Monte Carlo, observed fold/raise adaptation and a real-clock sample budget. Inspected but not selected for the initial three-profile suite. |
| [Aniket-tamu/heads-up-bundle](https://github.com/Aniket-tamu/heads-up-bundle) | `d6c8be0033c61ef060cc90c864ef3f7247fe9897` | Committed `plo5bot/bot.py` is a screenshot fragment referencing an undefined `Bot` and missing `exploit`/`strat_tuned` modules. Several required data arrays are absent. The committed agent cannot be evaluated as a complete submission; no result is claimed. |

## Source review and private benchmark bundles

The selected agents' callback dependencies were reviewed before execution.
No runtime outgoing network client, shell execution, unsafe pickle load, or
access to the actual hidden deal seed/opponent cards was found. Their
adaptation uses public betting history and legally revealed showdown cards.
Training tools, development harnesses and the HTTP server plumbing were not
executed. This is a targeted runtime-source review, not a general security
certification of every file in each repository.

No explicit license was located in the six forks or the original bundle.
Public readability is not treated as a permissive redistribution license.
The public submission therefore contains our own adapter/builder, commit
references, file hashes and numerical results; third-party source and weights
remain in private benchmark bundles outside this repository.

`tools/public_bot_opponents.py` builds three bundles from the pinned local
checkouts after checking every selected source/asset against its Git commit
and verifying that the shared official rule files match. Each bundle contains
`opponent.py`, an explicit provenance manifest and the minimal upstream runtime
files. The overnight runner copies these bundles immutably and fingerprints
every file before tuning. The runtime adapter:

* calls the actual upstream `new_half`, `choose_action` and `on_hand_end` hooks;
* isolates upstream imports, preventing a generic `strategy` or `bot` import
  from resolving to our production modules;
* shares the identical official observation/action API and strips server routes;
* fixes existing RNG seeds to `640193` without changing decision thresholds;
* escalates internally swallowed exceptions so a broken fallback cannot pass
  silently as a healthy benchmark opponent;
* unloads private modules after the final hand to avoid retaining model arrays
  across many search cases.

The original wall-time budgets remain active. Armaan's sample counts can vary
with CPU load even with a fixed RNG, so exact action replay is not promised.
Dushyant's pace projection assumes 1,000 hands; short screens can therefore
thin computation differently from a full event half. These direct-callback
matches impose no HTTP or official event time-bank enforcement. They measure
strategic challenge under the stated conditions, not certified deployment.

## Initial development characterization

The frozen neural bot played 100 duplicate deals against each selected public
agent on seed `public-probe-dev-v1`, with fresh learning for each half. This
is **600 actual hands**. All chosen actions were legal and no upstream internal
error counter or fallback exception was observed.

| Public opponent | Our bot bb / duplicate deal | Approximate card-paired 95% interval |
| --- | ---: | ---: |
| Armaan primary | −1.380 | [−3.501, +0.741] |
| Dushyant Blueprint | −1.900 | [−3.864, +0.064] |
| Malki EquityNet | +2.710 | [+0.376, +5.044] |

The first two results show why these are useful challenges beyond the earlier
synthetic pool. The small development sample and serial adaptation prevent a
reliable ranking. These cards are development data; they are not protected
final validation, and these results do not establish that the current bot beats
most online bots. Evidence, exact file hashes and full paired outcomes are in
`results/public-bot-development.json`. The production runtime manifest remained
unchanged throughout these matches.
