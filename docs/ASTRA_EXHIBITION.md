# Bot versus direct Astra decisions

Historical adaptive revision: `97883009569ad2d383b8145e46ee642262390049`.
This 40-hand pilot preceded the neural revision and its separate fresh test.

This is an untimed exhibition against actual `gpt-6-astra` model agents,
not another scripted poker bot. Two fresh isolated contexts played
opposite seats on 20 duplicate deals (40 actual hands). Each model
selected every action itself through a private observation endpoint.
No automated betting policy or Monte Carlo equity tool was supplied to
Astra. A current-made-hand category helper used visible cards only.

**Bot: -34 chips. Astra: +34 chips.**
Bot average: -0.850 bb per duplicate deal.
Approximate small-sample 95% interval: [-2.7485607291064627, 1.048560729106463].

The tournament time bank was deliberately not enforced: language-model
decision latency is incompatible with 8 seconds per 1,000 hands. This
small strategic exhibition cannot establish a reliable ranking or
tournament win probability. Do not conflate the chip lead with strong
statistical evidence of superiority.

Astra made 120 decisions. Invalid submitted actions: 0.
The replay audit reproduced every bot action from the unchanged policy,
verified every Astra observation against the official game state,
validated legal actions and zero-sum payouts, and checked that the
runtime source hashes remained unchanged. Neither half could access
the other half's endpoint; full trace files were saved only after
both halves finished. The private deal seed was not exposed.

Evidence: `results/astra-exhibition.json` and
`results/astra-exhibition-summary.json`. No bot training or tuning
was performed during this experiment.
