# Direct Astra exhibition: incomplete

The frozen neural candidate was scheduled to play 200 hands against actual
`gpt-6-astra` decisions: two 50-duplicate-deal matches with four fresh isolated
model contexts. Every Astra decision was chosen by the model from its own
legal observation; no automated betting policy or equity simulator was given
to those players. The candidate source, weights and parameters stayed frozen.

Two Astra turns stopped at the account's model usage limit. At that point
**189 of the planned 200 hands had completed**. No scripted replacement,
different model or automatic fold finished the missing hands. Consequently
there is **no completed 200-hand result and no claim that the bot beat Astra**.

| Match / context | Completed hands | Bot net chips | Astra net chips |
| --- | ---: | ---: | ---: |
| A / first context | 50 | +246 | −246 |
| A / second context | 49 | −528 | +528 |
| B / first context | 50 | −142 | +142 |
| B / second context | 40 | +102 | −102 |

The bot was provisionally behind by 322 chips across those 189 completed
hands. The seat-swapped halves have unequal lengths, and the unfinished
hand outcomes are unknown. This is not a valid complete duplicate-match
estimate or a reliable model ranking. The full traces were deliberately kept
in referee memory until both halves finished, so this partial checkpoint also
cannot receive the complete replay audit. The live referee state is preserved
without further actions; aggregate counts are recorded in
`results/astra-neural/incomplete-status.json`.

The predefined plan and source hashes are in `results/astra-neural/plan.json`.
All 680 recorded Astra requests were accepted without an invalid-action error.
The exhibition did not impose the event's thinking bank: actual language-model
latency is incompatible with that deployment allowance. Separately verified
CPU deployment and synthetic-opponent results are documented in
[NEURAL_REVISION.md](NEURAL_REVISION.md).

The earlier 40-hand exhibition in [ASTRA_EXHIBITION.md](ASTRA_EXHIBITION.md)
used the previous submitted bot and different deals. It must not be pooled
with this partial run or treated as a paired test of the upgrade.
