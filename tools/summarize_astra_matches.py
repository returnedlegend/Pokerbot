"""Audit and summarize the preregistered fixed-count direct-Astra exhibition."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_astra_exhibition import ROOT, audit


def summarize(plan_path: Path, paths: list[Path], output: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    assert len(paths) == plan["matches"] == 2, "Complete the planned two matches"
    assert len(set(path.resolve() for path in paths)) == len(paths)
    reports = [json.loads(path.read_text()) for path in paths]
    audits = [audit(path) for path in paths]
    pairs = []
    match_means = []
    sources = reports[0]["source_sha256"]
    for report in reports:
        assert report["astra_model"] == "gpt-6-astra"
        assert report["deals_per_half"] == plan["deals_per_half"]
        assert report["source_sha256"] == sources
        assert report["policy_parameters"] == reports[0]["policy_parameters"]
        assert not report["time_bank_enforced"]
        pairs.extend(report["paired_bb_bot"])
        match_means.append(statistics.mean(report["paired_bb_bot"]))
    for name, digest in plan["source_sha256"].items():
        assert sources[name] == digest
    for name, digest in sources.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    hands = sum(report["total_hands"] for report in reports)
    assert hands == plan["planned_actual_hands"]
    assert len(pairs) == plan["independent_deal_pairs"]
    mean = statistics.mean(pairs)
    se = statistics.stdev(pairs) / math.sqrt(len(pairs))
    # Two independent match trajectories; t(.975, df=1), intentionally wide.
    match_se = statistics.stdev(match_means) / math.sqrt(len(match_means))
    match_ci = [mean - 12.7062047364 * match_se, mean + 12.7062047364 * match_se]
    chips = sum(item["bot_net_chips"] for item in audits)
    assert math.isclose(chips, sum(pairs) * 2)
    summary = {
        "kind": "fixed_count_direct_Astra_exhibition_summary",
        "opponent_model": "gpt-6-astra",
        "plan": str(plan_path.relative_to(ROOT)),
        "source_sha256": sources,
        "matches": [dict(result=str(path.relative_to(ROOT)), **item)
                    for path, item in zip(paths, audits)],
        "actual_hands": hands,
        "duplicate_deals": len(pairs),
        "fresh_model_contexts": plan["opponent_contexts"],
        "bot_net_chips": chips,
        "astra_net_chips": -chips,
        "bot_bb_per_duplicate_deal": mean,
        "approximate_card_paired_ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "whole_match_t_ci95_two_matches": match_ci,
        "match_means_bb_per_duplicate_deal": match_means,
        "astra_decisions": sum(item["astra_decisions"] for item in audits),
        "submitted_invalid_actions": sum(item["submitted_invalid_actions"] for item in audits),
        "every_match_replay_audited": all(item["replay_audit_passed"] for item in audits),
        "runtime_source_unchanged": True,
        "tuning_after_start": False,
        "time_bank_enforced": False,
        "limitations": [
            "Only four isolated model contexts and two complete match trajectories.",
            "Card-paired normal interval ignores serial learning within each match.",
            "Two-match Student-t interval has just one degree of freedom.",
            "Untimed language-model actions are not a tournament deployment test.",
            "A chip lead cannot establish universal model superiority or tournament odds.",
        ],
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "results/astra-neural/plan.json")
    parser.add_argument("--matches", nargs=2, type=Path, default=[
        ROOT / "results/astra-neural/match-a/astra-exhibition.json",
        ROOT / "results/astra-neural/match-b/astra-exhibition.json",
    ])
    parser.add_argument("--output", type=Path, default=ROOT / "results/astra-neural/summary.json")
    args = parser.parse_args()
    result = summarize(args.plan.resolve(), [path.resolve() for path in args.matches], args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("matches", "source_sha256")}, indent=2))
