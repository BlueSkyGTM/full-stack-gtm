## Ship It

To ship this in a CI pipeline — where eval results from multiple tasks need to produce a single model leaderboard for a PR comment — you need the aggregator to accept `EvalRun` records and output both a machine-readable JSON and a human-readable markdown table:

```python
import json
from dataclasses import asdict
from datetime import datetime, timezone

@dataclass
class EvalRun:
    model_id: str
    task_id: str
    metric_name: str
    score: float
    category: str


def evalruns_to_leaderboards(runs: list[EvalRun]) -> list[Leaderboard]:
    task_groups: dict[str, list[EvalRun]] = {}
    for run in runs:
        task_groups.setdefault(run.task_id, []).append(run)

    leaderboards = []
    for task_id, task_runs in sorted(task_groups.items()):
        entries = [
            LeaderboardEntry(r.model_id, r.score)
            for r in sorted(task_runs, key=lambda x: x.score, reverse=True)
        ]
        leaderboards.append(Leaderboard(name=task_id, entries=entries))
    return leaderboards


def build_leaderboard_report(
    runs: list[EvalRun],
    output_format: str = "both",
) -> dict:
    leaderboards = evalruns_to_leaderboards(runs)
    models = all_models(leaderboards)

    borda = borda_count(leaderboards)
    copeland = copeland_method(leaderboards)
    kemeny_perm, kemeny_score = kemeny_young_approx(leaderboards)

    borda_ranks = rank_dict_from_scores(borda)
    copeland_ranks = rank_dict_from_scores(copeland)
    kemeny_ranks = {m: i + 1 for i, m in enumerate(kemeny_perm)}

    rows = []
    for model in sorted(models, key=lambda m: borda_ranks[m]):
        coverage = sum(1 for lb in leaderboards if model in lb.models())
        task_scores = {}
        for lb in leaderboards:
            entry = lb.get(model)
            if entry:
                task_scores[lb.name] = round(entry.score, 4)

        rows.append({
            "model_id": model,
            "borda_rank": borda_ranks[model],
            "copeland_rank": copeland_ranks[model],
            "kemeny_rank": kemeny_ranks[model],
            "borda_score": round(borda[model], 1),
            "copeland_score": round(copeland[model], 1),
            "tasks_completed": f"{coverage}/{len(leaderboards)}",
            "per_task_scores": task_scores,
        })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_tasks": len(leaderboards),
        "num_models": len(models),
        "kemeny_consensus": " > ".join(kemeny_perm),
        "kemeny_agreement_score": kemeny_score,
        "leaderboard": rows,
    }

    if output_format in ("json", "both"):
        print("=== JSON Report ===")
        print(json.dumps(report, indent=2))

    if output_format in ("markdown", "both"):
        print("\n=== Markdown Table ===\n")
        header = "| Model | Borda | Copeland | K-Y | Coverage |"
        sep = "|-------|-------|----------|-----|----------|"
        print(header)
        print(sep)
        for row in rows:
            print(
                f"| {row['model_id']} "
                f"| {row['borda_rank']} "
                f"| {row['copeland_rank']} "
                f"| {row['kemeny_rank']} "
                f"| {row['tasks_completed']} |"
            )

        disagreements = [
            r["model_id"] for r in rows
            if max(r["borda_rank"], r["copeland_rank"], r["kemeny_rank"])
            - min(r["borda_rank"], r["copeland_rank"], r["kemeny_rank"]) >= 2
        ]
        if disagreements:
            print(f"\n⚠ Methods disagree (spread ≥ 2) on: {', '.join(disagreements)}")

    return report


sample_runs = [
    EvalRun("model_a", "task_qa", "accuracy", 0.92, "reasoning"),
    EvalRun("model_b", "task_qa", "accuracy", 0.88, "reasoning"),
    EvalRun("model_c", "task_qa", "accuracy", 0.90, "reasoning"),
    EvalRun("model_a", "task_code", "pass@1", 0.78, "coding"),
    EvalRun("model_b", "task_code", "pass@1", 0.85, "coding"),
    EvalRun("model_c", "task_code", "pass@1", 0.72, "coding"),
    EvalRun("model_a", "task_math", "exact_match", 0.65, "reasoning"),
    EvalRun("model_b", "task_math", "exact_match", 0.70, "reasoning"),
    EvalRun("model_c", "task_math", "exact_match", 0.68, "reasoning"),
    EvalRun("model_a", "task_safety", "safety_rate", 0.99, "safety"),
    EvalRun("model_c", "task_safety", "safety_rate", 0.97, "safety"),
]

report = build_leaderboard_report(sample_runs, output_format="both")
```

This is the output format the eval runner in a CI pipeline can paste into a PR comment. The JSON goes to your tracking system; the markdown table goes to the diff review. The disagreement flag at the bottom is the ship/no-ship signal: if Borda and Kemeny-Young disagree on the top model, the consensus is weak and you should investigate before deploying.

The disagreement flag is the same mechanism you'd use in a Clay waterfall (Zone 2): when enrichment providers disagree on an account's priority, the spread between aggregation methods tells you whether the ranking is stable or fragile. A wide spread means the ranking depends on which aggregation properties you chose — positional scoring vs pairwise majority — and that choice should be explicit, not accidental. [CITATION NEEDED — concept: multi-signal disagreement detection in Clay enrichment workflows]