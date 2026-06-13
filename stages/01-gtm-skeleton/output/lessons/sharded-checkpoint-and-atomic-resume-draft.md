# Sharded Checkpoint and Atomic Resume

## Hook
When training across 8 GPUs with ZeRO-3, a checkpoint is N files written simultaneously. If rank 3 crashes mid-write, you have a corrupt, unrecoverable checkpoint. This lesson covers the pattern that prevents that.

## Concept
Two mechanisms compose here. **Sharded state dict**: each rank persists only its slice of model/optimizer state—no gathering to rank 0, no single-node memory bottleneck. **Atomic resume**: writes go to a temporary directory, then a rename swaps the directory into the canonical path. A reader sees either the complete shard set or the previous checkpoint—never a partial write. Together they give you crash-safe checkpointing at scale.

## Demo
A single-machine simulation using `torch.distributed.checkpoint` to save a sharded state dict, intentionally corrupt the temp directory mid-write, and demonstrate that atomic semantics prevent loading the partial state. Observable output confirms the guard works.

## Use It
Foundational for Zone 2 — infrastructure and scaling. The atomic write pattern (temp → rename) also appears in GTM data pipelines: writing scored account lists or enrichment outputs to shared storage. If Clay's waterfall writes partial results to a sheet, downstream teams acting on incomplete data burn pipeline hours. The same temp-then-swap discipline applies.

## Ship It
- **Easy**: Write a single-process checkpoint using temp-file-then-rename. Print confirmation that no partial file is visible at the target path.
- **Medium**: Shard a toy model's state dict across simulated ranks, save, kill mid-write, verify the loader rejects the incomplete set.
- **Hard**: Build a checkpoint manager class that tracks global step, handles atomic resume, and detects stale checkpoints from a previous run. Include a full integration test.

## References
- PyTorch `torch.distributed.checkpoint` API documentation
- [CITATION NEEDED — concept: ZeRO paper's state partitioning strategy across ranks]
- [CITATION NEEDED — concept: atomic directory rename semantics across POSIX and cloud object stores]