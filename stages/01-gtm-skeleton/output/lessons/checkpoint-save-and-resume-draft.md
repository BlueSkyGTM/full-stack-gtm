# Checkpoint Save and Resume

## Hook
Training runs die—GPU OOM, spot preemption, power loss. Without checkpointing, you lose hours of compute and the model state. This beat establishes why atomic state capture is non-negotiable for any training loop that runs longer than a few minutes.

## Concept
A checkpoint is a serialized snapshot of every state variable required to continue training: model weights, optimizer momentum buffers, learning rate scheduler position, epoch counter, and RNG seeds. The mechanism is dictionary serialization—you construct a dict with all necessary state, write it to disk atomically, and load it back to reconstruct the training context. Covers what happens when you omit optimizer state (learning rate resets, momentum is lost) versus when you capture everything. Introduces `torch.save`/`torch.load` with `weights_only=True` and the Safetensors format as a solution to pickle deserialization risks.

## Code
Build a minimal training loop that checkpoints every N steps to disk. Kill the loop mid-run, then run a resume function that loads the checkpoint, restores model + optimizer + scheduler + epoch, and continues training. Print loss values before and after resume to confirm continuity. All in a single self-contained script using a toy model and synthetic data.

## Use It
In GTM, the same save-and-resume pattern applies to long-running enrichment jobs—scoring thousands of accounts through a Clay waterfall, or running batch inference across a prospect list. If the job fails at row 4,000, you restart from row 4,000, not row 1. The redirect: this is the mechanism behind reliable batch processing in **Zone 2 (Enrichment & Scoring)**. Any multi-step pipeline that transforms data at scale needs checkpoint semantics.

Exercise hooks:
- (Easy) Modify the checkpoint interval and verify it saves at the correct step.
- (Medium) Delete one key from the checkpoint dict (optimizer state), load it, and observe training behavior diverge.
- (Hard) Implement a `best_model` checkpoint that only saves when validation loss improves, alongside the periodic checkpoint.

## Ship It
Production checkpoint patterns: save to cloud storage (S3/GCS) instead of local disk, implement checkpoint pruning to keep only the last N checkpoints, add validation-metric-triggered saves for best-model tracking, and handle the race condition where a save is interrupted. The practitioner ships a training script that survives preemption and resumes exactly where it left off, with disk usage bounded by a retention policy.

Exercise hooks:
- (Easy) Add a cleanup function that deletes checkpoints older than the last 3.
- (Medium) Modify the save path to write to a mounted cloud storage path instead of local disk.
- (Hard) Implement checksum verification on load—detect a corrupted checkpoint file and fall back to the previous one.

## Push It
Distributed checkpointing across multiple GPUs (DTensor, FSDP), memory-mapped loading for large models to avoid loading the full checkpoint into RAM, and quantized checkpoint formats for reduced storage. For practitioners not training foundation models: the same pattern generalizes to any stateful long-running process—agent workflows, multi-step enrichment pipelines, sequential API calls with rate limits.

Exercise hooks:
- (Easy) Measure the file size difference between a full checkpoint and a weights-only save.
- (Medium) Implement async checkpoint saving using a background thread so training doesn't pause during serialization.
- (Hard) Shard a checkpoint across multiple files and implement a loader that reconstructs the full state from shards.

---

**GTM Redirect:** Zone 2 (Enrichment & Scoring) — the save/resume mechanism is what makes batch enrichment fault-tolerant. Without it, any failure in a Clay waterfall or scoring pipeline means restarting from scratch. [CITATION NEEDED — concept: checkpoint recovery in enrichment pipelines]