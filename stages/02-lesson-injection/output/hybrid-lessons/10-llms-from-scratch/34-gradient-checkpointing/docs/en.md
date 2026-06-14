# Gradient Checkpointing and Activation Recomputation

## Learning Objectives

1. Compare memory consumption between standard backpropagation and gradient checkpointing strategies using profiling output.
2. Implement gradient checkpointing in a PyTorch training loop using `torch.utils.checkpoint`.
3. Configure checkpoint segment boundaries to trade compute overhead for memory savings.
4. Measure the wall-clock and peak-memory impact of recomputation on a multi-layer model.
5. Diagnose whether gradient checkpointing