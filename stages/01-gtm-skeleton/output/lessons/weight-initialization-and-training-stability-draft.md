# Weight Initialization and Training Stability

## 1. Hook

Show a neural network failing to train — not because the architecture or data is wrong, but because weights started in the wrong range. Display loss curves for identical networks where only initialization differs: one converges, one plateaus, one explodes to NaN.

## 2. Concept

Explain variance propagation: why each layer's output variance depends on the input variance times the number of inputs times the weight variance. Derive the Xavier/Glorot condition (variance = 1/fan_in for forward pass stability, or 2/(fan_in + fan_out) as compromise) and the He condition (variance = 2/fan_in for ReLU networks, which kill half their signal). State the mechanism: initialization controls the signal-to-noise ratio at the start of training, and gradients are just signals flowing backward through the same weights.

## 3. Demonstration

Build a 5-layer network from scratch in NumPy. Initialize three ways — all zeros, large random normal (std=1), and He (std=sqrt(2/fan_in)) — and track activation statistics (mean, variance, fraction zero) across layers after one forward pass. Print the statistics in a table. Then train a small PyTorch model on synthetic data with all three initializations and print final loss after N epochs to show convergence differences.

## 4. Use It

Foundational for Zone 1 (ML Fundamentals). The direct GTM connection: when fine-tuning transformer models for intent classification or lead scoring, initialization of the classifier head (random) versus the backbone (pretrained) determines whether the head destabilizes early training. This is why learning rate warmup exists — it is an initialization-adjacent stability mechanism. [CITATION NEEDED — concept: fine-tuning classifier heads in GTM pipelines]

**Exercise hook (easy):** Given a 3-layer network with ReLU activations and layer widths [512, 256, 128], compute the correct He initialization standard deviation for each layer. Print the values.

## 5. Ship It

Cover the practical checks: (1) log activation statistics at init time, (2) monitor gradient norms during the first 100 steps, (3) set a gradient norm ceiling as a safety net. Show PyTorch's `nn.init` module and where to hook a diagnostic that prints "WARNING: layer X activation variance = {v}" if variance drops below 1e-6 or exceeds 10. Discuss when to use pretrained initialization vs. custom — pretrained weights are already initialized, so this matters most for new heads and from-scratch training.

**Exercise hook (medium):** Add a gradient norm monitor to a small training loop. Detect and print which layer first causes gradient norm to exceed 100. Confirm the fix (He init) brings all gradient norms into a stable range.

## 6. Stretch

Explore why residual connections relax initialization requirements (signal has a bypass path), why LayerNorm makes training less sensitive to initialization (re-centering and re-scaling per layer), and why transformer architectures can train at all despite extreme depth. Implement a 20-layer network with and without residual connections, print the fraction of dead neurons at initialization for each.

**Exercise hook (hard):** Build a 10-layer ReLU network. Sweep initialization standard deviation from 0.001 to 3.0. Plot training loss at epoch 10 versus init std. Identify the critical range where convergence occurs and explain why the boundaries exist using the variance propagation formula.

---

## Learning Objectives

1. Diagnose vanishing and exploding gradients caused by poor weight initialization.
2. Implement Xavier/Glorot and He initialization from first principles using fan-in calculations.
3. Compare activation statistics across layers for different initialization strategies.
4. Configure the correct initialization method given an activation function.
5. Evaluate training stability by monitoring gradient norms during the first 100 optimization steps.

## GTM Redirect

Foundational for Zone 1. Direct application: initializing classifier heads when fine-tuning models for intent classification, lead scoring, or outreach prioritization. The mechanism (variance propagation) explains why a randomly-initialized head on a pretrained transformer can destabilize early training — and why warmup schedules and small head learning rates compensate.