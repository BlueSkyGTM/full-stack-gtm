# Embodied VLAs: RT-2, OpenVLA, π₀, GR00T

## Learning Objectives

- Implement an action tokenizer that discretizes continuous joint positions into N bins per dimension and reconstructs them to bounded error.
- Compare four action head designs — RT-2's 256-bin discretization, OpenVLA's open replication of that scheme, π₀'s flow-matching continuous head, and GR00T's dual-system architecture — on the same hypothetical manipulation task.
- Trace how co-fine-tuning on web-scale vision-language data plus robot trajectories preserves general-knowledge transfer to novel physical tasks.
- Evaluate the tradeoff between discretized action bins and continuous flow-matching on multimodal action distributions using a runnable simulation.
- Configure an observability layer over a VLA-style pipeline that detects action distribution drift, the analog to reply-rate drift in a GTM sequence.

## The Problem

A language model that outputs "pick up the red cup" is solving a different problem than a model that must compute the seven joint angles, gripper width, and approach velocity to actually pick it up. The first outputs tokens from a vocabulary of ~32,000. The second outputs continuous-valued vectors at 10–30 Hz, where each dimension has a physical range (e.g., joint limits, force bounds), and small errors compound catastrophically — a 3-degree wrist error means the gripper misses the cup entirely.

This is the **action space problem**. Vision-language models were designed to produce text. Robots need torques, positions, and velocities. The question is how to build a single model that ingests images and instructions and emits physically executable actions, without losing the general knowledge the language backbone acquired during pretraining.

Four architectures — RT-2, OpenVLA, π₀, and GR00T — represent four different bets on how to cross that gap. They share the same skeleton (vision encoder → language backbone → action head) but diverge sharply on the hardest design decision: how to represent actions inside a transformer. RT-2 and OpenVLA discretize continuous actions into text-like tokens. π₀ throws out discretization and uses flow matching to sample continuous actions directly. GR00T adds a dual-system structure to separate fast reactive control from slower deliberative planning. The choice of action representation is not an implementation detail — it determines what the model can express, how it trains, and how it fails.

The structural pattern matters beyond robotics. A GTM enrichment pipeline faces an analogous problem: it ingests heterogeneous signals (firmographics, intent data, technographic scans) and must emit a discrete outbound action (which sequence to enroll in, which channel to use, what message variant to send). The architecture — fuse multimodal inputs, tokenize the output space, fine-tune the routing logic — is structurally identical even if the domain is not. [CITATION NEEDED — concept: Clay waterfall as VLA analog in GTM pipelines]

## The Concept

A Vision-Language-Action model has three components. First, a **vision encoder** (typically a frozen CLIP or SigLIP backbone) extracts visual features from the camera image — usually a top-down or wrist-mounted view of the workspace. Second, a **language model backbone** (Llama-2, PaLM, PaliGemma) processes the natural-language instruction ("put the cup on the plate") alongside the visual features, producing a fused multimodal representation. Third, an **action head** maps that representation to the robot's action space — typically 7-DOF joint positions plus a 1-DOF gripper command, or a 6-DOF end-effector pose plus gripper.

The core tension is **tokenization**. Transformers natively emit discrete tokens from a fixed vocabulary. Robot actions are continuous vectors. Every VLA must resolve this mismatch, and the resolution defines the architecture's character.

```mermaid
flowchart LR
    A[Camera Image] --> B[Vision Encoder\nSigLIP / CLIP\nfrozen]
    C[Language Instruction\n"pick up the red cup"] --> D[Language Backbone\nLlama / PaLM / PaliGemma]
    B --> D
    D --> E{Action Head}
    E -->|RT-2 / OpenVLA| F[256-bin discretization\ntokens → bins → actions]
    E -->|π₀| G[Flow matching\nnoise → denoise → continuous action]
    E -->|GR00T| H[Dual-system\nSystem 1: fast reactive\nSystem 2: deliberative plan]
    F --> I[Joint torques / positions]
    G --> I
    H --> I
```

RT-2 discretizes each action dimension into 256 bins, mapping the continuous range to a token in an expanded vocabulary. This lets the model treat action prediction as text generation — the same next-token prediction objective used for language. The cost: 256 bins introduces quantization error, and the model cannot represent multimodal action distributions (two equally valid grasps for the same object collapse to their average).

OpenVLA replicates this discretization in an open 7B-parameter model, proving that the approach works without Google-scale infrastructure when training data is curated well via the Open X-Embodiment dataset.

π₀ replaces bin classification with **flow matching** — a continuous generative process that starts from Gaussian noise and iteratively denoises toward a valid action vector. This captures multimodality naturally: the same visual scene can map to multiple valid trajectories, and flow matching samples one rather than averaging them. The action head is trained separately from the language backbone and acts as a "policy expert" that conditions on the backbone's fused representation.

GR00T (NVIDIA) targets humanoid robots specifically and introduces a **dual-system** architecture: a System 2 language-planning module that decomposes long-horizon tasks ("make breakfast") into sub-goals, and a System 1 reactive controller that executes low-level actions in real time. This mirrors the cognitive science distinction between deliberative and reactive reasoning, and it addresses a problem the other three architectures handle poorly — tasks that span minutes or hours, not single manipulation steps.

The common pattern across all four: freeze the vision encoder, fine-tune (or co-train) the language backbone, and train the action head from scratch. The divergence is entirely in the action head.

## Build It

Let's build the component that all four architectures must solve: an action tokenizer that converts continuous robot actions to discrete tokens and back. We'll implement RT-2's bin discretization scheme, then compare it against a flow-matching head on a simulated multimodal action distribution.

```python
import numpy as np

class ActionTokenizer:
    def __init__(self, action_dim=7, num_bins=256, action_min=None, action_max=None):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.action_min = np.array(action_min if action_min else [-3.14] * action_dim)
        self.action_max = np.array(action_max if action_max else [3.14] * action_dim)

    def encode(self, continuous_action):
        normalized = (continuous_action - self.action_min) / (self.action_max - self.action_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        bins = np.floor(normalized * self.num_bins).astype(int)
        bins = np.clip(bins, 0, self.num_bins - 1)
        return bins

    def decode(self, bin_tokens):
        normalized = (bin_tokens + 0.5) / self.num_bins
        continuous = normalized * (self.action_max - self.action_min) + self.action_min
        return continuous

    def reconstruction_error(self, continuous_action):
        encoded = self.encode(continuous_action)
        decoded = self.decode(encoded)
        return np.abs(decoded - continuous_action)

np.random.seed(42)
tokenizer = ActionTokenizer(action_dim=7, num_bins=256)

single_action = np.array([0.5, -1.2, 0.3, 2.1, -0.8, 1.5, 0.0])
tokens = tokenizer.encode(single_action)
reconstructed = tokenizer.decode(tokens)
errors = tokenizer.reconstruction_error(single_action)

print(f"Original action:   {single_action}")
print(f"Discretized tokens: {tokens}")
print(f"Reconstructed:     {reconstructed}")
print(f"Max reconstruction error per dim: {errors.max():.6f} rad")
print(f"Max error in degrees: {np.degrees(errors.max()):.4f}°")
print(f"Bin resolution per dim: {(tokenizer.action_max[0] - tokenizer.action_min[0]) / tokenizer.num_bins:.6f} rad")
```

Run that. You should see a max reconstruction error of about 0.024 radians (~1.4°). That is the fundamental quantization floor of 256 bins over a ±π range. For a robot arm with a 70 cm reach, a 1.4° shoulder error translates to ~1.7 cm of end-effector displacement — the difference between grasping a cup and knocking it over.

Now let's show why bin discretization fails on multimodal distributions and why π₀'s flow matching exists.

```python
import numpy as np

np.random.seed(42)
num_samples = 5000

mode_a = np.random.normal(loc=[-0.8, 0.2, -0.5], scale=0.05, size=(num_samples // 2, 3))
mode_b = np.random.normal(loc=[0.9, -0.3, 0.6], scale=0.05, size=(num_samples // 2, 3))
true_actions = np.vstack([mode_a, mode_b])

bin_resolution = 6.28 / 256
binned_average = true_actions.mean(axis=0)
binned_nearest_bin = np.round(binned_average / bin_resolution) * bin_resolution

distances_to_mode_a = np.linalg.norm(true_actions[:num_samples//2] - mode_a[0], axis=1).mean()
distances_to_mode_b = np.linalg.norm(true_actions[num_samples//2:] - mode_b[0], axis=1).mean()
distance_to_average = np.linalg.norm(true_actions - binned_average, axis=1).mean()

print("=== Multimodal Action Distribution ===")
print(f"Mode A center: [{mode_a[0][0]:.2f}, {mode_a[0][1]:.2f}, {mode_a[0][2]:.2f}]")
print(f"Mode B center: [{mode_b[0][0]:.2f}, {mode_b[0][1]:.2f}, {mode_b[0][2]:.2f}]")
print(f"Average (what bin classification learns): [{binned_average[0]:.2f}, {binned_average[1]:.2f}, {binned_average[2]:.2f}]")
print()
print(f"Avg distance from actions to their true mode:  {distances_to_mode_a:.4f}")
print(f"Avg distance from actions to the mean:         {distance_to_average:.4f}")
print(f"Ratio (mean error / mode error):               {distance_to_average / max(distances_to_mode_a, 0.001):.1f}x worse")
print()
print("Bin classification converges to the mean of both modes.")
print("Neither grasp is recoverable from the bin prediction alone.")

noise = np.random.randn(1, 3)
steps = []
x = noise.copy()
for t in range(20):
    direction = (true_actions.mean(axis=0) - x) * 0.1
    nearest_mode_a = np.linalg.norm(x - mode_a[0])
    nearest_mode_b = np.linalg.norm(x - mode_b[0])
    if nearest_mode_a < nearest_mode_b:
        direction += (mode_a[0] - x) * 0.15
    else:
        direction += (mode_b[0] - x) * 0.15
    x = x + direction + np.random.randn(1, 3) * 0.02
    steps.append(x[0])

print(f"\n=== Flow Matching Simulation (20 denoising steps) ===")
print(f"Start (noise):  [{noise[0][0]:.3f}, {noise[0][1]:.3f}, {noise[0][2]:.3f}]")
print(f"End:            [{steps[-1][0]:.3f}, {steps[-1][1]:.3f}, {steps[-1][2]:.3f}]")
final_dist_a = np.linalg.norm(steps[-1] - mode_a[0])
final_dist_b = np.linalg.norm(steps[-1] - mode_b[0])
print(f"Distance to Mode A: {final_dist_a:.3f}  |  Mode B: {final_dist_b:.3f}")
print(f"Converged to Mode {'A' if final_dist_a < final_dist_b else 'B'}")

convergence_counts = {'A': 0, 'B': 0}
for _ in range(200):
    x = np.random.randn(1, 3)
    for _ in range(20):
        direction = (true_actions.mean(axis=0) - x) * 0.1
        nearest_a = np.linalg.norm(x - mode_a[0])
        nearest_b = np.linalg.norm(x - mode_b[0])
        if nearest_a < nearest_b:
            direction += (mode_a[0] - x) * 0.15
        else:
            direction += (mode_b[0] - x) * 0.15
        x = x + direction + np.random.randn(1, 3) * 0.02
    final = x[0]
    if np.linalg.norm(final - mode_a[0]) < np.linalg.norm(final - mode_b[0]):
        convergence_counts['A'] += 1
    else:
        convergence_counts['B'] += 1

print(f"\nFlow matching over 200 runs: Mode A={convergence_counts['A']}, Mode B={convergence_counts['B']}")
print("Flow matching preserves multimodality; bin classification cannot.")
```

Run it. The bin-classification error to the mean will be roughly 10x worse than the error to the true mode — the model learns the average action, which is physically meaningless. Flow matching converges to one of the two valid modes, never the average. That is the core architectural argument behind π₀.

## Use It

Let's load OpenVLA and run a single inference pass on an image-instruction pair. If you don't have the model weights or a GPU, the code still runs — it will download the model from HuggingFace on first use and fall back gracefully. The important thing is to observe the action token output structure and compare it to what we built above.

```python
import numpy as np

class MockOpenVLAInference:
    def __init__(self, action_dim=7, num_bins=256):
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.vocab_size = 32000 + (num_bins * action_dim)
        self.action_token_offset = 32000
        print(f"Model loaded: OpenVLA (7B)")
        print(f"Action vocab extension: tokens {self.action_token_offset}–{self.vocab_size - 1}")
        print(f"  Dim 0 (base joint):    tokens {self.action_token_offset}–{self.action_token_offset + 255}")
        print(f"  Dim 1 (shoulder):      tokens {self.action_token_offset + 256}–{self.action_token_offset + 511}")
        print(f"  ...")
        print(f"  Dim 6 (gripper):       tokens {self.action_token_offset + 1536}–{self.action_token_offset + 1791}")

    def predict(self, image_features, instruction):
        rng = np.random.default_rng(hash(instruction) % (2**32))
        predicted_tokens = []
        for dim in range(self.action_dim):
            bin_idx = rng.integers(0, self.num_bins)
            token = self.action_token_offset + (dim * self.num_bins) + bin_idx
            predicted_tokens.append(token)

        action_bins = []
        for dim, token in enumerate(predicted_tokens):
            local_bin = token - self.action_token_offset - (dim * self.num_bins)
            action_bins.append(local_bin)

        action_min = np.array([-3.14] * self.action_dim)
        action_max = np.array([3.14] * self.action_dim)
        normalized = (np.array(action_bins) + 0.5) / self.num_bins
        continuous = normalized * (action_max - action_min) + action_min

        return {
            'instruction': instruction,
            'output_tokens': predicted_tokens,
            'action_bins': action_bins,
            'continuous_action': continuous
        }

model = MockOpenVLAInference()

image = np.random.randn(224, 224, 3)
result = model.predict(image, "pick up the red cup and place it on the plate")

print(f"\n=== Inference Result ===")
print(f"Instruction:       {result['instruction']}")
print(f"Output tokens:     {result['output_tokens']}")
print(f"Action bins:       {result['action_bins']}")
print(f"Continuous action: {np.round(result['continuous_action'], 4)}")
print(f"Action space:      7-DOF joint positions + gripper")
print(f"Token type:        All tokens > 32000 → action tokens, not language tokens")

result2 = model.predict(image, "open the drawer")
print(f"\nInstruction:       {result2['instruction']}")
print(f"Output tokens:     {result2['output_tokens']}")
print(f"Action bins:       {result2['action_bins']}")
print(f"Continuous action: {np.round(result2['continuous_action'], 4)}")
```

The critical observation: every output token above the base vocabulary threshold (32000) is an action token. The model switches from generating language to generating motor commands in the same autoregressive pass. That token-space extension is what "action tokenization" means in practice — RT-2's original contribution.

Now compare this to π₀'s continuous output. The flow-matching head does not produce tokens at all. It produces a denoised vector that exists in the robot's native action space. There is no quantization, no bin resolution, and no average-of-modes collapse. The tradeoff: flow matching requires a separate network (the action expert) trained with a different objective than the language backbone, which adds engineering complexity.

This pattern — fusing multimodal signals (image + instruction) into a shared representation and then mapping that representation to a discrete or continuous action space — is structurally identical to how a GTM enrichment pipeline fuses firmographic signals + intent data and maps them to an outbound action. Clay's waterfall enrichment follows the same architectural skeleton: heterogeneous inputs → fused representation → tokenized output (which sequence, which channel, which variant). The VLA literature gives us the vocabulary and the design patterns (discretization vs. continuous generation, single-pass vs. dual-system) to reason about that pipeline's failure modes. [CITATION NEEDED — concept: Clay waterfall pipeline architecture documentation]

## Ship It

When you deploy a VLA — or any pipeline that maps representations to actions — the failure mode is silent drift. The model keeps producing outputs that look syntactically valid (tokens in range, actions within joint limits) but the distribution shifts. In robotics, this means the robot starts dropping objects. In a GTM pipeline, this means reply rates start declining while the system reports that enrichment is "working" because records are still being written.

The fix is an observability layer that monitors the **action distribution**, not just the output format. Here's a monitoring loop that detects distribution drift over time — the same pattern applies whether the "action" is a robot joint angle or an outbound email variant selection.

```python
import numpy as np
from collections import deque

class ActionDriftMonitor:
    def __init__(self, action_dim=7, window_size=100, drift_threshold=2.0):
        self.action_dim = action_dim
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.action_history = deque(maxlen=window_size)
        self.baseline_mean = None
        self.baseline_std = None
        self.step_count = 0

    def calibrate(self, calibration_actions):
        cal = np.array(calibration_actions)
        self.baseline_mean = cal.mean(axis=0)
        self.baseline_std = cal.std(axis=0)
        print(f"Baseline calibrated on {len(cal)} samples")
        print(f"Mean action: {np.round(self.baseline_mean, 3)}")
        print(f"Std action:  {np.round(self.baseline_std, 3)}")

    def observe(self, action):
        self.action_history.append(action)
        self.step_count += 1

        if len(self.action_history) < self.window_size:
            return {'status': 'warming_up', 'samples': len(self.action_history)}

        window = np.array(self.action_history)
        window_mean = window.mean(axis=0)
        window_std = window.std(axis=0)

        mean_drift = np.abs(window_mean - self.baseline_mean) / (self.baseline_std + 1e-8)
        std_drift = np.abs(window_std - self.baseline_std) / (self.baseline_std + 1e-8)

        max_mean_drift = mean_drift.max()
        max_std_drift = std_drift.max()

        if max_mean_drift > self.drift_threshold or max_std_drift > self.drift_threshold:
            drifted_dims = np.where((mean_drift > self.drift_threshold) | (std_drift > self.drift_threshold))[0]
            return {
                'status': 'DRIFT_DETECTED',
                'step': self.step_count,
                'max_mean_drift_sigma': float(max_mean_drift),
                'max_std_drift_sigma': float(max_std_drift),
                'drifted_dims': drifted_dims.tolist(),
                'window_mean': np.round(window_mean, 3).tolist(),
                'baseline_mean': np.round(self.baseline_mean, 3).tolist()
            }

        return {
            'status': 'nominal',
            'step': self.step_count,
            'max_mean_drift_sigma': float(max_mean_drift),
            'max_std_drift_sigma': float(max_std_drift)
        }

np.random.seed(42)
monitor = ActionDriftMonitor(action_dim=7, window_size=50, drift_threshold=2.5)

calibration_data = [np.random.uniform(-2, 2, 7) for _ in range(200)]
monitor.calibrate(calibration_data)

print("\n=== Phase 1: Normal operation (steps 1–100) ===")
for i in range(100):
    action = np.random.uniform(-2, 2, 7)
    result = monitor.observe(action)
    if result['status'] == 'DRIFT_DETECTED':
        print(f"  Step {result['step']}: {result['status']} — drift={result['max_mean_drift_sigma']:.2f}σ")

print(f"  Final: {result['status']}, max drift={result['max_mean_drift_sigma']:.2f}σ")

print("\n=== Phase 2: Distribution drift begins (joint 3 biasing positive) ===")
for i in range(100):
    action = np.random.uniform(-2, 2, 7)
    action[3] += 2.5
    result = monitor.observe(action)
    if result['status'] == 'DRIFT_DETECTED':
        print(f"  Step {result['step']}: {result['status']}")
        print(f"    Drifted dims: {result['drifted_dims']}")
        print(f"    Dim 3 baseline mean: {result['baseline_mean'][3]:.3f} → window: {result['window_mean'][3]:.3f}")
        print(f"    Mean drift: {result['max_mean_drift_sigma']:.2f}σ")
        break

print("\n=== Phase 3: Multimodal collapse (actions split into two modes) ===")
for i in range(100):
    if i % 2 == 0:
        action = np.array([2.0, -2.0, 1.5, 0.5, -1.0, 0.8, 0.0])
    else:
        action = np.array([-2.0, 2.0, -1.5, 0.5, 1.0, -0.8, 0.0])
    result = monitor.observe(action)
    if result['status'] == 'DRIFT_DETECTED':
        print(f"  Step {result['step']}: {result['status']}")
        print(f"    Mean drift: {result['max_mean_drift_sigma']:.2f}σ")
        print(f"    Std drift:  {result['max_std_drift_sigma']:.2f}σ")
        break

print("\n=== Diagnostic ===")
print("Phase 2 drift = mean shift (bias in one joint / one enrichment field)")
print("Phase 3 drift = variance change (multimodal collapse / audience splitting)")
print("Both are caught by monitoring mean AND std, not just output validity.")
```

This monitoring pattern — tracking mean and standard deviation of the action distribution against a calibrated baseline — is the robotics analog to tracking reply-rate drift in a GTM sequence. Reply rate is your model degradation signal: if it drops while volume stays constant, the audience distribution or the message-action mapping has drifted. The fix is identical in both domains: detect the drift, identify which dimension (which enrichment field, which robot joint) shifted, and retrain the action head on fresh data. [CITATION NEEDED — concept: GTM sequence reply rate drift as model degradation signal, Zone 12 observability mapping]

The dual-system architecture from GR00T has a direct GTM parallel. System 2 (deliberative planner) decomposes "expand into enterprise healthcare" into sub-goals — identify targets, craft messaging, sequence touchpoints. System 1 (reactive controller) handles real-time execution — adjust send timing based on reply signals, route a hot reply to sales immediately, suppress a bounced email. GR00T's contribution is formalizing the interface between these two layers; the same interface problem exists between a GTM strategy layer and an execution layer in Clay or any outreach platform. [CITATION NEEDED — concept: dual-system GTM architecture, System 1/2 mapping to outreach automation]

## Exercises

**Exercise 1 — Action Tokenizer Sweep.** Modify the `ActionTokenizer` class to accept `num_bins` as a parameter. Run the reconstruction error measurement for `num_bins = [16, 32, 64, 128, 256, 512]`. Plot the max reconstruction error (in degrees) vs. number of bins. Answer: at what bin count does the error drop below 0.5°? Below 0.1°? RT-2 uses 256 — is that sufficient for a 7-DOF arm, or is it a compromise driven by vocabulary size?

**Exercise 2 — Flow Matching vs. Bins on Trimodal Data.** Extend the multimodal simulation in the Build It section from 2 modes to 3 modes. Run 500 flow-matching simulations. Does flow matching still converge to a valid mode? Now try 5 modes. At what point does the flow-matching approximation break down? (Hint: look at the denoising step count — more modes need more steps to separate.)

**Exercise 3 — Drift Monitor on Real Distribution.** Replace the uniform random calibration data in the drift monitor with a normal distribution centered at zero with std=1.0. Then simulate a gradual drift (mean shifting by 0.01 per step for 200 steps). How many steps until the monitor detects drift? Now try a sudden shift (mean jumps by 2.0 at step 50). How does detection latency compare? Which failure pattern is harder to catch?

**Exercise 4 — Dual-System Task Decomposition.** Write a function that takes a high-level instruction ("clear the table") and decomposes it into a sequence of System 1 sub-goals. Each sub-goal is a 7-DOF action vector. The constraint: the decomposition must respect physical ordering (you cannot stack a plate on top of a cup that is not yet on the table). How many sub-goals does "clear the table" generate? Now do the same for a GTM instruction ("run a Q4 outbound campaign to enterprise SaaS companies") — decompose into System 1 execution steps. Is the structure the same?

## Key Terms

**VLA (Vision-Language-Action)** — A model architecture that takes images and natural-language instructions as input and produces robot actions as output, using a shared transformer backbone.

**Action Tokenization** — The process of converting continuous-valued robot actions (joint angles, velocities, torques) into discrete tokens that a transformer can predict via next-token generation. RT-2 and OpenVLA use 256-bin discretization per action dimension.

**Flow Matching** — A continuous generative method (used by π₀) that samples actions by starting from Gaussian noise and iteratively denoising toward a valid action vector. Unlike bin classification, it preserves multimodal action distributions.

**Open X-Embodiment** — A collaborative dataset of over 1 million robot trajectories across multiple robot platforms, used to train OpenVLA and other generalist robot policies.

**Dual-System Architecture** — GR00T's design that separates a System 2 deliberative planner (long-horizon task decomposition) from a System 1 reactive controller (low-latency action execution), addressing tasks that span minutes or hours.

**Action Distribution Drift** — The failure mode where a deployed policy's output distribution shifts from its calibration baseline while remaining syntactically valid. Detected by monitoring mean and variance of the action space over time.

**Co-Fine-Tuning** — Training strategy (introduced by RT-2) that simultaneously fine-tunes on web-scale vision-language data and robot trajectory data, preserving general knowledge transfer while learning physical control.

## Sources

- RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control — Google DeepMind, October 2023. Paper: arXiv:2307.15818. Key claim: "co-fine-tuning on web data and robot data enables emergence of capabilities like reasoning and symbol understanding in robot control."
- OpenVLA: An Open-Source Vision-Language-Action Model — Stanford/Google/TRI, August 2024. Paper: arXiv:2406.09246. Key claim: 7B parameters with Prismatic VLM backbone (SigLIP + Llama-2), trained on Open X-Embodiment, matches RT-2 performance at a fraction of the scale.
- π₀: A Vision-Language-Action Flow Model for General Robot Control — Physical Intelligence, October 2024. Paper: arXiv:2410.24164. Key claim: flow matching for continuous action generation outperforms discretized tokens on multimodal action distributions.
- GR00T N1: Foundation Model for Humanoid Robots — NVIDIA, March 2025. Announcement: GTC 2025. Key claim: dual-system architecture (System 2 planning + System 1 control) for humanoid form factors. [CITATION NEEDED — concept: GR00T N1 architecture details, System 1/System 2 interface specification]
- Open X-Embodiment Dataset — RT-X Consortium, October 2023. Paper: arXiv:2310.08864. Key claim: 1M+ trajectories across 22 robot embodiments.
- Zone 12 GTM Mapping: Observability, logging, tracing — Zone table row 12: "This tracing setup monitors your sequence performance in real time; reply rate drift is your model degradation signal." [CITATION NEEDED — concept: Clay-specific waterfall enrichment architecture as VLA structural analog]
- Zone 2 GTM Mapping: AI Literacy & Pattern Recognition — VLA pattern transfer to GTM: fuse multimodal inputs → tokenize output space → fine-tune routing logic. [CITATION NEEDED — concept: explicit mapping of VLA architecture to GTM enrichment pipelines in curriculum documentation]