# World Models & Video Diffusion

---

## Hook

Prediction is compression. A world model that can predict the next frame of video has necessarily learned a compressed representation of how scenes evolve over time—physics, motion, occlusion, persistence. Video diffusion inverts this: it generates plausible temporal sequences by denoising over both space and time. These two ideas are converging. This lesson covers the mechanisms behind both.

---

## Concept

**World models** are learned simulators. Given state $s_t$ and action $a_t$, they predict $s_{t+1}$. They compress high-dimensional observation streams into latent dynamics that can be rolled forward for planning, simulation, or data generation.

**Video diffusion** applies the denoising score-matching framework to spatiotemporal volumes—sequences of frames rather than single images. The key architectural addition is temporal attention: layers that attend across frames to enforce coherence.

The convergence: modern video diffusion systems (Sora, Runway Gen-3) behave as implicit world models. They don't just generate pixels—they simulate plausible physical trajectories.

Core vocabulary for this lesson:
- **Latent dynamics model**: a network that transitions latent states forward in time
- **Temporal attention**: cross-frame attention mechanism that enforces consistency
- **Spatiotemporal U-Net**: 3D convolution + temporal attention backbone for video diffusion
- **Autoregressive vs. diffusion generation**: next-frame prediction vs. joint denoising over a window

---

## Mechanism

### Part 1: World Models — Learned Simulation

The world model formulation (Ha & Schmidhuber, 2018; Hafner et al., 2020):

1. **Encoder**: $z_t = \text{Enc}(o_t)$ — compress observation to latent
2. **Dynamics**: $z_{t+1} = \text{RSSM}(z_t, a_t)$ — predict next latent state
3. **Decoder**: $\hat{o}_t = \text{Dec}(z_t)$ — reconstruct observation from latent

The Recurrent State Space Model (RSSM) combines:
- Deterministic path: recurrent network (GRU/LSTM) carries sequential state
- Stochastic path: latent variable captures uncertainty in transitions

Training minimizes reconstruction loss + KL divergence (keeping the latent space regularized) + reward prediction (optional, for RL applications).

The system learns physics not from equations but from observation streams. Roll it forward and it "dreams" plausible futures.

### Part 2: Video Diffusion — Denoising Over Time

Video diffusion extends image diffusion (see Lesson 047) by adding a temporal axis:

1. **Noise schedule** applied to full clip: $x_T \sim \mathcal{N}(0, I)$ where $x$ is a frame sequence
2. **Denoising network** processes spatiotemporal volume:
   - **3D convolutions** (or 2D + temporal) extract spatial features per frame
   - **Temporal attention layers** attend across all frames at each spatial position
   - **Spatial attention layers** attend within each frame (standard self-attention)
3. **Conditioning**: text prompts, reference images, or previous frames condition generation

The critical mechanism is temporal attention. Without it, each frame denoises independently and you get flickering incoherence. With it, the model learns that objects persist, that motion is smooth, that shadows move with their casters.

Training data requirements are massive: thousands to millions of video clips with aligned captions. The model learns world dynamics implicitly through the denoising objective.

### Part 3: The Convergence

Modern systems blur the boundary:

- **Sora** (OpenAI, 2024): diffusion model trained on video patches (spacetime tokens). Generates long, coherent clips. Behaves as a world simulator despite being trained only on denoising. [CITATION NEEDED — concept: Sora technical report architecture details]
- **GAIA-1** (Wayve, 2023): world model for autonomous driving using autoregressive + diffusion hybrid. Predicts future video frames conditioned on past frames + actions.
- **UniSim** (Brohan et al., 2023): universal simulator using video diffusion trained on robot interaction data.

The shared insight: a model trained to predict/generate video must learn physics, object permanence, and scene dynamics as a side effect.

---

## Code

### Example 1: Latent Dynamics — Forward Prediction with Uncertainty

This demonstrates the world model concept: encode state, transition in latent space, decode. Uses a simple 2D ball trajectory for clarity.

```python
import numpy as np
import struct

def encode_ball_state(x, y, vx, vy):
    """Compress 4D state to 2D latent (demonstration of compression)"""
    z1 = (x + vx) * 0.7
    z2 = (y + vy) * 0.7
    return np.array([z1, z2])

def decode_ball_state(z1, z2):
    """Reconstruct 4D state from 2D latent"""
    x = z1 / 0.7 * 0.5
    vx = z1 / 0.7 * 0.5
    y = z2 / 0.7 * 0.5
    vy = z2 / 0.7 * 0.5
    return x, y, vx, vy

def latent_transition(z, dt=0.1):
    """Predict next latent state (deterministic + stochastic)"""
    deterministic = z + 0.05 * z
    noise = np.random.randn(*z.shape) * 0.02
    return deterministic + noise

x, y, vx, vy = 1.0, 0.5, 2.0, 1.5
print("=== World Model: Latent Dynamics Demo ===")
print(f"Initial state: x={x:.2f} y={y:.2f} vx={vx:.2f} vy={vy:.2f}")

for step in range(5):
    z = encode_ball_state(x, y, vx, vy)
    z_next = latent_transition(z)
    x_r, y_r, vx_r, vy_r = decode_ball_state(z_next[0], z_next[1])
    x, y, vx, vy = x + vx * 0.1, y + vy * 0.1, vx_r, vy_r
    print(f"Step {step+1}: x={x:.3f} y={y:.3f} vx={vx:.3f} vy={vy:.3f}")

print("\nLatent compression: 4D state -> 2D latent -> 4D reconstruction")
print("Stochastic transition adds uncertainty (epistemic noise)")
```

**Expected output**: 5-step trajectory showing predicted ball positions. Each run differs slightly due to stochastic transitions.

### Example 2: Temporal Consistency — Measuring Frame Coherence

This demonstrates why temporal attention matters: it measures whether generated frames maintain object consistency.

```python
import numpy as np

def compute_temporal_consistency(frames):
    """Measure pixel-level consistency between consecutive frames.
    Lower values = more coherent (less random flickering)."""
    consistencies = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i+1].astype(float) - frames[i].astype(float))
        avg_diff = np.mean(diff)
        consistencies.append(avg_diff)
    return consistencies

def generate_coherent_frames(num_frames=5, size=16):
    """Simulate temporally coherent frames (object persists, moves slowly)."""
    frames = []
    obj_pos = size // 2
    for _ in range(num_frames):
        frame = np.zeros((size, size), dtype=np.uint8)
        frame[obj_pos, obj_pos] = 255
        obj_pos = min(obj_pos + 1, size - 1)
        frames.append(frame)
    return frames

def generate_incoherent_frames(num_frames=5, size=16):
    """Simulate temporally incoherent frames (random flickering)."""
    frames = []
    for _ in range(num_frames):
        frame = np.zeros((size, size), dtype=np.uint8)
        y, x = np.random.randint(0, size, 2)
        frame[y, x] = 255
        frames.append(frame)
    return frames

coherent = generate_coherent_frames()
incoherent = generate_incoherent_frames()

coh_scores = compute_temporal_consistency(coherent)
incoh_scores = compute_temporal_consistency(incoherent)

print("=== Temporal Consistency Measurement ===")
print(f"Coherent frames avg diff: {np.mean(coh_scores):.2f}")
print(f"Incoherent frames avg diff: {np.mean(incoh_scores):.2f}")
print(f"Ratio (incoherent/coherent): {np.mean(incoh_scores)/max(np.mean(coh_scores), 0.01):.1f}x more change")
print("Temporal attention in video diffusion minimizes this difference.")
```

**Expected output**: Coherent frames show low difference (~1-2), incoherent show high (~254). Demonstrates what temporal attention prevents.

### Example 3: Diffusion Timestep Visualization on Sequential Data

```python
import numpy as np

def add_noise_to_sequence(sequence, t, total_timesteps=100):
    """Apply forward diffusion to a sequence at timestep t."""
    beta_start, beta_end = 0.0001, 0.02
    betas = np.linspace(beta_start, beta_end, total_timesteps)
    alpha = 1.0 - betas
    alpha_cumprod = np.cumprod(alpha)
    
    sqrt_alpha_t = np.sqrt(alpha_cumprod[t])
    sqrt_one_minus_alpha_t = np.sqrt(1.0 - alpha_cumprod[t])
    
    noise = np.random.randn(*sequence.shape)
    noisy = sqrt_alpha_t * sequence + sqrt_one_minus_alpha_t * noise
    return noisy, noise

clean_sequence = np.array([np.sin(x * 0.5) for x in range(20)])
print("=== Diffusion Noise Schedule on Temporal Sequence ===")
print(f"Clean sequence (first 8): {clean_sequence[:8].round(3)}")

for t in [0, 25, 50, 75, 99]:
    noisy, _ = add_noise_to_sequence(clean_sequence, t)
    snr = np.var(clean_sequence) / max(np.var(noisy - clean_sequence), 1e-8)
    print(f"Timestep {t:3d}: SNR={snr:6.2f} | values: {noisy[:8].round(3)}")

print("\nAt t=0: nearly clean. At t=99: nearly pure noise.")
print("Video diffusion denoises this process over BOTH spatial and temporal axes.")
```

**Expected output**: Shows progressive noise corruption of a temporal signal. SNR decreases with timestep, illustrating the forward process video diffusion reverses.

---

## Use It

**GTM Cluster: Zone 5 — AI Foundations / Zone 3 — Content Engine**

World models and video diffusion are foundational for **predictive simulation** in GTM contexts:

1. **Product demo generation**: Video diffusion can generate walkthrough-style product demonstrations from text descriptions. If you have a Clay workflow or sequence of tool states, a trained video model could synthesize a visual demo without screen recording. This maps to the **Content Engine** cluster for generating marketing assets.

2. **Scenario planning as world modeling**: The latent dynamics framework (encode → predict → decode) is the same pattern used in pipeline forecasting. Encode pipeline state as a latent vector, transition it forward conditioned on "actions" (campaigns, outreach), decode to predicted pipeline metrics. This is not a stretch—it is literally the RSSM architecture applied to sales funnels instead of robot environments.

3. **Temporal consistency for brand coherence**: When generating sequences of marketing content (email drip, ad variants, social carousel), the temporal attention mechanism concept applies: each piece must attend to previous pieces to maintain coherent narrative. The math is different; the principle is the same.

**The clean GTM redirect**: World models are the mechanism behind **predictive pipeline simulation** (Zone 5 foundational). Video diffusion is the generation mechanism for **automated demo and explainer content** (Zone 3, Content Engine cluster).

---

## Ship It

### Easy
Implement a 1D world model: train a simple RSSM-style network (even a linear model) on a sine wave sequence. Report prediction error over a 20-step rollout. Measure how error compounds—this is the "dream divergence" problem.

### Medium
Build a temporal consistency evaluator: given a directory of image frames (from any source), compute frame-to-frame perceptual metrics (pixel diff, SSIM if you want to add a dependency). Output a coherence score. Test it on: (a) a real video extracted to frames, (b) randomly shuffled frames. Report the score difference.

### Hard
Fine-tune a small image diffusion model (not full video—compute constraints) to generate temporally adjacent pairs. Use a dataset of paired frames (e.g., a video dataset where each sample is frame $t$ and frame $t+1$). Condition generation of frame $t+1$ on frame $t$ by concatenating frame $t$ as additional channels in the input. Report whether the model learns object persistence (does the generated frame $t+1$ keep objects from frame $t$ in plausible positions?).

---

## Learning Objectives

1. **Implement** a latent dynamics model that encodes state, transitions forward, and decodes—and measure how prediction error compounds over multi-step rollouts.
2. **Explain** the role of temporal attention in video diffusion and why its absence produces frame-to-frame incoherence.
3. **Compare** autoregressive next-frame prediction with joint spatiotemporal denoising across accuracy, coherence, and compute tradeoffs.
4. **Evaluate** whether a generated video sequence exhibits temporal consistency using quantitative frame-difference metrics.
5. **Describe** how the world model formulation (encoder-dynamics-decoder) maps to predictive simulation in non-visual domains such as pipeline forecasting.