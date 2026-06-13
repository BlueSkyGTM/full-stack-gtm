# Embodied VLAs: RT-2, OpenVLA, π₀, GR00T

---

## Beat 1: Hook

The gap between a model that says "pick up the cup" and one that computes joint torques to do it. That gap is the action space problem, and VLAs are the current best answer. Four architectures, four different bets on how to cross it.

---

## Beat 2: Concept

**VLA = Vision-Language-Action.** The mechanism: a vision encoder extracts visual features, a language model processes the instruction, and a policy head maps the fused representation to continuous robot actions. The core tension is tokenization—how do you turn 7-DOF joint positions into tokens a transformer can emit? Each model answers this differently: RT-2 discretizes into action bins, OpenVLA does the same but open-weights, π₀ uses flow matching to generate continuous actions, and GR00T scales for humanoids. All share the same architectural skeleton: frozen vision backbone → language model backbone → action head.

---

## Beat 3: Mechanism

### RT-2 (Google, 2023)
- Backbone: PaLI-X or PaLM-E VLMs (55B+ parameters)
- Action tokenization: 256 discrete bins per dimension, 7 dimensions × 1 token each
- Training: co-fine-tunes on web data + robot trajectories simultaneously
- Key insight: web knowledge transfers to physical manipulation without domain-specific pretraining

### OpenVLA (Stanford/Google, 2024)
- Backbone: Prismatic VLM (SigLIP + Llama-2 7B)
- Same bin discretization as RT-2 but fully open-source
- Fine-tune on Open X-Embodiment dataset (1M+ episodes)
- Key insight: 7B parameters sufficient when training data is curated well

### π₀ (Physical Intelligence, 2024)
- Backbone: VLM (PaliGemma variant) + flow-matching action head
- Action generation: continuous via flow matching (not discretized bins)
- Trained on 10,000+ robot episodes across multiple embodiments
- Key insight: flow matching captures multimodal action distributions better than bin classification

### GR00T (NVIDIA, 2024)
- Backbone: Vision transformer + language decoder, targeting humanoid form factors
- Designed for Isaac Lab simulation → real transfer
- Key insight: sim-to-real pipeline scale via NVIDIA's compute stack

**Common pattern across all four:** freeze vision encoder, fine-tune LLM backbone, train action head from scratch. The divergence is in the action head design.

---

## Beat 4: Use It

**GTM Redirect:** Foundational for **Zone 2 — AI Literacy & Pattern Recognition**. The VLA pattern (fuse multimodal inputs → tokenize output space → fine-tune backbone) is structurally identical to how Clay fuses firmographic signals + intent data → tokenized outreach actions. The architecture pattern transfers even if the domain doesn't.

### Exercise Hooks

- **Easy:** Load OpenVLA from HuggingFace, run inference on a single image-instruction pair, print the predicted action tokens. Compare action dimensions to RT-2's bin scheme.
- **Medium:** Implement a minimal action tokenizer that discretizes continuous joint positions into N bins, then reconstructs. Measure reconstruction error as a function of bin count. This is what RT-2 and OpenVLA do internally.
- **Hard:** Compare π₀'s flow-matching action head to RT-2's discretized approach. Implement a toy 2D reacher environment. Train two policy heads: (1) classification over binned actions, (2) conditional flow matching. Report success rate on multimodal goal distributions (e.g., reach around obstacle—two valid paths).

---

## Beat 5: Ship It

Deploy OpenVLA on a simulated Franka Emika Panda in Isaac Sim or MuJoCo. Inference pipeline: capture RGB frame → preprocess to 224×224 → run through OpenVLA → extract action tokens → de-tokenize to joint positions → step simulation. Print per-step latency and action magnitudes. Compare to a scripted policy baseline on the same task.

**GTM Redirect:** Same deployment pattern as executing a Clay waterfall—observe state, select action, execute, observe again. The loop structure is the mechanism, not the domain.

---

## Beat 6: Evaluate

### Checkpoint Questions

1. **Mechanism:** RT-2 discretizes each action dimension into 256 bins. What is the reconstruction error bound for a joint range of [−π, π]? What happens to the bound if you halve the bin count?

2. **Comparison:** π₀ uses flow matching instead of bin classification. What specific failure mode of bin classification does flow matching address? (Hint: think about bimodal action distributions.)

3. **Architecture:** All four VLAs freeze the vision encoder. Why? What would happen during fine-tuning if you unfroze it—what does this trade off?

4. **Scale:** OpenVLA achieves competitive performance at 7B parameters while RT-2 uses 55B. What is the hypothesized mechanism for this efficiency? [CITATION NEEDED — concept: OpenVLA parameter efficiency vs RT-2]

5. **Deployment:** You are running OpenVLA inference at 5Hz on a UR5e. The robot jerks and misses grasps. List three possible causes ranked by likelihood, and for each, state what you would log to confirm.

---

## Learning Objectives

1. **Compare** the action tokenization strategies of RT-2/OpenVLA (discrete bins) vs. π₀ (flow matching) and explain when each fails.
2. **Implement** a minimal action tokenizer and measure reconstruction error as a function of bin count.
3. **Deploy** OpenVLA inference on a simulated robot and benchmark per-step latency against a scripted baseline.
4. **Explain** why all four VLA architectures freeze the vision encoder during fine-tuning and what this trades off.
5. **Evaluate** which VLA architecture is appropriate for a given embodiment and action space constraint.

---

## Notes

- π₀ and GR00T are not fully open-weight at time of writing. Mechanism descriptions are based on published papers and blog posts; inference code may not be reproducible without access. OpenVLA and RT-2 have publicly available weights/code.
- The GTM redirect is structural, not domain-specific. VLAs do not directly map to sales/marketing workflows. The pattern (multimodal fusion → tokenized action) is the transferable concept.