# Sim-to-Real Transfer

## Hook

You trained a policy to 99% success in simulation. You deploy it on real hardware. It fails immediately. The simulator's friction coefficient was 0.02 off from reality. That's the reality gap — and sim-to-real transfer is the engineering discipline of crossing it.

## Concept

Sim-to-real transfer is the family of techniques for taking models trained in simulated environments and making them work in production physical systems. The core problem is distribution shift: the simulator samples from an approximate state distribution, reality samples from the true one. This beat covers the taxonomy (domain randomization, domain adaptation, system identification, progressive transfer) and when each applies.

## Mechanism

**Domain randomization**: train over a wide distribution of simulator parameters (friction, mass, sensor noise, lighting) so the policy is invariant to any single parameter value. **Domain adaptation**: learn a mapping between sim and real representations (images, states) using adversarial or contrastive methods. **System identification**: measure real-world dynamics, then calibrate the simulator to minimize the divergence. Each technique makes a different tradeoff between compute cost, data requirements, and robustness guarantees.

## Use It

**GTM redirect**: Foundational for Zone 2 (outbound execution). The domain adaptation pattern underlying sim-to-real — training on one distribution, deploying on another — is the same pattern you encounter when a personalization model trained on SaaS verticals is asked to score Fintech prospects. Domain randomization thinking applies directly to synthetic data generation for GTM enrichment models: vary firmographic parameters during training so the model doesn't overfit to one industry's feature distribution.

## Ship It

Implement domain randomization on a simulated control task (inverted pendulum / cart-pole via OpenAI Gym or a pure-NumPy physics step). Train two agents: one with fixed sim parameters, one with randomized parameters. Perturb the "real" environment parameters and compare performance degradation. Observable output: reward curves for both agents under distribution shift.

**Exercise hooks:**
- **Easy**: Run the fixed-parameter agent. Print reward under nominal vs. perturbed dynamics.
- **Medium**: Add domain randomization. Compare reward retention under the same perturbations.
- **Hard**: Implement a crude system identification loop — measure performance, adjust sim parameters, retrain, measure again. Print convergence of sim-to-real parameter error.

## Evaluate

**Exercise hooks:**
- **Easy**: Given a scenario description, classify which sim-to-real technique applies and justify.
- **Medium**: Trace through a domain randomization training loop and predict which parameter ranges will cause the policy to fail at deployment.
- **Hard**: You have 10,000 real-world samples and an uncalibrated simulator. Design a hybrid approach that uses both. Explain the failure modes.

---

**Learning Objectives (testable):**
1. Detect when a deployment failure is caused by the reality gap vs. a model capacity problem.
2. Implement domain randomization for a simulated control task and measure robustness gains.
3. Compare domain randomization, domain adaptation, and system identification along axes of data cost, compute cost, and robustness.
4. Evaluate whether sim-to-real transfer is the right approach given a problem's data constraints and safety requirements.