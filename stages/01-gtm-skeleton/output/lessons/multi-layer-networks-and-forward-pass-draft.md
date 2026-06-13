# Multi-Layer Networks and Forward Pass

## Learning Objectives

1. Implement a forward pass through a multi-layer network using matrix multiplication.
2. Compare the representational capacity of single-layer vs. multi-layer architectures on non-linearly separable data.
3. Diagnose incorrect layer dimensions by inspecting weight matrix shapes.
4. Insert activation functions between layers and predict their effect on output range.
5. Trace a single input vector through three layers, printing intermediate values at each stage.

---

## Beat 1: Hook — The Wall Every Single-Layer Network Hits

Linear models draw straight lines. Most GTM problems — fit scoring, intent classification, churn prediction — require curved boundaries. Show a 2D dataset where a perceptron fails. Pose the question: how do we stack transformations to solve what one layer cannot? 

**Exercise hook (easy):** Generate a XOR dataset, confirm a single linear boundary misclassifies at least 25% of points.

---

## Beat 2: Concept — Layers as Sequential Transformations

Define a layer as a linear transformation (weights + bias) followed by a non-linear activation. Stack two or more to form a multi-layer network (MLP). Name the components: input layer, hidden layers, output layer. Clarify that depth changes the *composition* of functions, not merely the parameter count. Each hidden layer creates new features from the previous layer's output.

**Exercise hook (medium):** Given a network with architecture [4, 6, 3, 2], write out the shapes of every weight matrix and bias vector without looking at code.

---

## Beat 3: Mechanism — Matrix Multiplication, Activation, Repeat

Unpack the forward pass algorithm: for each layer *l*, compute `z[l] = W[l] @ a[l-1] + b[l]`, then `a[l] = activation(z[l])`. Explain why activation functions (ReLU, sigmoid, tanh) are necessary — without them, two linear layers collapse into one. Show the mathematical proof: `W2 @ (W1 @ x + b1) + b2 = (W2 @ W1) @ x + (W2 @ b1 + b2)`. Clarify that this is a single linear transformation, defeating the purpose of depth.

**Exercise hook (hard):** Implement a 3-layer forward pass from scratch using only `numpy`, then remove activations and prove the output is identical to a single-matrix transformation by computing `W_collapsed = W3 @ W2 @ W1`.

---

## Beat 4: Code — Build It, Trace It

Provide working Python code (numpy only, runs in terminal) that:
1. Defines a 3-layer MLP as dictionaries of weight matrices and bias vectors.
2. Runs a forward pass on a batch of inputs.
3. Prints the shape and range of activations at each layer.
4. Demonstrates dimension mismatch errors and how to fix them.

No PyTorch or TensorFlow yet — the mechanism must be visible in raw matrix operations before any framework abstracts it away.

**Exercise hook (medium):** Modify the hidden layer width from 8 to 16 and predict (then verify) how parameter count changes.

---

## Beat 5: Use It — GTM Redirect: Predictive Scoring Models

Multi-layer networks are the architecture behind predictive lead scoring, account-fit models, and engagement-prediction systems. In GTM Zone 2 (Scoring & Enrichment) [CITATION NEEDED — concept: GTM Zone 2 scoring models using MLPs], scored inputs (firmographic features, activity signals, intent data) pass through hidden layers that learn non-linear interactions — e.g., "company size × recent page visits × industry" creates a composite signal no linear model captures. The forward pass is literally the scoring function: input = account features, output = probability of conversion.

**Exercise hook (easy):** Build a toy scoring model: 5 account features → 3-layer MLP → single probability output. Feed 10 mock accounts and rank them.

---

## Beat 6: Ship It — Debugging the Forward Pass in Production

In production, forward passes fail silently. Covers: (1) numerical instability from unbounded activations, (2) exploding activations when weight initialization is wrong, (3) shape mismatches when input feature count changes upstream. Demonstrate gradient-free sanity checks: verify output shape, check activation ranges, confirm that random weights produce near-uniform distributions on untrained networks. These checks run before any training loop starts.

**Exercise hook (hard):** Inject three deliberate bugs (wrong shape, NaN-producing activation, missing bias) into a working forward pass. Write a diagnostic function that detects and identifies each bug type from the output alone.

---

## GTM Redirect Rules

- **Beat 5 redirect:** This is the mechanism behind predictive lead scoring and account-fit models in Zone 2. The forward pass computes the score; the hidden layers learn feature interactions that linear scoring cannot.
- **Beat 6 redirect:** Production scoring pipelines require the same forward-pass diagnostics. A scoring model that outputs NaN or silently reshapes inputs produces garbage rankings in your CRM.
- **Foundational note:** If the learner's GTM stack uses only rule-based scoring (no ML models), this lesson is foundational for Zone 3 (Segmentation) where non-linear boundaries separate high-intent from low-intent segments.