# Loss Functions

## Learning Objectives
1. Implement MSE, MAE, and cross-entropy loss functions from scratch in Python
2. Compare how different loss functions penalize prediction errors using executable code
3. Select the appropriate loss function for regression vs. classification tasks
4. Diagnose model training issues by interpreting loss curves
5. Explain why gradient magnitude depends on loss function choice

---

## Beat 1: Hook — "Wrong Needs a Number"

A model guesses, then it needs to know how wrong it was and in which direction. Loss functions translate "wrongness" into a single scalar that optimization can minimize. Without this, there's no gradient, no update, no learning.

---

## Beat 2: Concept — The Mechanism of Measuring Error

**Regression losses** (continuous output):
- Mean Squared Error (MSE): penalizes large errors quadratically. One outlier dominates.
- Mean Absolute Error (MAE): penalizes linearly. More robust to outliers, but zero gradient at zero error.

**Classification losses** (discrete output):
- Binary Cross-Entropy: measures divergence between predicted probability and true label (0 or 1). Punishes confident wrong answers heavily.
- Categorical Cross-Entropy: extends binary to multi-class. True class gets log(p), everything else gets log(1-p).

**Hinge loss** for maximum-margin classifiers (SVMs): penalizes predictions that are correct but not confident enough.

Key mechanism: the derivative of the loss function determines the gradient. Different loss = different gradient = different learning dynamics even with the same data.

---

## Beat 3: Demonstration — Same Predictions, Different Losses

Code example that:
- Generates a set of predictions and ground truth values
- Computes MSE, MAE, binary cross-entropy, and hinge loss on the same data
- Prints a comparison table
- Shows how a single outlier shifts each loss differently

Exercise hook (easy): Run the code, then change one prediction to an extreme outlier. Print the before/after for each loss function.

Exercise hook (medium): ImplementHuber loss (quadratic near zero, linear far away) and add it to the comparison. Show where it matches MSE and where it matches MAE.

Exercise hook (hard): Plot loss value vs. error magnitude for all four loss functions on a single graph to visualize quadratic vs. linear vs. logarithmic vs. piecewise penalty shapes.

---

## Beat 4: Use It — ICP Fit Scoring as a Loss Problem

[CITATION NEEDED — concept: ICP scoring models trained with loss functions in GTM]

ICP (Ideal Customer Profile) scoring is a classification problem: "does this account match our ICP?" The model predicts a probability. The loss function measures how far off that probability is from the labeled data.

Why this matters:
- **Binary cross-entropy** is the standard for ICP classification. If you label accounts "good fit" / "bad fit," BCE quantifies how well your model separates them.
- **Class imbalance** (few good-fit accounts, many bad-fit) distorts the loss. Weighted cross-entropy or focal loss corrects for this.
- **Threshold selection**: loss functions don't tell you the optimal cutoff for "qualified vs. unqualified." That requires a separate evaluation (precision/recall curves).

GTM redirect: When building or evaluating a lead/account scoring model, the loss function determines what the model optimizes for. Minimizing BCE on imbalanced ICP data without weighting produces a model that calls everything "bad fit" because that gives low loss. This is foundational for [Zone 01 — ICP & Targeting].

---

## Beat 5: Ship It — Loss in a Training Loop

Code example that:
- Builds a minimal single-neuron model (weights + bias)
- Implements forward pass, MSE loss computation, and manual gradient descent
- Runs 100 training steps on synthetic data
- Prints loss at step 0, 50, and 100 to confirm convergence
- Uses no framework beyond NumPy

Exercise hook (easy): Change the learning rate from 0.01 to 0.1 and then to 0.0001. Print loss at each step. Observe divergence vs. slow convergence.

Exercise hook (medium): Replace MSE with MAE in the training loop. Compare final loss values and number of steps to converge.

Exercise hook (hard): Implement a simple binary classification training loop with BCE loss. Generate synthetic 2D data (two clusters), train for 200 steps, print predicted probabilities for 5 test points.

---

## Beat 6: Review — What to Carry Forward

**Three things to remember:**
1. Loss function choice determines what the model optimizes. MSE cares about large errors. MAE cares about median error. BCE cares about probabilistic calibration.
2. The loss function's derivative is the gradient. Without a differentiable loss, you can't do gradient descent.
3. Outliers, class imbalance, and problem type (regression vs. classification) dictate loss selection. There is no universal best loss.

**Common failure mode**: using MSE for classification. It produces non-convex loss landscapes that gradient descent struggles with. Use cross-entropy.

**Next lesson connection**: Loss functions produce gradients. Optimizers decide what to do with those gradients.