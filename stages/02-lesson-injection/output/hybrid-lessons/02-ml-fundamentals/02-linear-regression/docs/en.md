# Linear Regression

## Learning Objectives

1. Implement ordinary least squares via the normal equation and print interpretable coefficients.
2. Compare closed-form and gradient descent solutions, confirming convergence to the same parameters.
3. Diagnose assumption violations using residual diagnostics and identify the specific failure mode from the residual pattern.
4. Build a lead scoring model using firmographic signals and decompose coefficient values into per-feature contributions.
5. Detect model drift and non-linear relationships to determine when linear regression is the wrong tool.

## The Problem

You already carry mental models about your pipeline. Bigger companies close bigger deals. Funded startups adopt faster than bootstrapped ones. Teams using certain technology stacks churn less. These intuitions are probably correct — but they are untestable until you formalize them. Saying "bigger companies close bigger deals" gives a sales rep no actionable threshold. Saying "each additional 100 employees increases expected deal size by $12,400, and this relationship explains 73% of variance in our closed-won pipeline" gives them a scoring rule.

Linear regression is the formal mechanism for testing that kind of guess. It finds the line — or hyperplane, when you have multiple inputs — that minimizes the total squared error between prediction and reality. The method forces you to name your inputs, name your target, and quantify the relationship. Everything else in predictive modeling builds on this loop: define a model with parameters, define a cost function that measures how wrong the parameters are, optimize the parameters to minimize cost.

This is not a warm-up you skip past. Linear regression runs in production for demand forecasting, A/B test analysis, pricing models, and as the baseline for every regression task. If your fancy gradient-boosted model cannot beat a straight line, the complexity is not justified. The linear model is the floor everything stands on.

## The Concept

### The Model and the Cost Function

Linear regression assumes a linear relationship between input features and output. For a single feature: `y = wx + b`, where `w` is the weight (how much y changes per unit increase in x) and `b` is the bias (the value of y when x is zero). For multiple features, this extends to `y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`, or in vector notation `ŷ = wᵀx + b`. The hat on ŷ distinguishes the prediction from the actual value.

The cost function is mean squared error (MSE): the average of squared differences between predictions and actuals across all training examples. Squaring does two things — it penalizes large errors more than small ones, and it makes the function differentiable everywhere (no absolute-value kink at zero). MSE for `m` training examples is `J(w,b) = (1/m) Σᵢ(ŷᵢ - yᵢ)²`. The goal is to find `w` and `b` that minimize this value.

### Two Paths to the Minimum

There are two ways to find the parameters that minimize MSE. The **normal equation** is a closed-form solution derived by taking the gradient of the cost function, setting it to zero, and solving algebraically. For the model `y = Xw` (with a bias column absorbed into X), the solution is `w = (XᵀX)⁻¹Xᵀy`. This gives you the exact answer in one step, but the matrix inversion is O(n³) in the number of features — fine for dozens of features, painful past a few thousand.

The second path is **gradient descent** — start with random weights, compute the gradient of MSE at that point, and step in the opposite direction. The update rules are `w := w - α · ∂J/∂w` and `b := b - α · ∂J/∂b`, where `α` is the learning rate. The gradient of MSE with respect to weights works out to `(2/m)·Xᵀ(ŷ - y)` and with respect to bias to `(2/m)·Σ(ŷᵢ - yᵢ)`. Gradient descent is iterative and approximate, but it scales to feature spaces where matrix inversion is intractable.

```mermaid
flowchart TD
    A["Training Data (X, y)"] --> B{"Choose Method"}
    B -->|"Closed Form"| C["Normal Equation:<br/>w = (XᵀX)⁻¹Xᵀy"]
    B -->|"Iterative"| D["Initialize w, b randomly"]
    C --> E["Exact w, b"]
    D --> F["Compute predictions