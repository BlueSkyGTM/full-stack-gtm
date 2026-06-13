# Calculus for Machine Learning

## Beat 1: Hook — Why Derivatives Run Everything in ML

One-sentence summary: Every model you train is just calculus — derivatives tell you which direction to push weights to reduce error, and if you can't read the gradient, you can't debug training.

Key framing: You don't need to derive integrals by hand. You need to read a gradient the way a pilot reads an instrument panel: what's happening, what's about to go wrong, what knob to turn.

---

## Beat 2: Learn It — Derivatives, Partial Derivatives, and the Chain Rule

**Concepts covered:**
- Derivative as slope: rate of change at a point
- Partial derivative: which input moves the output the most
- Chain rule: how gradients flow through composed functions (this is backpropagation)
- Numerical gradient: approximating derivatives with code to verify your math

**Code examples:**
- Compute numerical derivative of `f(x) = x²` and compare to analytical `2x`
- Compute partial derivatives of `f(x, y) = 3x² + y³` numerically
- Chain rule demonstration: `f(g(x))` where `g(x) = x²` and `f(z) = 3z`

Exercise hooks:
- Easy: Compute and print the numerical derivative of `f(x) = x³` at `x = 2`
- Medium: Compute both partial derivatives of `f(x,y) = x²y + y³` and verify against the analytical solution
- Hard: Implement a function that computes the gradient of an arbitrary n-dimensional function using numerical differentiation

---

## Beat 3: Use It — Gradient Descent as Optimization

**GTM Redirect:** Foundational for Zone 02 (AI Engineering Foundations). The mechanism is: lead scoring models and classification models in your GTM stack are trained by minimizing a loss function via gradient descent. Clay's scoring formulas, custom ML models for ICP matching — all optimize weights this way.

One-sentence summary: Gradient descent is the algorithm that finds the minimum of a function by repeatedly stepping in the direction of the negative gradient.

**Code examples:**
- Implement gradient descent on `f(x) = x² + 4x + 4` (convex, easy)
- Show divergence when learning rate is too high
- Implement gradient descent on a 2D function: `f(x,y) = x² + y²`

Exercise hooks:
- Easy: Run gradient descent on `f(x) = (x-3)²` starting from `x = 0`, print each step
- Medium: Add a learning rate parameter and demonstrate convergence failure at `lr = 1.1` vs convergence at `lr = 0.1`
- Hard: Implement gradient descent to find the minimum of `f(x,y) = (x-2)² + (y+1)²` and print the trajectory

---

## Beat 4: Build It — Training a Linear Regression from Scratch

One-sentence summary: Combine partial derivatives, the chain rule, and gradient descent to train a single-neuron model (linear regression) on real data, computing gradients by hand.

**Code examples:**
- Generate synthetic data: `y = 3x + 7 + noise`
- Implement MSE loss function
- Compute gradients of MSE with respect to weight and bias
- Training loop: update weights, print loss every 10 epochs
- Plot or print final weights vs true weights

Exercise hooks:
- Easy: Train the model with a learning rate of `0.01` for 100 epochs and print final weight/bias
- Medium: Modify the training loop to stop when loss improvement is below a threshold (early stopping)
- Hard: Add L2 regularization (weight decay) to the loss function and compute the modified gradient

---

## Beat 5: Ship It — Gradient Checking and Learning Rate Scheduling

**GTM Redirect:** Foundational for Zone 02. When you deploy a custom scoring model — whether in Clay, a Python microservice, or a notebook pipeline — gradient checking verifies your implementation is correct, and learning rate scheduling ensures convergence on real data. [CITATION NEEDED — concept: production gradient verification in deployed ML models]

One-sentence summary: Gradient checking confirms your backprop implementation matches the numerical gradient; learning rate scheduling controls convergence speed in production training.

**Code examples:**
- Implement gradient checking: compare analytical gradient to numerical gradient, return relative error
- Implement step-decay learning rate schedule
- Implement exponential decay schedule
- Demonstrate training with and without scheduling on the same dataset

Exercise hooks:
- Easy: Run gradient check on the linear regression model from Beat 4 and print the relative error
- Medium: Implement cosine annealing learning rate schedule and compare convergence to step decay
- Hard: Implement a gradient check for a 2-layer neural network (one hidden layer with ReLU)

---

## Beat 6: Extend It — Where Calculus Gets Heavier

One-sentence summary: The Jacobian generalizes gradients to vector-valued functions (used in every neural network layer), the Hessian tells you about curvature (used in second-order optimizers), and automatic differentiation is what PyTorch and JAX actually compute under the hood.

**Topics listed, not fully built:**
- Jacobian matrices: gradient of a vector output
- Hessian: second derivatives, curvature, why they matter for optimization landscape
- Automatic differentiation vs numerical differentiation vs symbolic differentiation
- Why you never compute Hessians in practice (L-BFGS approximates them)

**Reading pointers:**
- Chapter 4 of *Deep Learning* (Goodfellow et al.) — "Numerical Computation"
- 3Blue1Brown's "Essence of Calculus" series (visual intuition)
- PyTorch autograd documentation (for seeing autodiff in production)

---

## Learning Objectives

1. **Compute** numerical gradients of scalar and multivariable functions and verify them against analytical solutions.
2. **Implement** gradient descent from scratch and diagnose convergence failures caused by learning rate misconfiguration.
3. **Train** a linear regression model using manually computed gradients and a training loop.
4. **Evaluate** the correctness of gradient implementations using numerical gradient checking.
5. **Compare** learning rate schedules (step decay, exponential decay, cosine annealing) and their effect on convergence speed.

---

## GTM Redirect Rules (Summary)

- **Beat 3 (Use It):** Gradient descent is the optimization mechanism behind lead scoring models, ICP classifiers, and any custom ML model in a GTM stack. Foundational for Zone 02.
- **Beat 5 (Ship It):** Gradient checking and learning rate scheduling are production verification tools for any deployed model that trains on GTM data. Foundational for Zone 02.
- No forced connection: calculus does not directly map to Clay waterfalls or enrichment logic. The redirect is honest — this is the math that makes model training work.