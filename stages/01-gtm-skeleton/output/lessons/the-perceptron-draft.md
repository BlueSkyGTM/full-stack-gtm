# The Perceptron

## Hook

The perceptron is the simplest trainable classifier: a linear decision boundary that learns from mistakes. Every neural network is stacked perceptrons; every lead score is a weighted sum run through a threshold. Start here or nothing downstream makes sense.

## Concept

A perceptron computes `z = w·x + b`, passes it through a step function (output: 0 or 1), and updates weights on misclassifications using `w = w + α(y - ŷ)x`. Geometrically: it finds a hyperplane that separates two classes. Mechanism to cover: the update rule is proven to converge if and only if the data is linearly separable. If it isn't, the algorithm never settles — this is the XOR problem, and the reason multi-layer networks exist.

## Use It

**GTM redirect:** Binary classification is the mechanism behind lead scoring, ICP matching, and routing decisions (Zone 1: Lead Intelligence, or Zone 2: Enrichment & Qualification in `gtm-topic-map.md`). A perceptron is a lead score with learnable weights instead of manually-tuned ones. Show how a perceptron trained on firmographic features produces the same kind of go/no-go decision a routing rule does — except it learned the thresholds from data instead of a human guessing.

## Build It

Implement a perceptron from scratch in Python: init weights to zero, loop over training data, apply the update rule, print weights and accuracy each epoch. Train on a small linearly separable dataset (e.g., 2D points where x1 + x2 > 5 is class 1). Then train on XOR to watch it fail. All output printed to terminal.

*Exercise hooks:*
- **Easy:** Run the perceptron on provided separable data, report final weights and accuracy.
- **Medium:** Change the learning rate from 0.1 to 1.0 to 0.001 and compare convergence speed — print epoch count for each.
- **Hard:** Generate a 2D dataset where the perceptron converges slowly (points near the decision boundary) and explain why, using the weight update formula.

## Ship It

A single perceptron in production is a logistic regression without the probability. It's useful as a fast, interpretable binary filter: "route to sales or don't." Cover the deployment pattern: serialize weights, load at inference time, assert that input dimensionality matches training. Warn that the perceptron has no confidence calibration — it outputs 0 or 1, no gradient. For GTM: this means no "warm lead" tier, only hot/cold. That limitation is the redirect to the next lesson (logistic regression / sigmoid).

*Exercise hooks:*
- **Easy:** Save trained weights to JSON, write a second script that loads them and classifies a new point.
- **Medium:** Write a function that takes a batch of inputs and returns predictions without a loop (vectorized dot product).
- **Hard:** Instrument the perceptron to log every misclassification during training; output a CSV of (epoch, index, predicted, actual) and identify which points the model struggled on most.

## Drill It

Three questions, mechanism-focused:
1. Given concrete weights `[2, -3]` and bias `1`, classify the point `(4, 2)` by hand — show the arithmetic.
2. A perceptron trained on a dataset runs 1000 epochs without convergence. Name the necessary condition for convergence that is being violated.
3. Compare the perceptron output function (step) to a typical GTM routing rule (`if score > 80 then route to sales`). Explain why the perceptron's threshold is learnable but the routing rule's isn't.

GTM redirect in drill answers: question 3 bridges directly to why ML-based scoring replaces manual rules — same mechanism, learned parameters instead of hardcoded ones.