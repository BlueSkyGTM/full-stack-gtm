# Activation Functions

## Hook

A neural network without activation functions is just a linear regression model — no matter how many layers you stack. Activation functions are what let networks learn non-linear decision boundaries. This beat shows what breaks when you remove them, and why that matters for any model that classifies, scores, or ranks.

## Concept

This beat covers the mechanism: each function compresses or transforms its input into a bounded or semi-bounded range. Walk through ReLU (piecewise linear, cheap, suffers dead neurons), sigmoid (bounded 0–1, vanishing gradient), tanh (bounded –1 to 1, zero-centered), softmax (probability distribution over classes), and GELU (smooth approximation used in transformer architectures). For each: the math, the gradient, and the failure mode. Mechanism first — ReLU zeros out negatives; sigmoid squashes extremes; softmax normalizes via exponentiation.

## Code It

Build a single script that implements each activation function from scratch (pure Python or NumPy), feeds a range of inputs through each, and prints the input-to-output mapping plus the derivative at key points. Observable output: a printed table showing how each function transforms –10, –1, 0, 1, 10 and what the gradient looks like at each point. No PyTorch or TensorFlow — just the math.

## Use It

**GTM Redirect:** Sigmoid outputs map directly to lead scoring models that produce a "probability to convert" — this is the mechanism behind Zone 1 enrichment pipelines that classify inbound leads. Softmax powers multi-class intent classification (which ICP segment, which competitor, which stage of buyer journey). If the model scores leads or classifies intent, an activation function is the final transform that turns raw logits into an actionable number. Foundational for Zone 1 (Data Foundation) scoring and Zone 2 enrichment workflows.

## Ship It

**Easy:** Extend the code block to plot each activation function and its derivative side-by-side using matplotlib, saved as a PNG.

**Medium:** Build a toy 2-layer network (NumPy only) that classifies a synthetic dataset. Run it twice — once with no activation function (identity), once with ReLU — and print accuracy for both to demonstrate the linear bottleneck.

**Hard:** Implement a dead-neuron detector: feed 1,000 random inputs through a ReLU layer, track which neurons output zero for >95% of inputs, and print the indices of dead neurons. This is the debugging pattern used when training scoring models that plateau.

## Dig Deeper

- Link to the original ReLU paper (Nair & Hinton, 2010) and the GELU paper (Hendrycks & Gimpel, 2016)
- Reference the vanishing gradient problem as explained in the backpropagation lesson
- Point ahead to the lesson on loss functions — softmax and cross-entropy are paired by design, and that connection is where scoring models get their calibration
- [CITATION NEEDED — concept: activation function choice impact on lead scoring model convergence rates]