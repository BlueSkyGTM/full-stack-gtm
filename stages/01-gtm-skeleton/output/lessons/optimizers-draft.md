# Optimizers

## Hook It

The loss landscape is a high-dimensional surface your model must descend. The optimizer is the navigation algorithm. Wrong optimizer, wrong minimum, wrong predictions — no amount of data or architecture fixes that.

## See It

Implement gradient descent from scratch on a convex 2D surface, print coordinates at each step, show convergence. Then swap momentum in and print the same path — fewer steps, same minimum. Observable output confirms the mechanism before any framework enters.

**Exercise hook (easy):** Run vanilla GD with three different learning rates on the same bowl. Print step count to convergence. Identify which diverges.

## Build It

Construct SGD → SGD+momentum → Adam in plain Python. Each variant builds on the previous by adding one term (velocity, then adaptive learning rates). Every implementation prints parameter values and loss at each step against the same toy objective.

**Exercise hook (medium):** Implement Adam from the paper equations. No peeking at PyTorch source. Validate your output matches `torch.optim.Adam` on the same synthetic loss.

## Use It

Fine-tuning a classifier for lead scoring requires choosing an optimizer and learning rate. Adam converges faster on sparse tabular features; SGD with momentum generalizes better on small datasets. [CITATION NEEDED — concept: optimizer selection for fine-tuning on structured GTM data]. This is foundational for Zone 2 (enrichment models) and Zone 3 (scoring models) where fine-tuned classifiers process enriched firmographic data.

**Exercise hook (medium):** Fine-tune a small classifier on a synthetic lead-scoring dataset twice — once with SGD+momentum, once with Adam. Print final loss and accuracy for both. Write one sentence explaining which you'd ship and why.

## Ship It

Wrap optimizer selection in a training config. Implement a learning rate finder: start at 1e-6, scale up by 10x each mini-batch, print loss at each step, halt when loss diverges. The steepest descent point is your learning rate. This is the same procedure fast.ai popularized; implement it from mechanism, not library.

**Exercise hook (hard):** Build a reusable `select_optimizer` function that accepts a model, dataset, and config dict. It runs the LR finder, prints the plot data (step, lr, loss), then returns an initialized optimizer with the selected learning rate. Test it on two different synthetic datasets.

## Debug It

**Loss spikes after epoch 10:** Learning rate too high for Adam — divide by 10, print loss per step to confirm spikes vanish.  
**Loss plateaus but training loss keeps dropping:** Not an optimizer problem — that's overfitting, addressed by regularization, not LR schedule.  
**NaN loss:** Gradient explosion. Print gradient norms per step. If norms exceed 1e3, clip gradients at 1.0 and re-run.  
**Adam converges but SGD doesn't:** Scale SGD learning rate up by 10x — Adam's adaptive scaling hides the LR sensitivity that SGD exposes directly.

**Exercise hook (easy):** Given a training log with NaN appearing at step 47, add gradient norm printing to identify the explosion point, then apply gradient clipping and confirm recovery.

---

**GTM Redirect:** Foundational for Zone 2 and Zone 3 — optimizer choice determines whether a fine-tuned enrichment or scoring model converges to a useful minimum or stalls on noisy firmographic features.