# Training Loop and Evaluation

---

## Beat 1: Hook

The training loop is the engine that turns random weights into a working model. Without understanding what happens inside each epoch—forward pass, loss computation, backward pass, parameter update—you're just tuning hyperparameters blindfolded. Evaluation metrics are how you know whether the loop is actually learning or just memorizing noise.

---

## Beat 2: Concept

### The Training Loop Mechanism

A training loop has four steps that repeat until convergence or a stopping condition: (1) forward pass to compute predictions, (2) loss calculation to measure error, (3) backward pass to compute gradients, (4) optimizer step to update weights. This cycle runs across batches within epochs. A batch is a slice of data; an epoch is one full pass through the dataset.

### Evaluation Mechanism

Evaluation runs the forward pass without gradient computation (no learning). Metrics like accuracy, precision, recall, and F1 score quantify different aspects of model performance. A model that achieves 99% accuracy on training data but 60% on held-out data has overfit—it memorized patterns specific to training samples rather than learning generalizable features.

### Train/Val/Test Split

The dataset splits into three partitions. Training data drives weight updates. Validation data informs hyperparameter decisions and early stopping. Test data is held out entirely until final evaluation—using it during development contaminates the signal.

---

## Beat 3: Demo

Build a minimal training loop from scratch on a synthetic binary classification problem. Show the loss curve decreasing across epochs. Demonstrate overfitting by running too many epochs and showing training accuracy diverge from validation accuracy. Print metrics at each epoch to observable output.

```python
import numpy as np

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

X = np.random.randn(200, 3)
true_weights = np.array([1.5, -2.0, 0.8])
true_bias = -0.5
logits = X @ true_weights + true_bias
y = (logits + np.random.randn(200) * 0.5 > 0).astype(float)

split = 140
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

weights = np.zeros(3)
bias = 0.0
lr = 0.1

for epoch in range(50):
    logits = X_train @ weights + bias
    preds = sigmoid(logits)
    
    error = preds - y_train
    grad_w = (X_train.T @ error) / len(y_train)
    grad_b = error.mean()
    
    weights -= lr * grad_w
    bias -= lr * grad_b
    
    if epoch % 10 == 0 or epoch == 49:
        train_preds = (sigmoid(X_train @ weights + bias) > 0.5).astype(float)
        val_preds = (sigmoid(X_val @ weights + bias) > 0.5).astype(float)
        train_acc = (train_preds == y_train).mean()
        val_acc = (val_preds == y_val).mean()
        loss = -np.mean(y_train * np.log(preds + 1e-15) + (1 - y_train) * np.log(1 - preds + 1e-15))
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

print(f"\nLearned weights: {weights}")
print(f"True weights:   {true_weights}")
```

---

## Beat 4: Use It

### GTM Redirect: Lead Scoring Model Training

This training loop pattern is the same one used when training a binary classifier on enriched company data to predict ICP fit—the core mechanism behind Zone 2 lead scoring in [CITATION NEEDED — concept: GTM Zone 2 scoring model training]. The labels come from historical win/loss data. The features come from enrichment pipelines. The evaluation split prevents the model from memorizing your existing customer base instead of generalizing to new prospects.

Every scoring model in a GTM stack either trains this way internally or was pre-trained on different data and fine-tuned. The loss curve tells you whether your labeled examples contain enough signal. If validation accuracy plateaus at 55%, the features (enrichment fields) don't separate converters from non-converters. No amount of model architecture fixes bad input data.

---

## Beat 5: Practice

### Easy

Modify the learning rate to 0.01 and 1.0. Print the loss curves side by side. Observe which converges and which diverges.

### Medium

Add an early stopping condition: halt training if validation accuracy does not improve for 5 consecutive epochs. Print the epoch where training stopped.

### Hard

Implement a precision and recall calculation from the confusion matrix values (true positives, false positives, false negatives). Run it on the validation set. Explain why accuracy alone is misleading when the dataset is imbalanced (e.g., 90% negative class).

---

## Beat 6: Ship It

Export the trained weights and bias to a JSON file. Write a separate inference script that loads the weights, takes a single input array, and prints the prediction probability. This is the same deployment pattern used when shipping a scoring model to a Clay waterfall—weights stored, inference run per record. [CITATION NEEDED — concept: deploying custom scoring models in Clay workflows]

---

## Learning Objectives

1. **Implement** a training loop with forward pass, loss computation, gradient calculation, and parameter updates.
2. **Detect** overfitting by comparing training accuracy against validation accuracy across epochs.
3. **Configure** train/validation/test splits and explain the role of each partition.
4. **Evaluate** model performance using accuracy, precision, and recall on held-out data.
5. **Export** trained parameters for reuse in a separate inference script.