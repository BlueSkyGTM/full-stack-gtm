# CNNs and RNNs for Text

## Learning Objectives

1. Implement a 1D convolutional layer over token sequences and extract n-gram features
2. Build an RNN cell that maintains hidden state across a text sequence
3. Compare the inductive biases of CNNs (local pattern detection) vs RNNs (sequential state)
4. Diagnose when a CNN or RNN is the wrong choice for a given text task
5. Evaluate both architectures on the same classification task and interpret the performance gap

---

## Beat 1: Hook

**Convolution and recurrence are competing hypotheses about what matters in text.** A CNN bets that local n-gram patterns carry the signal. An RNN bets that order-dependent accumulation of meaning matters. The practitioner's job is to figure out which bet pays off for their specific task — or whether neither does and a transformer is what they actually need.

---

## Beat 2: Concept

**Mechanism breakdown of both architectures on token sequences:**

- **1D Convolution over text:** A filter of width *k* slides across a token embedding matrix, producing one scalar per position. This is n-gram detection — the filter learns to fire when a specific contiguous pattern appears. Stacking filters gives you a bag of learned n-grams. Max-pooling collapses the position dimension, losing order information entirely. This is why CNNs classify well but generate poorly.

- **RNN sequential processing:** At each timestep *t*, the RNN takes the current token embedding *x_t* and the previous hidden state *h_{t-1}*, and computes *h_t = f(x_t, h_{t-1})*. The hidden state is a lossy compression of everything seen so far. Vanishing gradients make this compression increasingly lossy over long sequences — hence LSTMs (gated memory) and GRUs (simplified gating).

- **The key difference:** CNNs see all positions simultaneously through parallel convolution. RNNs see positions sequentially, with each step dependent on the previous. CNNs are fast to train but position-agnostic after pooling. RNNs are slow to train but carry sequential information to the final hidden state.

---

## Beat 3: Demonstration

**Working PyTorch code that trains both a CNN and an LSTM on the same synthetic text classification task and prints:**

- Model architecture summaries
- Per-epoch loss for both models
- Final accuracy comparison
- Inference time comparison

Code requirements:
- Generates synthetic "sentiment" data (short sequences of random tokens with label-correlated patterns)
- No external datasets or downloads
- Runs in terminal, prints all output
- No comments in code

Exercise hooks:
- **Easy:** Modify the CNN filter width and observe accuracy change
- **Medium:** Replace the LSTM with a GRU and compare training speed
- **Hard:** Implement a bidirectional LSTM and explain why accuracy changes (or doesn't)

---

## Beat 4: Use It

**GTM Redirect: Zone 1 — Signals [CITATION NEEDED — concept: exact GTM zone mapping for text classification models in GTM topic map]**

Text classification is the backbone of intent signal detection. When a prospect sends an email, posts a question, or fills a form, something needs to classify that text into "sales-ready signal" vs "noise."

- **CNN use case:** Short-text classification where local keyword patterns predict intent — "looking for a solution like [product]" is a 5-gram that a CNN filter can learn to fire on. Fast inference, good for real-time routing.

- **RNN use case:** Longer conversational text where the meaning depends on what came before — a prospect saying "we tried X and it didn't work, now we need Y" requires sequential processing to capture the contrast.

- **When neither fits:** If your GTM system needs to understand long documents (RFPs, contracts), both CNNs and RNNs will underperform compared to transformer-based approaches. The honest take: for most modern GTM text tasks, you should be using a pre-trained transformer, not building CNNs or RNNs from scratch. This lesson exists so you understand what those transformers replaced and why.

Exercise hooks:
- **Easy:** Classify synthetic prospect messages with both models and compare
- **Medium:** Tokenize real email subject lines (provided as a list in code) and run CNN classification
- **Hard:** Build a simple "intent score" pipeline that outputs a float from 0–1 for "sales readiness"

---

## Beat 5: Ship It

**Deploy a text classifier that reads from stdin and outputs a label + confidence score.**

Both models (CNN and LSTM) are serialized. A CLI script loads one, reads text input, and prints classification output. The practitioner picks which model to deploy based on the Beat 4 analysis of their specific use case.

Constraints:
- Must run as `python classify.py --model cnn "text to classify"`
- Must print label and confidence score
- Must load model weights from a file (saved during Beat 3 training)
- No browser dependency, no API calls

Exercise hooks:
- **Easy:** Ship the CNN classifier with a fixed threshold
- **Medium:** Ship both models behind a `--model` flag and include a benchmark script that compares them on 100 inputs
- **Hard:** Add a `--timing` flag that profiles inference latency and prints a recommendation ("use CNN for <50ms requirement, use LSTM for higher accuracy requirement")

---

## Beat 6: Evaluate

**Assessment anchors (not a quiz bank — must be backed by `docs/en.md` objectives):**

1. **Mechanism question:** Given a 1D CNN with filter width 3 over a 10-token sequence, how many convolution operations occur before padding? What does each operation detect? (Maps to Objective 1)

2. **Architecture decision question:** A GTM system needs to classify support tickets (200+ words) into urgency tiers. You have a CNN and an LSTM trained on the same data. The CNN achieves 82% accuracy in 2ms inference. The LSTM achieves 84% accuracy in 15ms inference. Which do you deploy and what is the specific reasoning? (Maps to Objective 4)

3. **Debugging question:** An RNN trained on prospect email sequences has 60% accuracy on training data after 50 epochs and 58% on validation data. A CNN on the same data reaches 88% training accuracy and 85% validation accuracy. What does this gap indicate about the data structure? (Maps to Objective 3)

4. **Implementation question:** Write the forward pass of an LSTM cell (no library usage) given input *x_t* and previous state *(h_{t-1}, c_{t-1})* with weight matrices *W_f, W_i, W_o, W_c*. Print each gate activation. (Maps to Objective 2)

5. **Evaluation question:** Both models are tested on a held-out set of 500 intent-labeled messages. The CNN has higher precision but lower recall than the LSTM. For a GTM system that routes hot leads to sales, which metric matters more and why? (Maps to Objective 5)

---

## GTM Redirect Rules (Summary)

- **Beat 4 ("Use It")** redirects to Zone 1 Signals: text classification for intent detection [CITATION NEEDED — concept: exact GTM zone mapping for text classification models in GTM topic map]
- **Beat 5 ("Ship It")** deploys a classifier that could plug into a GTM enrichment pipeline
- **Honest caveat:** For production GTM text tasks in 2024+, transformers are the default. CNNs and RNNs are taught here as mechanism prerequisites, not deployment recommendations.