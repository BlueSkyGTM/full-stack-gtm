# Audio Classification — From k-NN on MFCCs to AST and BEATs

## Learning Objectives

1. Extract MFCC features from audio signals and compute their dimensionality reduction properties across frame windows.
2. Implement k-NN classification on MFCC vectors and evaluate baseline accuracy using cross-validation.
3. Compare spectrogram-based transformer architectures (AST) against traditional feature pipelines on the same input signal.
4. Run zero-shot audio classification using a pre-trained AST model and retrieve top-k predictions with confidence scores.
5. Diagnose when a simple MFCC+classifier pipeline outperforms a transformer and articulate the latency, data, and accuracy tradeoff.

## The Problem

You have a 10-second audio clip. You need to answer one question: what is it? The "it" could be a dog bark, a fire alarm, a Portuguese speaker, a frustrated customer, or a competitor mention in a sales call. Every one of these tasks is audio classification — map a variable-length waveform to one or more discrete labels.

Two pipelines solve this problem. The first is 40 years old: extract hand-crafted spectral features called MFCCs, feed them to a simple classifier like k-NN or logistic regression. The second is 3 years old: convert the waveform to a log-mel spectrogram, split it into patches, and feed those patches through a Vision Transformer adapted for audio. The first pipeline runs on a CPU in milliseconds. The second requires a GPU and produces state-of-the-art accuracy on benchmarks like AudioSet. The gap between them — in accuracy, latency, data requirements, and engineering complexity — is your design space.

The hard part of audio classification is rarely the architecture. It is the data. Audio datasets suffer brutal class imbalance (thousands of "speech" clips, twenty "gunshot" clips), strong domain shift (studio recordings vs phone calls recorded in a moving car), and label noise (who decided that "restaurant babble" is meaningfully different from "crowd noise"?). The architecture choice matters, but data curation, augmentation, and honest evaluation drive 80% of your production accuracy.

## The Concept

### Three Representations, Three Classifiers

Every audio classifier operates on one of three input representations. Each representation encodes a different inductive bias — an assumption about what makes two sounds similar.

**MFCC vectors** compress a waveform into 13–40 coefficients per frame by simulating how the human cochlea filters sound into frequency bands, then applying a discrete cosine transform (DCT) to decorrelate those bands. The result is a compact vector where Euclidean distance correlates with perceived similarity. k-NN and logistic regression work well here because the feature space is small (13–40 dimensions) and the distance metric is meaningful.

**Log-mel spectrograms** preserve more information than MFCCs. They represent audio as a 2D image: time on one axis, frequency (on a perceptual mel scale) on the other, with pixel intensity indicating energy. A 2D CNN treats this as an image classification problem, learning local time-frequency patterns (e.g., the harmonic structure of a siren, the burst pattern of a clap). This was the dominant approach from 2015 to 2021.

**Spectrogram patches** split the log-mel spectrogram into fixed-size tiles (typically 16×16, borrowed directly from Vision Transformers) and let a transformer's attention mechanism decide which patches matter for the classification decision. Any patch can attend to any other patch — there is no locality constraint. This is the Audio Spectrogram Transformer (AST), which achieved 0.485 mAP on AudioSet using pure attention, no convolution, no recurrence.

```mermaid
flowchart TD
    WAV["Raw Waveform<br/>16kHz – 48kHz"] --> MEL["Mel Spectrogram<br/>(perceptual frequency scale)"]

    MEL --> DCT["Discrete Cosine Transform<br/>(decorrelate frequency bands)"]
    DCT --> MFCC["MFCCs: 13–40 coefficients<br/>per 25ms frame"]
    MFCC --> KNN["k-NN / Logistic Regression<br/>(CPU, < 1ms inference)"]

    MEL --> LOG["Log-Mel Spectrogram<br/>(time × frequency image)"]
    LOG --> PATCH["Split into 16×16 patches"]
    PATCH --> POS["Add positional embeddings"]
    POS --> ENC["Transformer Encoder<br/>(12 layers, 768 hidden)"]
    ENC --> CLS["CLS token → Softmax<br/>(GPU, ~50ms inference)"]

    WAV --> MASK["Masked audio prediction<br/>(self-supervised pre-training)"]
    MASK --> TOKEN["Learned audio tokenizer<br/>(BEATs)"]
    TOKEN --> FT["Frozen features +<br/>fine-tuned linear head"]
```

### AST: Vision Transformer, but for Sound

The Audio Spectrogram Transformer takes the exact architecture of ViT and applies it to log-mel spectrograms. The spectrogram — say, 1024 time steps × 128 mel bins — gets divided into 16×16 patches, yielding a sequence of flattened patch embeddings. Positional embeddings are added so the model knows where each patch came from in the time-frequency grid. A `[CLS]` token is prepended. The transformer encoder processes the full sequence with self-attention, and the final hidden state of the `[CLS]` token drives a linear classifier.

The key property: attention is global from layer 1. A CNN builds up receptive field gradually (layer 1 sees 3×3 patches, layer 5 sees larger regions). AST's attention mechanism can connect a patch at t=0.1s with a patch at t=9.8s in the first layer. For sounds where long-range temporal structure matters — a door opening followed by footsteps, a question followed by a pause — this helps. For sounds defined by local texture — a single drum hit — it adds computational cost without clear benefit.

### BEATs: Self-Supervised Pre-training with a Learned Tokenizer

BEATs (Bootstrap, Encode, Attend, Time-series) takes a different route to the same destination. Instead of supervised training on labeled audio, BEATs first learns a tokenizer through masked audio prediction: mask portions of the spectrogram, predict the masked content, and learn discrete audio tokens in the process. This tokenizer is learned without labels, trained on unlabeled audio. Once the tokenizer exists, BEATs is fine-tuned on labeled data with a standard classification head.

The practical advantage: BEATs learns general audio representations from massive unlabeled datasets (millions of hours), then transfers those representations to your specific task with minimal labeled data. If you have 100 labeled examples of your target sound, BEATs' frozen features + a linear probe will likely outperform an AST trained from scratch on the same 100 examples. If you have 100,000 labeled examples, the gap narrows or disappears.

### Where k-NN on MFCCs Still Wins

Transformers are not universally better. k-NN on MFCCs remains the right choice when: you have fewer than ~500 training samples per class (transformers overfit), you need sub-10ms latency on CPU (AST inference is ~50ms on GPU, >500ms on CPU), your task is simple phonetic or acoustic event distinction (dog vs cat, speech vs music), or you need full interpretability (you can show which training examples the classifier matched). The transformer wins when classes are defined by complex temporal structure, your dataset is large, and you have GPU budget.

## Build It

### Step 1: Extract MFCCs and Inspect the Feature Space

MFCC extraction is a three-step pipeline: compute the short-time Fourier transform (STFT) to get a spectrogram, convert to the mel scale to approximate human frequency perception, then apply a DCT to decorrelate the mel coefficients. The output is a matrix of shape `(n_mfcc, n_frames)` where `n_mfcc` is typically 13–40 and `n_frames` depends on audio length and hop size.

```python
import librosa
import numpy as np

y, sr = librosa.load(librosa.ex('trumpet'), sr=22050)

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)

print(f"Waveform samples: {y.shape[0]}")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {y.shape[0] / sr:.2f} seconds")
print(f"MFCC shape: {mfccs.shape}")
print(f"Frames: {mfccs.shape[1]}")
print(f"Compression ratio: {y.shape[0]} samples -> {mfccs.size} coefficients ({mfccs.size / y.shape[0] * 100:.1f}%)")
print(f"\nMean MFCC vector (13 coefficients):")
print(mfccs.mean(axis=1).round(2))
```

This produces output showing the dimensionality reduction: a ~6-second trumpet recording at 22.05kHz has ~130,000 samples compressed to a 13×~250 matrix — roughly 3,250 values, a 97% reduction. The mean MFCC vector captures the spectral envelope of the sound.

### Step 2: Build a k-NN Classifier on MFCC Features

To classify with MFCCs, you need a fixed-length vector per clip. The standard approach: average MFCCs across the time axis to get one 13-dimensional vector per recording, then run k-NN.

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)

def generate_mfcc_like(class_center, n_samples, n_mfcc=13):
    return np.random.randn(n_samples, n_mfcc) * 0.5 + class_center

trumpet = generate_mfcc_like(np.array([3, -2, 1, 0.5, -0.3, 0.2, 0, 0, 0, 0, 0, 0, 0]), 60)
drum = generate_mfcc_like(np.array([-1, 4, -2, 1, -0.5, 0.3, 0.1, 0, 0, 0, 0, 0, 0]), 60)
speech = generate_mfcc_like(np.array([0, 0, 3, -1, 2, -1, 0.5, 0.2, 0, 0, 0, 0, 0]), 60)

X = np.vstack([trumpet, drum, speech])
y = np.array(['trumpet'] * 60 + ['drum'] * 60 + ['speech'] * 60)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
])

scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print(f"5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"Per-fold scores: {scores.round(3)}")

pipe.fit(X, y)
test_clip = generate_mfcc_like(np.array([2.8, -1.8, 0.9, 0.4, -0.2, 0.1, 0, 0, 0, 0, 0, 0, 0]), 1)
prediction = pipe.predict(test_clip)
print(f"Test clip predicted as: {prediction[0]}")
print(f"Neighbor distances: {pipe.named_steps['knn'].kneighbors(pipe.named_steps['scaler'].transform(test_clip))[0].round(3)}")
```

This builds a complete classification pipeline: feature standardization (critical for distance-based methods), k-NN with k=5, and 5-fold cross-validation. The output confirms the classifier works and shows which training examples are nearest to the test clip — the interpretability advantage of k-NN over a transformer.

### Step 3: Run AST Zero-Shot on the Same Signal

Now load a pre-trained AST and classify the same trumpet recording. The model was trained on AudioSet, a dataset of 2 million YouTube clips labeled with 527 sound classes. It has never seen your specific clip, but it has learned general audio patterns.

```python
import torch
from transformers import pipeline, AutoFeatureExtractor, ASTForAudioClassification
import librosa
import warnings
warnings.filterwarnings('ignore')

y, sr = librosa.load(librosa.ex('trumpet'), sr=16000)

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = ASTForAudioClassification.from_pretrained(model_name)

inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.topk(logits, k=5, dim=-1).indices[0].tolist()
probs = torch.softmax(logits, dim=-1)[0]

print("AST top-5 predictions:")
for idx in predicted_class_ids:
    label = model.config.id2label[idx]
    print(f"  {label}: {probs[idx].item():.4f}")

print(f"\nInput spectrogram shape fed to AST: {inputs['input_values'].shape}")
print(f"AST parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

The AST processes the raw waveform through the feature extractor (which computes the log-mel spectrogram internally), patches it, and produces logits over 527 AudioSet classes. The output shows the model's confidence distribution. Note the parameter count — ~86M for the base AST, compared to zero parameters for MFCC extraction and k parameters for k-NN.

## Use It

The embedding-based classification that routes inbound leads in a Signal Machine (Zone 06) follows the same architecture pattern as audio classification: raw input → feature extraction → classifier → routing decision. In text-based GTM, Claygent classifies a company as "product vs service" or "enterprise vs SMB" using GPT via OpenAI API [CITATION NEEDED — concept: Clay classification pipeline for company typing]. In audio-based GTM, the pipeline classifies call recordings: does this segment contain a competitor mention, a pricing objection, or a buying signal?

Conversation intelligence platforms — Gong, Chorus, Fireflies — run audio classification on every minute of every sales call. The classification target depends on the GTM motion. For inbound-led outbound (the Zone 06 use case), you classify calls to detect which inbound conversations indicate high buying intent, then route those accounts into priority sequences before they go cold. The "Signal Machine" pattern: audio classification detects the signal, the classification output triggers a workflow, the workflow routes the lead.

The mechanism is identical to the k-NN vs AST tradeoff you just built. A simple MFCC + logistic regression pipeline can classify "silence vs speech vs loud noise" in under 1ms per frame on a CPU — fast enough to run real-time call quality monitoring on thousands of concurrent calls. An AST model classifies "competitor mention vs pricing objection vs buying signal" with higher accuracy but requires GPU inference and a 50ms+ budget per segment. Most production systems layer both: the cheap classifier gates which segments get sent to the expensive classifier.

```python
import librosa
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)

def simulate_call_segment_features(segment_type, n=50):
    base = {
        'silence': np.zeros(13),
        'agent_talking': np.array([2, -1, 0.5, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'prospect_talking': np.array([-1, 2, -0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        'overlap_shouting': np.array([4, 3, 2, 1, 0.5, 0.3, 0.2, 0, 0, 0, 0, 0, 0]),
    }
    return np.random.randn(n, 13) * 0.3 + base[segment_type]

X = np.vstack([
    simulate_call_segment_features('silence'),
    simulate_call_segment_features('agent_talking'),
    simulate_call_segment_features('prospect_talking'),
    simulate_call_segment_features('overlap_shouting'),
])
y_labels = (
    ['silence'] * 50 + ['agent_talking'] * 50 +
    ['prospect_talking'] * 50 + ['overlap_shouting'] * 50
)

gate = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7))
])
gate.fit