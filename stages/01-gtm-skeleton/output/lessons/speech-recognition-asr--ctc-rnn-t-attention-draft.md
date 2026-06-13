# Speech Recognition (ASR) — CTC, RNN-T, Attention

---

## Beat 1: Hook

Audio is the only modality where the input length wildly exceeds the output length — a 10-second clip produces ~1,000 acoustic frames but only ~20 tokens of text. Three mechanisms solve this alignment problem: CTC collapses blanks, RNN-T streams token-by-token, and attention learns soft alignment. Pick wrong and your voice agent has 3-second latency. Pick right and it responds in real-time.

---

## Beat 2: Concept

**CTC (Connectionist Temporal Classification):** Introduces a blank token and a collapse rule — consecutive identical tokens merge, blanks delete. The model emits one token per acoustic frame, then the collapse function reduces the sequence. Training marginalizes over all valid alignments using forward-backward. Limitation: assumes output tokens are conditionally independent given the alignment — no explicit language model in the decoder. [CITATION NEEDED — concept: CTC loss gradient derivation]

**RNN-T (Recurrent Neural Network Transducer):** Extends CTC by replacing the independence assumption with a autoregressive prediction network (a language model over previously emitted tokens). The transducer computes a joint network over encoder states and prediction network states, producing a lattice of possible paths. Enables streaming because each token decision uses only past context. [CITATION NEEDED — concept: RNN-T forward algorithm complexity]

**Attention-based ASR:** Encoder-decoder architecture where the decoder attends over the full encoder output at each decoding step. Attention weights are learned — no monotonic alignment constraint. Produces higher accuracy than CTC/RNN-T on non-streaming benchmarks but requires the full utterance before decoding begins. Listen, Attend, Spell (LAS) is the canonical architecture. [CITATION NEEDED — concept: monotonic attention variants for streaming]

**Comparison axis:** latency (CTC/RNN-T stream, attention doesn't), accuracy (attention > RNN-T > CTC typically), training complexity (CTC simplest, RNN-T most complex).

---

## Beat 3: Demo

Implement CTC collapse from scratch on a toy example, then run Whisper (attention-based) and observe the alignment patterns in cross-attention weights.

```
import torch

def ctc_collapse(tokens):
    collapsed = []
    prev = None
    for t in tokens:
        if t != prev and t != 0:
            collapsed.append(t)
        prev = t
    return collapsed

raw = [1, 1, 0, 0, 2, 2, 0, 3, 3, 3, 0, 0]
collapsed = ctc_collapse(raw)
print(f"Raw frames:      {raw}")
print(f"After CTC merge: {collapsed}")
```

Exercise hooks:
- **Easy:** Run CTC collapse on 5 hand-crafted sequences and verify output.
- **Medium:** Generate a random logit matrix (T=50, V=5), apply argmax per frame, collapse, and count how many blank-only frames exist.
- **Hard:** Implement the CTC forward algorithm for a single training example and verify the loss equals `-log(sum of all valid path probabilities)`.

---

## Beat 4: Use It

GTM Redirect: **Zone 1 — Voice Agent Pipeline** (conversation intelligence / AI SDR voice calls).

ASR sits at the front of every voice agent pipeline. Latency budget for a sales call bot: end-to-end < 800ms. Of that, ASR typically gets 200ms. RNN-T architectures are the mechanism behind streaming ASR systems (Whisper streaming mode, various real-time APIs). CTC models are the backbone of on-device dictation because they're cheap to run. Attention-based models handle post-call transcription where latency doesn't matter but accuracy does — Gong, Chorus, and similar conversation intelligence tools process recordings this way.

Exercise hooks:
- **Easy:** Record a 10-second audio clip, transcribe with Whisper, print the token-level timestamps to observe alignment.
- **Medium:** Compare streaming vs. non-streaming transcription latency on the same audio clip and plot the latency difference.
- **Hard:** Build a real-time transcription loop that reads audio chunks from a microphone, sends each chunk to a streaming ASR endpoint, and displays partial transcripts with `<200ms` per-chunk latency.

---

## Beat 5: Ship It

Production ASR selection is a latency/accuracy/cost tradeoff:

| Constraint | Mechanism | Implementation |
|---|---|---|
| On-device, <100ms | CTC | Vosk, sherpa-onnx |
| Streaming cloud, <300ms | RNN-T or streaming attention | Various cloud ASR APIs |
| Batch transcription, max accuracy | Full attention | Whisper large-v3 |

Deploying Whisper in production: batch requests, pad inputs to fixed lengths for GPU utilization, cache the encoder output if the same audio is re-transcribed. For streaming RNN-T: maintain encoder state across chunks, implement endpointing (detect when the speaker pauses to trigger finalization). [CITATION NEEDED — concept: ASR endpointing algorithms]

Exercise hooks:
- **Easy:** Deploy Whisper via a FastAPI endpoint and benchmark single-request latency for audio clips of 5s, 30s, and 120s.
- **Medium:** Implement a batching wrapper that queues incoming audio and processes in batches of 8 — measure throughput improvement.
- **Hard:** Build an endpointing detector that fires after 600ms of silence, then triggers final transcript emission. Evaluate false-trigger rate on a dataset with pauses.

---

## Beat 6: Gotchas

- **CTC peaky behavior:** CTC-trained models often emit long blank runs then spike. This is a known artifact of the loss landscape, not a bug. It can cause instability in streaming if endpointing relies on non-blank frame density.
- **RNN-T training instability:** The transducer loss landscape has more local minima than CTC. Training often requires careful learning rate scheduling. [CITATION NEEDED — concept: RNN-T training tricks and convergence]
- **Attention misalignment on long utterances:** Attention-based models can skip or repeat segments on audio >30s. Whisper mitigates this with chunked decoding but it's still observable.
- **Language model confusion:** CTC and RNN-T are often paired with external language models (LM fusion). The fusion weight is sensitive — too high and rare words get suppressed, too low and acoustic errors leak through.
- **Numerical stability in CTC loss:** The forward-backward algorithm operates in log-space to avoid underflow. Naive implementations will produce `-inf` losses on long sequences. Use the log-sum-exp trick consistently.

Exercise hooks:
- **Easy:** Run CTC collapse on a sequence that should produce "hello" and on a subtly wrong sequence — observe how a single frame error changes output.
- **Medium:** Transcribe 10 clips from a noisy dataset (e.g., restaurants) and count substitution errors per phoneme category — report which phonemes are most confused.
- **Hard:** Implement a minimal external LM rescoring function that takes CTC beam search outputs, scores each with a simple n-gram LM, and reranks. Measure WER improvement.