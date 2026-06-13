# Audio Evaluation — WER, MOS, UTMOS, MMAU, FAD, and the Open Leaderboards

---

## Beat 1: Hook — You Can't Ship Voice AI Without a Score

Audio evaluation is fragmented across five+ metrics that each measure different failure modes. A TTS system with a low FAD can still produce unintelligible speech. A speech-to-text model with 2% WER can fail catastrophically on domain-specific jargon. Practitioners need to know which metric catches which failure mode, and which ones can be automated versus which still require human ears.

---

## Beat 2: Concept — The Metric Taxonomy

Six metrics, organized by what they actually measure:

- **WER (Word Error Rate):** Measures transcription accuracy. Edit distance (insertions, deletions, substitutions) divided by total words in reference. Still the standard for ASR evaluation.
- **MOS (Mean Opinion Score):** Human-rated audio quality on a 1–5 scale. Ground truth for perceived quality, but expensive and slow. No automation possible by definition.
- **UTMOS:** A learned predictor that estimates MOS from audio without human raters. Trained on Voice MOS Challenge data. Correlates ~0.83 with human MOS on out-of-domain data [CITATION NEEDED — concept: UTMOS correlation coefficient on benchmark].
- **MMAU (Multi-dimensional Audio Understanding Assessment):** Evaluates audio-language models across reasoning, comprehension, and generation tasks. LLM-style benchmark adapted for audio.
- **FAD (Fréchet Audio Distance):** Distributional metric for audio generation quality. Embeds generated and reference audio using a pretrained model (typically VGGish or PANNs), computes Fréchet distance between Gaussian fits of those embeddings. Lower = closer to reference distribution.
- **Open Leaderboards (Hugging Face, AudioBench, etc.):** Aggregated rankings. Useful for relative positioning, dangerous for absolute quality claims. Leaderboard gaming is real.

---

## Beat 3: Mechanism — How Each Metric Computes Its Score

**WER computation:**
Dynamic programming alignment (Levenshtein distance at word level). Three error types: substitution (wrong word), deletion (missed word), insertion (hallucinated word). Formula: `(S + D + I) / N` where N = reference word count. Case-normalized, punctuation-stripped before computation. Weakness: treats all words equally — "the" and "pharmaceutical" count the same.

**FAD computation:**
1. Pass reference corpus through embedding model → distribution μ_ref, Σ_ref
2. Pass generated corpus through same model → distribution μ_gen, Σ_gen
3. Compute Fréchet distance: `||μ_ref - μ_gen||² + Tr(Σ_ref + Σ_gen - 2(Σ_ref·Σ_gen)^½)`
4. Embedding model choice determines what "quality" means — VGGish captures perceptual similarity, PANNs capture audio event semantics.

**UTMOS mechanism:**
Fine-tuned speech model (typically wav2vec 2.0 backbone) trained to predict human MOS scores. Input: raw audio. Output: scalar estimate 1–5. The model learns acoustic features that correlate with human quality judgments — it is a proxy, not a ground truth.

**MMAU mechanism:**
Multiple-choice and open-ended evaluation across audio tasks. Scores aggregated per-dimension (reasoning, temporal understanding, etc.). Model generates response, compared to reference answers using LLM-as-judge or exact match depending on task type.

---

## Beat 4: Use It — Running Audio Evaluation Pipelines

Build evaluation scripts that compute WER from reference/hypothesis pairs, calculate FAD between two audio directories, and run UTMOS inference on generated samples. Includes working code using `jiwer` for WER and `frechet_audio_distance` for FAD, with printed outputs confirming each score.

*Exercise hooks:*
- **Easy:** Compute WER for five provided reference/hypothesis transcription pairs and print per-sample and aggregate scores.
- **Medium:** Generate synthetic speech samples with varying noise levels, compute FAD against a clean reference set, and plot quality degradation.
- **Hard:** Build a composite evaluation pipeline that runs WER + UTMOS + FAD on a TTS output directory and produces a single JSON report with pass/fail thresholds.

*GTM redirect:* Voice AI in GTM — conversation intelligence platforms (Gong, Chorus), AI SDR voice agents, and call coaching tools all require audio quality evaluation. This is foundational for Zone 3 (Engagement) voice agent deployments where unintelligible TTS directly kills conversion rates.

---

## Beat 5: Ship It — Evaluation as a CI Gate

Integrate audio metrics into deployment pipelines. Define threshold-based quality gates: WER < 5% for ASR, FAD < 2.0 for TTS, UTMOS > 3.5 for perceptual quality. Run evaluation on every model update against a held-out test set. Log metrics to a tracking system. Working code demonstrates a pytest-compatible evaluation harness that fails the build when thresholds are breached.

*Exercise hooks:*
- **Easy:** Write a pytest module that asserts WER on a test set stays below a configurable threshold.
- **Medium:** Create a GitHub Actions workflow step that runs FAD evaluation on a TTS model artifact and posts results as a PR comment.
- **Hard:** Implement a regression detector that compares current evaluation scores against the previous commit's scores and flags any metric degradation exceeding 10%.

*GTM redirect:* Shipping voice AI into production GTM workflows (Zone 3 voice agents, Zone 2 conversation intelligence) requires automated quality gates. A regression in TTS quality or ASR accuracy that reaches live calls is a revenue-impacting incident.

---

## Beat 6: Extend It — Leaderboards, Gaming, and the Frontiers

Open leaderboard dynamics: what they actually measure (relative model ordering, not absolute quality), how training data contamination inflates scores, and why leaderboard rank ≠ production performance. Reading Hugging Face audio leaderboard entries and understanding which benchmarks each model was evaluated on. The gap between benchmark performance and real-world domain-specific audio (accent coverage, domain jargon, noisy environments).

*Exercise hooks:*
- **Easy:** Pick three models from a public audio leaderboard, compare their reported metrics, and identify which evaluation dimensions each optimizes for.
- **Medium:** Replicate a subset of MMAU evaluation on two open audio-language models and compare your scores to reported leaderboard numbers.
- **Hard:** Design an evaluation suite for a specific GTM domain (sales call transcription, voicemail generation, or meeting summarization) that captures failure modes not measured by standard benchmarks.

*GTM redirect:* Foundational for Zone 3. Leaderboard scores do not predict performance on GTM-specific audio (sales calls with domain jargon, accented speech in global SDR outreach, compressed voicemail audio). Custom evaluation on domain data is mandatory before deploying voice AI in any GTM pipeline.