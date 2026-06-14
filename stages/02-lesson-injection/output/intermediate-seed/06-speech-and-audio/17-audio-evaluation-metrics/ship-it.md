## Ship It

Shipping voice AI into a production GTM stack requires an evaluation pipeline that runs continuously, not just at model selection time. The pipeline needs three layers.

**Layer 1 — Automated gate (every build).** Run WER on a held-out domain-specific test set and UTMOS on generated audio samples. Set thresholds: WER below your domain target (8% for conversational, 3% for dictation), UTMOS above 3.5. These run on every commit to your model branch. A regression triggers a block.

**Layer 2 — Distributional check (weekly).** Run FAD against a fixed reference corpus of real customer audio. This catches mode collapse and distribution drift that sample-level metrics miss. If your TTS model was fine-tuned on a new speaker and started producing a narrower pitch range, FAD will spike before any individual sample sounds wrong.

**Layer 3 — Human panel (per release).** MOS with five to ten raters on a stratified sample. This is the ground truth check that catches failures no automated metric predicts — the profane phoneme, the uncanny valley prosody, the accent that sounds wrong to a native speaker. You cannot skip this layer for customer-facing releases.

The leaderboards serve a different purpose in shipping: competitive positioning, not quality assurance. If you are building a voice product and the Hugging Face Open ASR Leaderboard shows three models within 0.3% WER of each other, the leaderboard is telling you that ASR quality is no longer your differentiator — latency, cost, domain adaptation, or integration depth is. Conversely, if your model is 5 WER points behind the frontier, the leaderboard is telling you to use an API instead of self-hosting until you close the gap.

Leaderboard gaming is a real risk for absolute quality claims. Models are sometimes trained on the test set distribution. Some leaderboards allow selective reporting on favorable subsets. Treat a model's leaderboard rank as a signal that it is worth evaluating on your own data, not as proof it will work on your data. The only WER that matters is the one computed on your customers' audio.