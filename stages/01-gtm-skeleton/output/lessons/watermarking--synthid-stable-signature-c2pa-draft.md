# Watermarking — SynthID, Stable Signature, C2PA

## Hook

AI-generated content is flooding outbound channels. Watermarking is the detection layer that lets you prove what's synthetic, what's modified, and what's original — before a prospect or regulator asks you to. Three approaches compete: statistical watermarking at generation time (SynthID), latent-space signatures fine-tuned into the model (Stable Signature), and cryptographic provenance chains bolted on after the fact (C2PA). Each makes a different tradeoff between robustness, stealth, and ecosystem adoption.

---

## Learn It

**Mechanism 1 — Statistical watermarking (SynthID pattern):** During generation, the model perturbs its logit distribution using a secret key. The perturbation is imperceptible to humans but statistically detectable by a keyed function that measures deviation from the expected distribution. The detector returns a confidence score, not a binary yes/no. The watermark survives cropping, color shifts, and moderate compression. It does not survive severe regeneration that reshuffles pixel statistics.

**Mechanism 2 — Latent-space binary signatures (Stable Signature pattern):** An image generator (typically diffusion-based) is fine-tuned to encode a fixed binary key into the latent representation. A separate extractor network recovers the key from any output image. Matching is Hamming distance between extracted and enrolled keys. Robustness comes from training the encoder-decoder pair against augmentation (resize, crop, JPEG). The key is tiny (typically 48–256 bits), so the signal is low but highly targeted.

**Mechanism 3 — Cryptographic provenance manifests (C2PA pattern):** Not a watermark in the signal-processing sense. C2PA attaches a signed JSON manifest to content (embedded in EXIF, sidecar file, or XMP). The manifest chains assertions (who generated, with what tool, when) using X.509 certificates. Verification checks signature validity and chain integrity. Fragile by design — any modification breaks the signature chain. Strength is legal traceability; weakness is that stripping the manifest is trivial if the attacker controls the file.

**Comparison axis:** stealth (can an adversary detect and remove it?), robustness (does it survive transformation?), adoption (who runs the infrastructure?), and cost (compute overhead at generation time).

---

## See It

**Demo 1 — Simulated statistical watermark detection (Python):** Generate synthetic logit scores for watermarked and unwatermarked text. Implement a simplified detection function that computes a z-score against the expected perturbation pattern. Print detection results for both samples.

**Demo 2 — Binary signature embedding and extraction (Python):** Encode a 48-bit message into an image's DCT coefficients. Extract the message after JPEG compression and resize. Print Hamming distance before and after transformation.

**Demo 3 — C2PA manifest structure inspection:** Parse a minimal C2PA-compliant JSON manifest. Verify signature validity using a toy RSA keypair. Print assertion chain and verification result.

---

## Do It

**Easy:** Modify the detection threshold in Demo 1. Observe how false positive and false negative rates shift. Print both rates at three threshold values.

**Medium:** In Demo 2, apply progressive JPEG compression (quality 90, 70, 50, 30). Plot Hamming distance vs. compression level. Report the quality level at which the signature breaks (Hamming distance > threshold).

**Hard:** Build a minimal provenance chain: generate an image, create a C2PA-style manifest with two assertions (created_by, modified_by), sign each assertion, and write a verifier that detects if a middle assertion was tampered with. Print "valid" or "tampered" for both clean and manipulated manifests.

---

## Use It

Content watermarking is foundational for **Zone 03 — Content & Enrichment** in the GTM topic map. Specifically:

- **Brand safety verification:** When your outbound pipeline generates hundreds of images or copy variants, watermarking lets you later verify which assets are yours — useful when competitors or scrapers repurpose your content. [CITATION NEEDED — concept: watermarking for brand asset tracking in outbound]
- **Compliance post-hoc:** EU AI Act requires disclosure of AI-generated content. C2PA manifests provide an auditable provenance trail if your stack needs to demonstrate compliance. SynthID and Stable Signature provide the detection capability without requiring the viewer to opt into a provenance standard.
- **This is not a Clay waterfall integration.** Watermarking operates at the model inference layer, not the enrichment layer. The GTM connection is operational risk mitigation, not workflow automation.

---

## Ship It

**Checkpoint questions (not a quiz bank — requires `docs/en.md` with objectives):**

1. Given a detection z-score of 2.1 and a threshold of 2.0, what is the risk of false positive if the null distribution is standard normal? (Compute and interpret.)
2. A Stable Signature extractor returns a 48-bit key with Hamming distance 3 from the enrolled key. Is this a match under a threshold of 5? What happens to the match confidence after the image is resized to 50%?
3. An image file has its C2PA manifest stripped and replaced with a different manifest signed by a different certificate. Does C2PA verification pass or fail? Why does this design choice matter for adversarial settings?

**Deliverable:** A Python script that takes an image path, checks for all three watermark types (statistical via a pre-computed score, binary signature via DCT extraction, C2PA manifest via EXIF parsing), and prints a provenance report. Must run in terminal with no browser dependency.

---

## Learning Objectives

1. Compare the mechanisms of statistical watermarking, latent-space signatures, and cryptographic provenance chains — and state which properties each preserves under adversarial transformation.
2. Implement a simplified statistical watermark detector that computes z-scores against a keyed perturbation pattern.
3. Encode and extract a binary signature from image frequency coefficients, measuring robustness under compression.
4. Construct and verify a C2PA-style manifest with chained assertions and detect tampering.
5. Evaluate which watermarking approach is appropriate for a given GTM content pipeline based on threat model, adoption requirements, and computational constraints.