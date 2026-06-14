# Multimodal Evaluation

## Learning Objectives

- Compute CLIPScore between image-caption pairs using a pretrained vision-language encoder and interpret the resulting alignment signal.
- Implement CIDEr from token-level n-gram statistics to measure caption quality against human references.
- Build an LLM-as-judge evaluation harness that scores multimodal outputs against a structured rubric and returns parseable verdicts.
- Compare modality-specific metrics (BLEU, CIDEr) against cross-modal alignment scores (CLIPScore) on the same eval set and reason about when each is appropriate.
- Construct a weighted multi-metric pipeline that combines retrieval, generation, and judge-based scores into a single decision signal for production gating.

## The Problem

You trained a vision-language model. Loss went down. Now someone asks: is it any good? For a text-only model, you have a shelf of standard answers — exact match, BLEU, ROUGE, F1. These metrics are cheap, well-understood, and map cleanly to what humans consider "correct." A generated token either matches the reference or it doesn't.

Multimodal models break this convenience. When a model generates a caption for an image, "correct" becomes a judgment call. The caption "A dog runs through a field" might be accurate for an image where the reference caption says "A golden retriever sprints across a grassy park." Both are correct; a rigid string match says they share zero words in sequence. BLEU-4 would give this a low score despite the semantic overlap. Meanwhile, a model that outputs the exact reference caption word-for-word might still be wrong if the image has changed — BLEU doesn't look at the image at all.

This is the core tension: modality-specific metrics (BLEU, ROUGE, CIDEr) measure text quality without considering the image, while cross-modal metrics (CLIPScore, retrieval recall) measure alignment without judging text fluency. Neither alone is sufficient. A model can produce fluent captions that describe the wrong image (high BLEU, low CLIPScore) or accurate but ungrammatical descriptions (high CLIPScore, low BLEU). In a go-to-market enrichment pipeline, the first failure mode means you extract the wrong signal from a prospect's website. The second means your downstream formatting breaks. You need both axes.

The third evaluation surface — LLM-as-judge — exists because some qualities are genuinely subjective and no formula captures them. "Does this caption capture the sentiment of this image?" is not a question n-gram overlap can answer. A strong language model can rate captions against a rubric, handling nuance that formulaic metrics cannot. The cost is latency, expense, and non-determinism. The benefit is that you can evaluate qualities you cannot formally define.

## The Concept

Three families of metrics cover the multimodal evaluation landscape. Each family measures something the others cannot.

**Modality-specific metrics** operate on one modality at a time, ignoring the other. BLEU-4 computes the geometric mean of 1-gram through 4-gram precision between generated and reference text, multiplied by a brevity penalty that penalizes short outputs. CIDEr weights n-grams by their TF-IDF scores across a reference corpus, rewarding words that are specific to this image rather than generic ("a," "the," "image"). ROUGE focuses on recall — how much of the reference appears in the generation. All three treat text as sequences of tokens. They never see the image. In a GTM context, these metrics tell you whether the text your vision-language model extracted from a prospect's homepage reads well, regardless of whether it actually describes what's on the page.

**Cross-modal alignment metrics** check whether text and image actually correspond. CLIPScore is the canonical example: encode the image and the caption through a CLIP model, compute cosine similarity between the two embeddings, and scale by 100. A high score means the text and image live in the same region of the shared embedding space. This catches the failure mode where a model produces a grammatically perfect caption describing the wrong scene. Retrieval metrics like Recall@K work on the same embeddings but frame the task as ranking: given a caption, does the correct image appear in the top K nearest neighbors? This is the metric that matters when your enrichment pipeline needs to match a prospect's logo image to the right company record.

**Judge-based metrics** use an LLM to evaluate outputs against a rubric. The judge sees the image, the generated caption, and optionally a reference caption, then assigns a score or preference. This is the only family that can evaluate subjective qualities like helpfulness, tone, or completeness. In enrichment pipelines, this is how you'd evaluate whether a vision-language model correctly identified that a prospect's website shows "enterprise pricing tiers" versus "free trial" — a judgment call that n-grams and cosine similarity cannot make.

```mermaid
flowchart TD
    A[Multimodal Output] --> B{Evaluation Family}
    B --> C[Modality-Specific]
    B --> D[Cross-Modal Alignment]
    B --> E[Judge-Based]
    
    C --> C1[BLEU-4: n-gram precision]
    C --> C2[CIDEr: TF-IDF weighted n-grams]
    C --> C3[ROUGE: n-gram recall]
    C --> C4[Measures: text quality only]
    C4 --> C5[Blind to image content]
    
    D --> D1[CLIPScore: cosine similarity]
    D --> D2[Recall@K: ranking]
    D --> D3[Measures: image-text correspondence]
    D3 --> D4[Blind to text fluency]
    
    E --> E1[LLM sees image + text + rubric]
    E --> E2[Returns score or preference]
    E --> E3[Measures: subjective quality]
    E3 --> E4[Expensive, non-deterministic]
    
    C5 --> F[Use together: no single metric is sufficient]
    D4 --> F
    E4 --> F
    
    F --> G[Production Gate: weighted combination]
```

The decision tree is practical. If your task is text generation (captioning, description extraction), you need BLEU or CIDEr to check fluency and CLIPScore to check alignment. If your task is retrieval (matching images to records, finding the right logo), you need Recall@K on a shared embedding space. If your task involves judgment (does this page show enterprise pricing?), you need an LLM-as-judge. Most real pipelines need two of these three families running in parallel.

## Build It

This script computes three evaluation surfaces on synthetic data. It requires `torch`, `transformers`, `nltk`, and `Pillow`. Install with `pip install torch transformers nltk pillow` if needed. The script downloads a CLIP model on first run (~600MB) and uses NLTK's punkt tokenizer.

```python
import math
import torch
from transformers import CLIPProcessor, CLIPModel
from collections import Counter
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_score(image, text):
    inputs = clip_processor(
        text=[text], images=image, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    cosine = (image_features * text_features).sum(dim=-1)
    score = cosine.item() * 2.5
    
    return score

def modified_precision(candidate, references, n):
    candidate_ngrams = Counter(
        [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
    )
    
    max_ref_counts = Counter()
    for ref in references:
        ref_ngrams = Counter(
            [tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)]
        )
        for ngram, count in ref_ngrams.items():
            max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
    
    clipped_count = 0
    total_count = 0
    for ngram, count in candidate_ngrams.items():
        clipped_count += min(count, max_ref_counts.get(ngram, 0))
        total_count += count
    
    if total_count == 0:
        return 0.0
    return clipped_count / total_count

def compute_bleu4(candidate, references):
    candidate = candidate.lower().split()
    references = [r.lower().split() for r in references]
    
    precisions = []
    for n in range(1, 5):
        precisions.append(modified_precision(candidate, references, n))
    
    if min(precisions) == 0:
        return 0.0
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
    
    bp = min(1.0, math.exp(1 - len(references[0]) / len(candidate)))
    
    return bp * geometric_mean

def compute_cider(candidate, references):
    candidate_tokens = candidate.lower().split()
    ref_tokens_list = [r.lower().split() for r in references]
    
    def get_ngram_tfidf(tokens, n, df_counts, N):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return {}
        counter = Counter(ngrams)
        total = sum(counter.values())
        result = {}
        for ngram, count in counter.items():
            tf = count / total
            df = df_counts.get(ngram, 1)
            idf = math.log(N / df)
            result[ngram] = tf * idf
        return result
    
    all_docs = [candidate_tokens] + ref_tokens_list
    N = len(all_docs)
    
    df_counts = {}
    for n in range(1, 5):
        for doc in all_docs:
            ngrams = set(tuple(doc[i:i+n]) for i in range(len(doc) - n + 1))
            for ng in ngrams:
                df_counts[ng] = df_counts.get(ng, 0) + 1
    
    candidate_vecs = {}
    ref_vecs_list = []
    for n in range(1, 5):
        candidate_vecs[n] = get_ngram_tfidf(candidate_tokens, n, df_counts, N)
    
    for ref_tokens in ref_tokens_list:
        ref_vec = {}
        for n in range(1, 5):
            ref_vec[n] = get_ngram_tfidf(ref_tokens, n, df_counts, N)
        ref_vecs_list.append(ref_vec)
    
    def cosine_sim(vec1, vec2):
        dot = sum(v1 * vec2.get(k, 0) for k, v1 in vec1.items())
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    scores = []
    for ref_vec in ref_vecs_list:
        n_scores = []
        for n in range(1, 5):
            cs = cosine_sim(candidate_vecs[n], ref_vec[n])
            n_scores.append(cs)
        avg = sum(n_scores) / len(n_scores)
        scores.append(avg)
    
    return max(scores) * 10

width, height = 224, 224
red_image = Image.new("RGB", (width, height), color=(220, 50, 50))
blue_image = Image.new("RGB", (width, height), color=(50, 50, 220))
mixed_image = Image.new("RGB", (width, height), color=(50, 200, 50))

test_cases = [
    {
        "label": "Good caption match",
        "image": red_image,
        "candidate": "A solid red colored square filling the entire frame",
        "references": [
            "A completely red square image with no other colors",
            "An image filled entirely with red pixels"
        ]
    },
    {
        "label": "Poor caption match (wrong color)",
        "image": red_image,
        "candidate": "A beautiful blue ocean under a cloudy sky",
        "references": [
            "A completely red square image with no other colors"
        ]
    },
    {
        "label": "Fluent but generic",
        "image": mixed_image,
        "candidate": "An image of something",
        "references": [
            "A solid green square filling the frame completely"
        ]
    }
]

print("=" * 70)
print("MULTIMODAL EVALUATION RESULTS")
print("=" * 70)

for i, case in enumerate(test_cases):
    print(f"\n--- Case {i+1}: {case['label']} ---")
    print(f"  Candidate: \"{case['candidate']}\"")
    print(f"  Reference: \"{case['references'][0]}\"")
    
    clip_score = compute_clip_score(case["image"], case["candidate"])
    bleu_score = compute_bleu4(case["candidate"], case["references"])
    cider_score = compute_cider(case["candidate"], case["references"])
    
    print(f"\n  CLIPScore:  {clip_score:.4f}  (alignment: 0=worst, ~0.3=good)")
    print(f"  BLEU-4:     {bleu_score:.4f}  (n-gram match: 0=none, 1=perfect)")
    print(f"  CIDEr:      {cider_score:.4f}  (TF-IDF weighted: 0=none, >1=good)")

print("\n" + "=" * 70)
print("CROSS-METRIC COMPARISON")
print("=" * 70)

case = test_cases[0]
ref_caption = case["references"][0]
wrong_caption = case["candidate"] if case["label"] == "Poor caption match" else None
alt_caption = "A red rectangle on a white background"

clip_correct = compute_clip_score(case["image"], ref_caption)
clip_alt = compute_clip_score(case["image"], alt_caption)
clip_wrong = compute_clip_score(red_image, "A blue square")

print(f"\n  Same image (red), different captions:")
print(f"    'A completely red square...' -> CLIPScore: {clip_correct:.4f}")
print(f"    'A red rectangle...'         -> CLIPScore: {clip_alt:.4f}")
print(f"    'A blue square'              -> CLIPScore: {clip_wrong:.4f}")
print(f"\n  Key insight: wrong-color caption scores {clip_wrong:.4f}")
print(f"  vs correct caption at {clip_correct:.4f}")
print(f"  Gap: {clip_correct - clip_wrong:.4f} (higher = CLIP detects mismatch)")

retrieval_captions = [
    "A solid red square",
    "A blue ocean scene",
    "A green colored square",
    "A red rectangle shape",
    "A yellow sun on horizon"
]

print("\n" + "=" * 70)
print("RETRIEVAL: Rank images for each caption (Recall@1, @3)")
print("=" * 70)

images = {"red": red_image, "blue": blue_image, "green": mixed_image}
correct_mapping = {
    "A solid red square": "red",
    "A blue ocean scene": "blue",
    "A green colored square": "green"
}

r1_hits = 0
r3_hits = 0
total = 0

for caption in retrieval_captions:
    scores = {}
    for img_name, img in images.items():
        s = compute_clip_score(img, caption)
        scores[img_name] = s
    
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top1 = ranked[0][0]
    top3 = [r[0] for r in ranked[:3]]
    
    if caption in correct_mapping:
        correct = correct_mapping[caption]
        total += 1
        if top1 == correct:
            r1_hits += 1
        if correct in top3:
            r3_hits += 1
    
    print(f"\n  Caption: \"{caption}\"")
    for name, score in ranked:
        marker = " <-- correct" if caption in correct_mapping and name == correct_mapping[caption] else ""
        print(f"    {name:6s}: {score:.4f}{marker}")

print(f"\n  Recall@1: {r1_hits}/{total} = {r1_hits/total:.2%}")
print(f"  Recall@3: {r3_hits}/{total} = {r3_hits/total:.2%}")
random_r1 = 1 / len(images)
random_r3 = min(3 / len(images), 1.0)
print(f"  Random baseline R@1: {random_r1:.2%}, R@3: {random_r3:.2%}")
```

When you run this, you'll observe three things. First, CLIPScore penalizes the wrong-color caption — "A blue square" scored against a red image will land noticeably lower than "A completely red square." Second, BLEU-4 gives a low score even for the "good caption match" case because the candidate and reference share few exact n-grams despite describing the same image. Third, the retrieval ranking correctly places the red image first for "A solid red square" but may struggle with less literal phrasing, showing exactly where CLIP's embedding space generalizes and where it doesn't.

Now the LLM-as-judge harness. This uses an OpenAI-compatible API call with a vision model. If you don't have an API key, the prompt template and expected output format are still worth studying — the evaluation design is the point, not the specific API.

```python
import json
import os

try:
    from openai import OpenAI
    client = OpenAI() if os.environ.get("OPENAI_API_KEY") else None
except ImportError:
    client = None

def encode_image_pil(image):
    import io
    import base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def llm_judge_caption(image, candidate_caption, reference_caption=None):
    if client is None:
        print("  [LLM-JUDGE] No API key set. Showing prompt only.")
        prompt = f"""
You are evaluating a generated caption for an image.

Generated caption: "{candidate_caption}"
Reference caption (if provided): "{reference_caption or 'N/A'}"

Rate the generated caption on three criteria, each 1-5:
1. ACCURACY: Does the caption correctly describe what is in the image?
2. FLUENCY: Is the caption grammatically correct and natural to read?
3. COMPLETENESS: Does the caption capture the important elements of the image?

Respond in JSON format:
{{"accuracy": <int>, "fluency": <int>, "completeness": <int>, "reasoning": "<one sentence>"}}
"""
        print(prompt)
        return None
    
    base64_image = encode_image_pil(image)
    
    prompt = f"""You are evaluating a generated caption for an image.

Generated caption: "{candidate_caption}"
Reference caption (if provided): "{reference_caption or 'N/A'}"

Rate the generated caption on three criteria, each 1-5:
1. ACCURACY: Does the caption correctly describe what is in the image?
2. FLUENCY: Is the caption grammatically correct and natural to read?
3. COMPLETENESS: Does the caption capture the important elements of the image?

Respond in JSON format only:
{{"accuracy": <int>, "fluency": <int>, "completeness": <int>, "reasoning": "<one sentence>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ],
        temperature=0.0,
        max_tokens=200
    )
    
    result = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(result)
        return parsed
    except json.JSONDecodeError:
        return {"raw_response": result}

judge_cases = [
    {
        "label": "Accurate caption",
        "image": red_image,
        "candidate": "A solid red square filling the entire image",
        "reference": "A completely red square"
    },
    {
        "label": "Inaccurate caption",
        "image": red_image,
        "candidate": "A blue ocean at sunset",
        "reference": "A completely red square"
    }
]

print("\n" + "=" * 70)
print("LLM-AS-JUDGE EVALUATION")
print("=" * 70)

for case in judge_cases:
    print(f"\n--- {case['label']} ---")
    print(f"  Candidate: \"{case['candidate']}\"")
    result = llm_judge_caption(case["image"], case["candidate"], case["reference"])
    if result:
        print(f"  Accuracy:    {result.get('accuracy', 'N/A')}/5")
        print(f"  Fluency:     {result.get('fluency', 'N/A')}/5")
        print(f"  Completeness:{result.get('completeness', 'N/A')}/5")
        print(f"  Reasoning:   {result.get('reasoning', 'N/A')}")
```

The judge catches what formulaic metrics miss: "An image of something" is fluent but inaccurate and incomplete. The rubric separates these axes instead of collapsing them into a single number. This is the mechanism that lets you evaluate subjective signals like "does this homepage communicate enterprise readiness?" in a GTM enrichment pipeline.

## Use It

The enrichment patterns in go-to-market workflows routinely involve vision-language models extracting signals from multimodal content. When your pipeline ingests a prospect's website, screenshots, PDFs, or video assets, you need to know whether the extracted signals are trustworthy before they enter a scoring waterfall. CLIPScore answers: does the text your model produced actually describe the image it was given? This matters directly when a vision model is screenshotting company homepages to detect pricing page layouts, technology stack badges, or integration partner logos. A high CLIPScore between the model's description and the actual screenshot pixels means the extraction is grounded in the visual content. A low score means the model hallucinated — and that hallucinated signal is about to flow into lead scoring.

The scoring waterfall pattern — where signals pass through enrichment stages before reaching a final score — inherits whatever error rate each stage introduces. If your vision-language model extracts "enterprise pricing detected" from a homepage with a CLIPScore of 0.15 (meaning the description barely matches the image), that signal should be downweighted or discarded before it reaches the waterfall. Multimodal evaluation is the quality gate that sits between extraction and scoring. Zone 2 (Enrichment) in the GTM stack depends on this gate functioning, though the specific integration points between multimodal eval outputs and Clay waterfall stages are not publicly documented in detail [CITATION NEEDED — concept: Clay waterfall integration with multimodal enrichment signals].

In practice, the evaluation loop looks like this: your enrichment pipeline runs a vision-language model over 1,000 prospect screenshots and produces structured signals. You sample 100 of those, run CLIPScore on each image-signal pair, and compute the distribution. If the median CLIPScore is above 0.25, the extractions are generally grounded. If it's below 0.15, you have a systematic problem — the model isn't reading the images, it's pattern-matching on the domain name or URL structure. BLEU and CIDEr play a secondary role here: they tell you whether the extracted text is well-formed enough to parse downstream. An LLM-as-judge handles the cases where the signal is subjective: "Does this company's visual branding suggest enterprise or SMB?" The weighted combination of these three families gives you a confidence score per enrichment record, which is what you need to decide whether to trust the signal in the waterfall.

The RAG pattern in Zone 19 of the GTM topic map — "knowledge-augmented outreach" — also intersects here. When your outbound agent retrieves case studies, product docs, and customer stories to ground its messaging, multimodal evaluation measures whether the agent is retrieving and representing visual content correctly. A case study PDF with charts, screenshots, and diagrams requires multimodal understanding, not just text extraction. CLIPScore on the retrieved chunks tells you whether the agent's representation of a chart matches the actual chart pixels.

## Ship It

Production multimodal evaluation has three cost tiers and the tradeoffs are concrete. CLIPScore costs one forward pass through a 150M-parameter model — roughly 5-20ms per image-caption pair on GPU, 50-100ms on CPU. You can run this on every enrichment record in real time. BLEU and CIDEr are pure Python computation on token sequences — sub-millisecond per pair. The LLM-as-judge is the expensive one: each evaluation call is a full inference pass through a frontier model with vision capabilities, costing $0.01-0.05 per evaluation at current API prices and adding 1-5 seconds of latency. You cannot run LLM-as-judge on every record. You sample.

The production pattern is tiered evaluation. Run CLIPScore and CIDEr on every record in the batch — they're cheap enough. Flag any record where CLIPScore falls below a threshold (0.15 is a reasonable starting point based on CLIP's score distribution for mismatched pairs). Route flagged records to the LLM-as-judge for a deeper evaluation against a rubric. This gives you full coverage on the cheap metrics and targeted deep evaluation on the records most likely to be wrong. For an enrichment pipeline processing 10,000 prospect websites, this means 10,000 CLIPScore computations (seconds), 10,000 CIDEr computations (seconds), and maybe 500 LLM-judge calls (minutes, dollars).

Human-in-the-loop sampling is the third tier. Even LLM-as-judge has blind spots — it shares biases with its training data, and it can be fooled by adversarial inputs. A weekly sample of 50-100 enrichment records reviewed by a human catches systematic drift that automated metrics miss. The sampling strategy matters: don't sample randomly. Sample stratified by CLIPScore bucket, by industry vertical, by signal type. The records where CLIPScore is moderate (0.15-0.25) — not obviously right, not obviously wrong — are where human review adds the most information.

Versioning is the hidden cost. Every time you swap the vision-language model in your enrichment pipeline, you need to re-run evaluation on a held-out set to confirm the new model is actually better. "Better" means higher scores across multiple metrics, not just lower training loss. Maintain a frozen evaluation set of 200-500 prospect screenshots with reference captions and human-labeled signals. Run the full metric suite on every model version. Track CLIPScore distribution, CIDEr mean, BLEU-4 mean, and LLM-judge accuracy across versions. If a new model improves BLEU but drops CLIPScore, it's writing better English about the wrong images — a regression, not an improvement.

## Exercises

**Easy:** Run the CLIPScore evaluation script above. Modify the `test_cases` list to add five new image-caption pairs using different solid colors and descriptions. Identify which pairs score below 0.20 and write a one-sentence hypothesis for why each low-scoring pair failed. Is the failure in the image encoding, the text encoding, or genuine mismatch?

**Medium:** Implement a retrieval Recall@K metric for a text-to-image retrieval task. Create 10 synthetic images (solid colors, simple shapes drawn with PIL) and 15 captions (10 correct, 5 distractors). Compute Recall@1, Recall@5, and Recall@10. Then generate a random baseline by shuffling the rankings. Report both and compute the lift over random. How does performance change when you add more distractor captions?

**Hard:** Build a multi-metric evaluation pipeline that combines CLIPScore, CIDEr, and LLM-judge into a single weighted score. Create a labeled dataset of 20 image-caption pairs with human quality ratings (1-5). Tune the weights (w_clip, w_cider, w_judge) to maximize Spearman correlation with human judgments. Report the optimal weights, the correlation achieved, and which metric contributes most to the correlation. Then remove one metric at a time and report how correlation drops — which metric is most replaceable?

## Key Terms

- **CLIPScore** — Cosine similarity between CLIP-encoded image and text embeddings, scaled by 2.5. Measures cross-modal alignment without requiring reference captions.
- **BLEU-4** — Geometric mean of 1-gram through 4-gram precision between generated and reference text, multiplied by a brevity penalty. Measures text overlap; blind to image content.
- **CIDEr** — TF-IDF weighted n-gram overlap between candidate and reference captions. Rewards image-specific vocabulary over generic words. Standard for image captioning benchmarks.
- **Recall@K** — Fraction of queries where the correct item appears in the top K ranked results. Used for retrieval tasks (text-to-image, image-to-text).
- **LLM-as-judge** — Using a language model (with vision) to evaluate outputs against a rubric. Handles subjective qualities that formulaic metrics cannot measure. Expensive and non-deterministic.
- **Cross-modal alignment** — Whether text and image correspond semantically. Measured by shared embedding space similarity, not surface-level text features.
- **Brevity penalty** — Multiplicative factor in BLEU that penalizes generated text shorter than the reference, preventing trivially short outputs from gaming precision.
- **Modified precision** — BLEU's n-gram precision clipped by the maximum reference count for each n-gram, preventing repeated words from inflating the score.

## Sources

- CLIPScore as a reference-free image captioning metric: Hessel et al., "CLIPScore: A Reference-free Evaluation Metric for Image Captioning," EMNLP 2021. The 2.5 scaling factor and cosine similarity formulation derive from this paper.
- BLEU-4 formulation including brevity penalty and modified n-gram precision: Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation," ACL 2002.
- CIDEr TF-IDF weighted n-gram consensus metric: Vedantam et al., "CIDEr: Consensus-based Image Description Evaluation," CVPR 2015.
- CLIP model architecture and joint embedding space: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021.
- LLM-as-judge methodology for evaluating model outputs: Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," NeurIPS 2023.
- Recall@K as a retrieval evaluation metric for image-caption matching: Karpathy & Fei-Fei, "Deep Visual-Semantic Alignments for Generating Image Descriptions," CVPR 2015.
- [CITATION NEEDED — concept: Clay waterfall integration with multimodal enrichment signals] — The specific mechanism by which multimodal evaluation scores feed into Clay's scoring waterfall stages is not documented in public Clay resources. The pattern described (evaluation as a quality gate between extraction and scoring) is a general pipeline design principle, not a Clay-specific feature.