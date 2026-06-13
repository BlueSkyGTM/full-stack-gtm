# Open Models: Architecture Walkthroughs

## Hook
You're comparing two models for a classification pipeline. One has 32 attention heads and rotary embeddings. The other has GQA and tied embeddings. The spec sheets market both as "7B parameters." Without reading the architecture, you're gambling on inference cost, latency, and quality. This lesson makes that spec sheet legible.

---

## Concept (Beat 1)
**Description:** Dissect the transformer architecture as implemented in popular open-weight model families (Llama, Mistral, Qwen). Walk through each component — embedding layer, attention mechanism (MHA vs GQA vs MQA), position encoding (RoPE vs ALiBi vs learned), MLP block (standard vs SwiGLU), normalization (LayerNorm vs RMSNorm), and the residual stream connecting them. Map each component to a config.json key so the practitioner can read any Hugging Face model card and predict behavior.

**Exercise hooks:**
- *Easy:* Given two config.json files, identify which model uses GQA and calculate its KV cache savings ratio.
- *Medium:* Trace a single token through a 2-layer toy transformer, printing the tensor shape at each operation.
- *Hard:* Modify a config.json to convert MHA to GQA, then calculate the new parameter count and memory footprint.

---

## Demonstration (Beat 2)
**Description:** Load three model configs (Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B) from Hugging Face. Parse each config.json, extract architecture parameters, and build a comparison table. Then load one model and inspect its layer structure programmatically — print each module, its parameter count, and its dtype. Show that "7B parameters" hides meaningful architectural differences that affect inference speed and context handling.

**Exercise hooks:**
- *Easy:* Write a script that downloads a model config and prints the attention head configuration.
- *Medium:* Write a script that loads two models and prints a side-by-side parameter comparison for each layer type.
- *Hard:* Write a script that estimates peak GPU memory for inference given a model config, sequence length, and batch size.

---

## Use It (Beat 3)
**Description:** GTM redirect to **Signal Enrichment** (Zone 1). When building enrichment pipelines that classify, extract, or score leads using local models, architecture determines: context window (how much firmographic data fits in one call), GQA ratio (how many concurrent enrichment jobs fit on one GPU), and SwiGLU vs standard MLP (accuracy on entity extraction tasks). Demonstrate: given a enrichment job processing 1000 company descriptions through a classifier, calculate throughput difference between Llama-style GQA and Mistral-style MHA on the same GPU. This is not abstract — it's the difference between a 4-hour batch job and a 12-hour batch job.

**Exercise hooks:**
- *Easy:* Calculate KV cache memory for a batch of 32 company descriptions at 512 tokens each for two model configs.
- *Medium:* Write a cost estimator that takes model config, GPU type, and batch size, and returns estimated enrichment throughput.
- *Hard:* Benchmark two architectures on a real classification task (e.g., industry categorization from company descriptions) and report accuracy vs latency tradeoff.

---

## Ship It (Beat 4)
**Description:** Production checklist for deploying open models in GTM tooling. Validate architecture against inference backend compatibility (vLLM requires specific attention implementations, llama.cpp quantizes differently per architecture). Read model cards for capability claims vs architectural reality — a model marketed as "128K context" may use sliding window attention that degrades retrieval at long contexts. Write a pre-deployment validation script that checks: does the model config match the serving backend's supported features? Does the stated context window match the position encoding configuration? Does the claimed parameter count match the actual weight files?

**Exercise hooks:**
- *Easy:* Write a script that loads a model card's stated capabilities and cross-references them against the config.json.
- *Medium:* Write a validation script that checks if a given model config is compatible with vLLM's supported feature set.
- *Hard:* Write a deployment readiness checker that takes a model ID, inference backend, and target GPU, and returns pass/fail for memory, context length, and attention compatibility.

---

## Evaluate (Beat 5)
**Description:** Five questions, mapped to objectives. Questions require reading real config.json files and predicting behavior — no definitions, no trivia.

**Exercise hooks:**
- *Easy:* Multiple choice: "Given this config.json, how many KV cache entries are needed for a 2048-token sequence?"
- *Medium:* Short answer: "This model uses RoPE with base 500000. What happens to recall if you extend beyond the trained context window?"
- *Hard:* Analysis: "You switch from Llama-3.1-8B to Mistral-7B-v0.3 for an enrichment pipeline. Both claim 32K context. Your jobs use 24K tokens. One fails silently on recall. Which one and why?"

---

## Learning Objectives

1. **Parse** a Hugging Face config.json and identify the attention mechanism, position encoding, and normalization strategy.
2. **Calculate** KV cache memory requirements given model architecture parameters and batch configuration.
3. **Compare** two open model architectures and predict inference throughput differences for a batch processing workload.
4. **Validate** a model's stated capabilities against its architectural configuration.
5. **Select** an appropriate open model architecture for a GTM enrichment pipeline based on context window, attention efficiency, and deployment constraints.

---

## GTM Redirect Rules

- **Primary redirect:** Signal Enrichment (Zone 1) — architecture determines throughput and cost for classification/extraction pipelines.
- **Secondary redirect:** Foundational for Zone 2 (AI SDR Agents) — architecture context window and attention pattern constraints affect how much prospect context fits in a single inference call.
- **Fallback:** If the AI concept does not cleanly map to a specific GTM workflow, redirect is "foundational for model selection in any local inference pipeline" — no fabricated application.