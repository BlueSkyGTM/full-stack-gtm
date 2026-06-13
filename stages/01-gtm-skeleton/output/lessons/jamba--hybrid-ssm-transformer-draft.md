# Lesson: Jamba — Hybrid SSM-Transformer

---

## Hook

Pure Transformers scale quadratically with sequence length. Pure State Space Models (SSMs) scale linearly but lose fine-grained token-to-token reasoning. Jamba interleaves both, and that tradeoff decision shows up any time you process long GTM artifacts — call transcripts, enrichment payloads, multi-step agent traces.

---

## Concept

**Mechanism:** Jamba stacks Mamba (selective SSM) blocks alongside standard Transformer blocks (attention + MLP) in a single sequence. The Mamba blocks handle the bulk of sequence modeling at O(n) cost. The attention layers provide precise retrieval over the full context. A MoE (Mixture of Experts) MLP replaces the dense MLP, increasing parameter count without proportional compute increase.

The result: a 52B parameter model (12B active during inference) with a 256K token context window that fits in a single GPU.

---

## Mechanism

1. **Selective SSM (Mamba) refresher** — input-dependent state transitions, hardware-aware scan kernel, O(n) time and memory.
2. **Transformer block refresher** — multi-head attention + MLP, O(n²) time, O(n) memory with FlashAttention.
3. **Interleaving pattern** — Jamba places attention layers at specific intervals within a stack of Mamba blocks. Not every layer attends. Ratio and placement determine the quality/cost curve.
4. **MoE MLP** — router selects top-k experts per token. Sparse activation means total parameters >> active parameters.
5. **Memory flow** — SSM state carries information between attention layers. Attention layers act as "refresh points" that pull from arbitrary positions.

```
[Mamba] → [Mamba] → [Mamba] → [Attention + MoE MLP] → [Mamba] → ...
```

The SSM state is the connective tissue. Attention is the surgical instrument.

---

## Code

```python
import subprocess
import json

result = subprocess.run(
    ["curl", "-s", "https://api.ai21labs.com/v1/chat/completions",
     "-H", "Authorization: Bearer $AI21_API_KEY",
     "-H", "Content-Type: application/json",
     "-d", json.dumps({
         "model": "jamba-1.5-large",
         "messages": [
             {"role": "system", "content": "You process long sales transcripts."},
             {"role": "user", "content": "Summarize the key objection raised in the last quarter of this transcript: [paste 100K+ token transcript here]"}
         ],
         "max_tokens": 200
     })],
    capture_output=True,
    text=True
)

response = json.loads(result.stdout)
print(response["choices"][0]["message"]["content"])
print(f"\nModel: {response['model']}")
print(f"Usage: {response['usage']}")
```

Second example — compare token throughput at increasing context lengths to observe the linear vs. quadratic scaling difference empirically.

Exercise hooks:
- **Easy:** Run the API call above with a real transcript. Print the response.
- **Medium:** Write a script that sends the same prompt at 1K, 10K, and 50K context lengths, measures latency for each, and prints a simple table showing context_length vs. response_time.
- **Hard:** Build a CLI tool that takes a directory of call transcripts, concatenates them up to Jamba's context limit, and extracts all decision-maker objections into a structured JSON array.

---

## Use It

**GTM Redirect:** Zone 1 (Enrich) — processing large enrichment payloads.

When you aggregate data from multiple providers (Clay waterfall output, SEC filings, GitHub activity, Slack transcripts) for a single account, the combined payload can exceed 100K tokens. Jamba's architecture handles this in a single pass without chunking or retrieval augmentation.

Specific application: feed a full account research dossier (technographic data, intent signals, call history, email threads) into a single Jamba context window. Ask for a prioritized outreach angle. No RAG. No chunking. No lost context between chunks.

The SSM layers compress the long tail of the dossier efficiently. The attention layers nail the retrieval when the model needs to cross-reference a detail from page 3 with one from page 47.

Exercise hooks:
- **Easy:** Take a Clay export (CSV) for 5 accounts, concatenate into a single text block, and ask Jamba to rank them by ICP fit.
- **Medium:** Feed 10 call transcripts from the same account into Jamba. Extract a timeline of objection evolution across the calls.
- **Hard:** Build an enrichment aggregator that pulls from 3 sources, assembles the payload, and uses Jamba to generate account-specific talk tracks — all within a single context window.

---

## Ship It

Deploy a long-context enrichment pipeline that replaces a chunked-RAG approach with a single Jamba call.

**Validation criteria:**
- Pipeline accepts inputs up to 200K tokens
- Latency is measured and logged per request
- Output quality is compared against a chunked baseline (at least 5 test accounts)
- Cost per enrichment is calculated (tokens × pricing)

Exercise hooks:
- **Easy:** Write a script that wraps the Jamba API in a function with input length validation and retries.
- **Medium:** Build an enrichment endpoint that accepts an account ID, fetches enrichment data from a mock source, assembles the full context, calls Jamba, and returns a structured account summary.
- **Hard:** Implement a side-by-side evaluation: same 20 accounts enriched via (a) chunked RAG with a standard LLM and (b) full-context Jamba. Score both on factuality and coherence. Print the comparison table.

---

**Learning Objectives:**

1. Compare SSM and Transformer attention mechanisms by computational complexity at varying sequence lengths.
2. Diagram the Jamba block interleaving pattern and identify which operations are O(n) vs. O(n²).
3. Implement a long-context enrichment call using Jamba's API with observable latency and token usage output.
4. Evaluate when a hybrid SSM-Transformer outperforms pure Transformer architectures for GTM enrichment workloads.
5. Measure and compare cost-per-enrichment between chunked-RAG and full-context approaches.