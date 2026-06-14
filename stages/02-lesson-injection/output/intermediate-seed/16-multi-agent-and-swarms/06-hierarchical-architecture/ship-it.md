## Ship It

Here is a configurable orchestrator that accepts a YAML tree definition, runs the hierarchy, logs per-node latency and token usage, and detects context dilution by comparing leaf outputs to root summary via a lightweight text-similarity heuristic (Jaccard overlap on token sets—no embedding API required to demonstrate the mechanism).

```python
import yaml
import time
import json
from collections import Counter

def mock_llm(prompt, context=""):
    if "revenue" in prompt.lower():
        return "Revenue: $45M ARR. Growth: 32% YoY. Gross margin: 78%."
    elif "hiring" in prompt.lower():
        return "Hiring: 12 open roles, 8 in engineering. 3 in sales. Senior-heavy."
    elif "news" in prompt.lower():
        return "News: Series B announced. New CTO hired. Product launch in Q3."
    elif "funding" in prompt.lower():
        return "Funding: $50M Series B from Sequoia. Post-money valuation $300M."
    else:
        return f"Generic analysis based on: {context[:100]}"

class ConfigurableNode:
    def __init__(self, config):
        self.name = config["name"]
        self.prompt = config["prompt"]
        self.children = [ConfigurableNode(c) for c in config.get("children", [])]
        self.depth = 0
        self.latency_ms = 0
        self.token_count_in = 0
        self.token_count_out = 0
        self.raw_child_outputs = []
        self.output = ""

    def run(self, depth=0):
        self.depth = depth
        start = time.time()

        if self.children:
            child_results = []
            for child in self.children:
                result = child.run(depth + 1)
                child_results.append(result)
                self.raw_child_outputs.append(result)

            combined = " ".join(child_results)
            self.token_count_in = len(combined.split())
            self.output = self._summarize(combined)
            self.token_count_out = len(self.output.split())
        else:
            self.output = mock_llm(self.prompt)
            self.token_count_out = len(self.output.split())

        self.latency_ms = round((time.time() - start) * 1000, 2)
        return self.output

    def _summarize(self, text):
        sentences = text.split(". ")
        if len(sentences) <= 2:
            return text
        return ". ".join(sentences[:2]) + "."

def load_tree(yaml_str):
    config = yaml.safe_load(yaml_str)
    return ConfigurableNode(config)

def collect_metrics(node, metrics=None):
    if metrics is None:
        metrics = []
    metrics.append({
        "name": node.name,
        "depth": node.depth,
        "latency_ms": node.latency_ms,
        "tokens_in": node.token_count_in,
        "tokens_out": node.token_count_out,
        "is_leaf": len(node.children) == 0
    })
    for child in node.children:
        collect_metrics(child, metrics)
    return metrics

def collect_leaf_outputs(node, leaves=None):
    if leaves is None:
        leaves = []
    if not node.children:
        leaves.append((node.name, node.output))
    for child in node.children:
        collect_leaf_outputs(child, leaves)
    return leaves

def jaccard_similarity(text_a, text_b):
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)

def detect_dilution(node, threshold=0.15):
    root_output = node.output
    leaf_outputs = collect_leaf_outputs(node)
    results = []
    for leaf_name, leaf_text in leaf_outputs:
        similarity = jaccard_similarity(root_output, leaf_text)
        status = "DILUTED" if similarity < threshold else "OK"
        results.append({
            "leaf": leaf_name,
            "jaccard_to_root": round(similarity, 4),
            "status": status
        })
    return results

def flatten_tree(node):
    flat_prompts = []
    def _collect(n):
        if not n.children:
            flat_prompts.append((n.name, n.prompt))
        for child in n.children:
            _collect(child)
    _collect(node)
    return flat_prompts

tree_yaml = """
name: company_analysis_root
prompt: "Produce a comprehensive company brief."
children:
  - name: market_subteam
    prompt: "Analyze market position."
    children:
      - name: revenue_leaf
        prompt: "Extract revenue and growth metrics from financials."
      - name: funding_leaf
        prompt: "Summarize recent funding events."
  - name: signals_subteam
    prompt: "Analyze growth signals."
    children:
      - name: hiring_leaf
        prompt: "Analyze hiring velocity and open roles."
      - name: news_leaf
        prompt: "Summarize recent company news."
"""

tree = load_tree(tree_yaml)
tree.run()

print("=== HIERARCHICAL EXECUTION ===")
metrics = collect_metrics(tree)
for m in metrics:
    print(f"  [d={m['depth']}] {m['name']}: {m['latency_ms']}ms, "
          f"tokens in={m['tokens_in']}, out={m['tokens_out']}, leaf={m['is_leaf']}")

total_latency = sum(m["latency_ms"] for m in metrics)
total_tokens_in = sum(m["tokens_in"] for m in metrics)
total_tokens_out = sum(m["tokens_out"] for m in metrics)
print(f"\n  TOTAL: latency={total_latency}ms, tokens_in={total_tokens_in}, tokens_out={total_tokens_out}")

print("\n=== CONTEXT DILUTION DETECTION ===")
dilution = detect_dilution(tree, threshold=0.15)
for d in dilution:
    print(f"  {d['leaf']}: jaccard={d['jaccard_to_root']} → {d['status']}")

diluted_count = sum(1 for d in dilution if d["status"] == "DILUTED")
dilution_ratio = diluted_count / len(dilution) if dilution else 0
print(f"\n  Diluted leaves: {diluted_count}/{len(dilution)} (ratio: {dilution_ratio:.2f})")

if dilution_ratio > 0.5:
    print("\n=== AUTO-COLLAPSE TRIGGERED ===")
    print("  More than 50% of leaf outputs are diluted. Flattening to fan-out.")
    flat_prompts = flatten_tree(tree)
    print(f"  Flat fan-out: {len(flat_prompts)} parallel calls")
    flat_results = {}
    for name, prompt in flat_prompts:
        flat_results[name] = mock_llm(prompt)
    merged = " ".join(flat_results.values())
    print(f"\n  MERGED OUTPUT ({len(merged.split())} tokens):")
    print(f"  {merged}")

    re_check_similarity = max(
        jaccard_similarity(merged, leaf_text)
        for _, leaf_text in collect_leaf_outputs(tree)
    )
    print(f"\n  Best leaf-to-merged similarity: {re_check_similarity:.4f}")
    print(f"  → Flattened merge preserves {'more' if re_check_similarity > max(d['jaccard_to_root'] for d in dilution) else 'less'} leaf signal than hierarchical root")
else:
    print("\n  Dilution below threshold. Hierarchy preserved.")

print("\n=== LATENCY COMPARISON ===")
hierarchical_latency = total_latency
flat_latency_estimate = max(m["latency_ms"] for m in metrics if m["is_leaf"]) + 50
print(f"  Hierarchical (serial): {hierarchical_latency}ms")
print(f"  Flat fan-out (parallel + merge): ~{flat_latency_estimate}ms")
print(f"  Speedup factor: {hierarchical_latency / flat_latency_estimate:.1f}x")
```

This produces observable output showing every node's latency, token counts, the dilution score for each leaf, and—when dilution exceeds threshold—the automatic collapse to a flat fan-out with the resulting merged output. The Jaccard similarity is a stand-in for embedding cosine distance; in production you would use `sentence-transformers` or an embedding API, but the detection mechanism is the same: measure how much of the leaf signal survives to the root.

The auto-collapse logic is the key production pattern. When more than half your leaf outputs are diluted below threshold, the hierarchy is destroying more signal than it organizes. Flatten and re-run. This is directly applicable to enrichment: if your waterfall's final record bears little resemblance to the raw provider data that fed it, stop serializing and run providers in parallel.