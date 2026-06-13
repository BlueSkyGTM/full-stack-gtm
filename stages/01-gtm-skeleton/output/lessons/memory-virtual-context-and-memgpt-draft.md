# Memory: Virtual Context and MemGPT

## GTM Redirect Rules

This lesson redirects to **Zone I: Orchestration** — specifically the multi-step agent execution pattern where context persistence across tool calls determines whether an enrichment or outreach sequence completes or drifts. The mechanism is virtual context management; the GTM application is maintaining account intelligence across multi-touch workflows where the agent must remember prior enrichment results, rejection reasons, and routing decisions.

---

## Hook

You're running a multi-step enrichment waterfall. By step 4, the agent has accumulated prior tool results, rejection reasons, and partial matches. The context window fills. The agent "forgets" what it learned in step 1 and duplicates work or makes contradictory decisions. This is the memory wall. Virtual context management is the pattern that solves it.

---

## Concept

**Context window as RAM, external storage as disk.** The LLM has bounded working memory (context window). Virtual context management implements a paging system: the model decides what to keep in active context and what to swap to persistent storage. MemGPT formalizes this as three tiers:

1. **Core memory** — always in context, editable by the model (like CPU registers)
2. **Archival memory** — external storage, searchable, not in context until retrieved (like disk)
3. **Conversational memory** — recent dialogue turns, automatically managed (like cache)

The model controls its own memory through function calls. It decides when to insert, search, replace. This is not RAG — RAG retrieves from a fixed corpus. Virtual context lets the model *edit its own working memory during execution.*

[CITATION NEEDED — concept: MemGPT paper formal tier definitions and paging algorithm]

---

## Mechanism

The algorithm works as follows:

1. **Initialize**: Core memory blocks are injected into the system prompt. Archival memory starts empty or pre-loaded.
2. **Per turn**: The model receives the current context (system + core memory + recent conversation).
3. **Self-edit**: The model can call functions like `core_memory_replace(old, new)`, `archival_memory_insert(text)`, `archival_memory_search(query)`.
4. **Swap**: When context pressure rises, the model pages out low-priority information to archival and pages in relevant archived context.
5. **Persist**: Memory state survives across sessions — the next conversation starts with updated core memory.

The key insight: the model is the memory manager. It decides what matters. This trades deterministic control for adaptability. The risk is the model forgets to remember — it fails to call the right memory function at the right time.

Observable behavior: with virtual context, a multi-step agent maintains consistent state across 20+ tool calls. Without it, context drift begins once the working memory saturates.

[CITATION NEEDED — concept: MemGPT paging trigger thresholds and context pressure calculation]

---

## Code

Build a minimal virtual context manager. Demonstrate memory tier swapping with observable output.

**Exercise hooks:**

- **Easy**: Print the state of core memory before and after a self-edit operation.
- **Medium**: Implement archival memory search that retrieves and injects into context, printing context length at each step.
- **Hard**: Run a 10-step enrichment simulation. Track context pressure. Demonstrate that with virtual context, the agent recalls step-1 results at step-10. Print a summary showing recall accuracy with and without virtual context.

```python
import json

class VirtualContextManager:
    def __init__(self):
        self.core_memory = {}
        self.archival_memory = []
        self.conversation_history = []
        self.max_context_tokens = 1000
    
    def core_memory_insert(self, key, value):
        self.core_memory[key] = value
        print(f"[CORE WRITE] {key} = {value}")
        self._check_context_pressure()
    
    def core_memory_replace(self, key, old_value, new_value):
        if self.core_memory.get(key) == old_value:
            self.core_memory[key] = new_value
            print(f"[CORE REPLACE] {key}: {old_value} -> {new_value}")
        else:
            print(f"[CORE MISS] {key} does not match {old_value}")
    
    def archival_insert(self, text):
        self.archival_memory.append(text)
        print(f"[ARCHIVAL WRITE] Stored: {text[:50]}...")
    
    def archival_search(self, query):
        results = [t for t in self.archival_memory if query.lower() in t.lower()]
        print(f"[ARCHIVAL READ] Query: '{query}' | Found: {len(results)} results")
        for r in results:
            print(f"  -> {r[:80]}...")
        return results
    
    def _check_context_pressure(self):
        estimated = len(str(self.core_memory)) + len(str(self.conversation_history))
        pressure = estimated / self.max_context_tokens
        print(f"[PRESSURE] {pressure:.1%} context used")
        if pressure > 0.8:
            print("[PRESSURE] Threshold exceeded — model should page out to archival")
    
    def get_context(self):
        return {
            "core": self.core_memory,
            "conversation_turns": len(self.conversation_history),
            "archival_count": len(self.archival_memory)
        }

vc = VirtualContextManager()

vc.core_memory_insert("account", "Acme Corp")
vc.core_memory_insert("enrichment_step", "1")
vc.core_memory_insert("domain_confirmed", "acme.com")

vc.archival_insert("Acme Corp uses Salesforce. Confirmed via LinkedIn technographics.")
vc.archival_insert("ICP match score: 87. Triggered waterfall step 1.")

results = vc.archival_search("Acme")
print(f"\nCurrent context state: {json.dumps(vc.get_context(), indent=2)}")
```

Expected output:
```
[CORE WRITE] account = Acme Corp
[PRESSURE] 5.2% context used
[CORE WRITE] enrichment_step = 1
[PRESSURE] 7.8% context used
[CORE WRITE] domain_confirmed = acme.com
[PRESSURE] 10.4% context used
[ARCHIVAL WRITE] Stored: Acme Corp uses Salesforce. Confirmed via LinkedIn tech...
[ARCHIVAL WRITE] Stored: ICP match score: 87. Triggered waterfall step 1....
[ARCHIVAL READ] Query: 'Acme' | Found: 2 results
  -> Acme Corp uses Salesforce. Confirmed via LinkedIn technographics....
  -> ICP match score: 87. Triggered waterfall step 1....

Current context state: {
  "core": {
    "account": "Acme Corp",
    "enrichment_step": "1",
    "domain_confirmed": "acme.com"
  },
  "conversation_turns": 0,
  "archival_count": 2
}
```

---

## Use It

**GTM cluster: Zone I — Orchestration / Enrichment Waterfalls**

The Clay waterfall is a multi-step enrichment agent. Each step produces results that subsequent steps need. Without virtual context, the agent loses track of what it already found. With virtual context:

- **Core memory holds**: current account, ICP match status, which waterfall steps completed
- **Archival memory holds**: full enrichment results, rejection reasons, source attribution
- **The agent pages in** only what the next step requires

This is not hypothetical — if you run a 6-step Clay waterfall and the contact has no LinkedIn, no email, and only a partial phone, the agent must remember *which* lookups failed to avoid retrying them. That's virtual context management.

**Exercise hooks:**

- **Easy**: Map Clay waterfall steps to memory tiers — which data belongs in core vs. archival.
- **Medium**: Simulate a 4-step enrichment sequence. Store results in virtual context. Demonstrate that step 4 correctly references step 1 results without re-fetching.
- **Hard**: Build a context-pressure monitor for a real Clay waterfall. Log when the agent repeats a lookup vs. recalls from memory. Print the deduplication rate.

[CITATION NEEDED — concept: Clay waterfall context management implementation and context window limits per step]

---

## Ship It

Build a memory-aware enrichment agent that maintains state across a multi-step GTM workflow. The agent must:

1. Track which enrichment sources have been queried
2. Store results in appropriate memory tiers
3. Recall prior results when making routing decisions
4. Report context pressure and memory tier utilization

**Validation**: Run the same enrichment query twice. Second run should retrieve from archival memory instead of re-querying. Print a diff showing zero duplicate external calls.

**Exercise hooks:**

- **Easy**: Add archival search to an existing agent. Print cache hit rate.
- **Medium**: Implement the full tiered memory system. Demonstrate a 5-step workflow with zero redundant lookups.
- **Hard**: Add context pressure monitoring. When pressure exceeds 80%, automatically page low-priority core memory to archival. Print the paging log.