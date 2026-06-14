# Context Engineering: Windows, Budgets, Memory, and Retrieval

## Learning Objectives

- Calculate token budgets across system prompt, tool definitions, conversation history, retrieved context, and output reservation
- Implement three memory architectures (full history, sliding window, summarization) and compare their token cost on the same conversation
- Trace the retrieval-augmented generation loop as a budget allocation problem, not a search problem
- Build a context assembler that dynamically allocates tokens across segments and reports headroom before the API call
- Diagnose context overflow conditions (truncation, silent failure, hard error) and predict provider behavior at the boundary

## The Problem

Every LLM call has a hard ceiling: the context window. The window is not "memory" — it is a fixed-size buffer the model processes in a single forward pass. Token count, not character count, determines what fits. Retrieval determines what enters the buffer. Four constraints govern every prompt you ship: window size, token budget, memory strategy, and retrieval policy. Ignore any one and the model either errors, truncates silently, or — the worst case — runs degraded with attention smeared across irrelevant context while you believe it is working.

The trap is that the numbers look enormous. Claude Opus 4.5 has a 200K token window. GPT-5 has 400K. Gemini 3 Pro has 2M. Llama 4 claims 10M. These ceilings sound infinite until you fill them with tool definitions, retrieved documents, and multi-turn history. A coding assistant with 50 tool definitions, 10 retrieved doc chunks, and 12 conversation turns burns 20K+ tokens before the user's question even arrives. That is 2% of a 1M window — and 25% of a 128K window. Add a long output budget and you are at the wall.

Attention does not scale linearly with context length either. A model with 128K tokens of context pays quadratic attention cost in vanilla transformers and exhibits "lost in the middle" degradation where facts buried mid-prompt get ignored. The window limit is the hard constraint; attention degradation is the soft constraint. Context engineering is the discipline of staying well inside both — and knowing which segment to sacrifice first when you cannot.

## The Concept

### Context Window Mechanics

The context window is the maximum number of tokens a model can process in a single call. It is a fixed-size buffer, not persistent memory. Each call receives a fresh window unless you explicitly refill it. The budget equation is mechanical: `input_tokens + output_tokens ≤ context_window`. Input tokens cover system prompt, tool definitions, conversation history, retrieved context, and the current user message. Output tokens are reserved for the model's response — and if you do not reserve them, the model has nothing to generate with.

When the budget is exceeded, behavior depends on the provider. OpenAI returns a hard error (`context_length_exceeded`). Anthropic returns a 400 with a similar message. Some runtimes silently truncate the input from the front, dropping the oldest turns without telling you. None of these are recoverable without re-engineering what went into the prompt. The first job of a context engineer is to never discover the boundary by surprise — measure before you call.

### Token Budgeting

Token count is not character count. Modern LLMs use Byte Pair Encoding (BPE): common subwords and short words get assigned single tokens; rare or long words fragment into multiple tokens. "I love cats" is three tokens. "Antidisestablishmentarianism" is six or seven tokens despite fewer characters per token. Punctuation, whitespace, and special characters all consume tokens. JSON and code are token-dense because of braces, quotes, and indentation. A 1,000-character JSON payload can cost 400+ tokens where the same content as prose might cost 200.

`tiktoken` is the tokenizer OpenAI publishes to estimate token counts for GPT-family models before an API call. Anthropic and Google have their own tokenizers with slightly different counts, but `tiktoken` is a reasonable cross-provider approximation when exact counts are not critical. The discipline it enables is budget reservation: you reserve tokens for output (say 1K–4K), then allocate the remaining budget across system prompt, retrieved context, and conversation history. Every segment competes for the same finite space. [CITATION NEEDED — concept: token budget accounting across prompt segments in production systems]

### Memory Architectures

When a conversation exceeds a single turn, you decide what prior context to carry forward. Three patterns dominate. **Full history** stuffs every prior turn into the prompt, scales linearly with turn count, and hits the ceiling fast — but preserves verbatim recall. **Sliding window** keeps the last N turns and drops the oldest, giving constant token cost per call at the price of long-range context loss. **Summarization** compresses older turns into a summary, appends recent turns verbatim, and trades exact quotes for thematic recall at a fraction of the token cost.

Each trades recall fidelity for token efficiency. The right choice depends on whether the model needs exact quotes from earlier turns or thematic understanding of the conversation arc. A support bot handling a 20-message troubleshooting thread usually wants summarization — the user's original problem statement matters, but the back-and-forth debugging steps can be compressed. A coding assistant editing a file across turns usually wants sliding window with full history of the most recent edits — exact code matters.

### Retrieval-Augmented Context

When the needed information exceeds the window or never appeared in conversation, retrieval fills the gap. The RAG loop is mechanical: embed the query, run similarity search against a vector store, retrieve the top-K chunks, inject them into the prompt, then generate. Every chunk you retrieve consumes tokens that could have been used for instructions, history, or output. Retrieval is not free — it is a budget decision made before generation begins, and a lazy `top_k=10` can eat 3,000 tokens that your output reservation needed.

The retrieval step forces an ordering question: where in the prompt do retrieved chunks go, and in what sequence? Research on "lost in the middle" attention suggests placing the most relevant chunks at the beginning and end of the retrieved block, with lower-relevance chunks in the middle. Some engineers rerank retrieved chunks with a cross-encoder before injection to improve signal-to-noise. Both tactics operate within the same budget constraint — they reorder what fits, they do not expand the window.

```mermaid
flowchart TD
    A[User Query] --> B[Token Budget Check]
    B --> C{Total fits in window?}
    C -->|Yes| H[Assemble Final Prompt]
    C -->|No| D[Apply Memory Strategy]
    D --> D1[Full History]
    D --> D2[Sliding Window N]
    D --> D3[Summarize Older]
    D --> E{Still over budget?}
    E -->|Yes| F[Retrieve Top-K Chunks]
    E -->|No| H
    F --> G[Rerank by Relevance]
    G --> E
    H --> I[Reserve Output Tokens]
    I --> J[LLM Call]
    J --> K[Response]
```

## Build It

### Token Budget Reporter

This script takes a realistic GTM prompt — account research with system instructions, tool definitions, retrieved firmographics, and conversation history — and prints a budget report showing how many tokens each segment consumes against a stated window. Run it before every API call in production.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text):
    return len(enc.encode(text))

system_prompt = """You are a GTM engineering assistant specialized in account research.
Given account firmographics, technographics, and intent signals, produce a one-paragraph
account summary and a 1-5 priority score with justification."""

tool_definitions = """tools:
- enrich_account(domain): fetch firmographics and technographics
- get_intent_signals(domain): fetch third-party intent data
- score_account(account_data): apply priority scoring rubric
- draft_personalization(account, channel): draft outreach message
- search_similar_accounts(account): find lookalike accounts in CRM"""

retrieved_context = """Acme Corp is a Series B SaaS company in the HR tech space.
Headcount: 240. ARR estimate: $18M. Tech stack includes Salesforce, HubSpot, Segment,
Snowflake, and dbt. Intent signals show elevated research activity on 'sales engagement
platforms' and 'revenue operations tools' in the past 14 days. Recent funding of $25M
Series B led by Bessemer Venture Partners. Key contacts: CTO (eng tooling buyer),
VP Sales Ops (RevOps tooling buyer). Competitors in their stack: Outreach."""

conversation_history = """user: Research Acme Corp for account prioritization.
assistant: I can help with that. What signals matter most for your scoring?
user: Funding recency, tech stack overlap, and intent signals. Focus on RevOps tools.
assistant: Based on initial signals, Acme shows strong fit. Want me to enrich further?
user: Yes, pull technographics and intent. Score it against our tier-1 criteria."""

current_message = "Score Acme Corp and draft a personalized email to the VP Sales Ops."

output_reservation = 2048
context_window = 128000

segments = {
    "system_prompt": system_prompt,
    "tool_definitions": tool_definitions,
    "retrieved_context": retrieved_context,
    "conversation_history": conversation_history,
    "current_message": current_message,
}

token_counts = {name: count_tokens(text) for name, text in segments.items()}
total_input = sum(token_counts.values())
total_reserved = total_input + output_reservation
headroom = context_window - total_reserved

print(f"{'Segment':<25} {'Tokens':>8}")
print("-" * 35)
for name, count in token_counts.items():
    print(f"{name:<25} {count:>8}")
print("-" * 35)
print(f"{'Total input':<25} {total_input:>8}")
print(f"{'Output reservation':<25} {output_reservation:>8}")
print(f"{'Total reserved':<25} {total_reserved:>8}")
print(f"{'Context window':<25} {context_window:>8}")
print(f"{'Headroom':<25} {headroom:>8}")
print(f"{'Utilization':<25} {total_reserved / context_window:>7.1%}")
```

Running this produces a segment-by-segment breakdown. You see exactly which segment is eating budget and can decide what to cut before the call.

### Memory Architecture Comparison

Now compare the three memory strategies on the same conversation. This script simulates a 20-turn conversation and measures how each strategy scales.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

def count_tokens(text):
    return len(enc.encode(text))

turns = []
for i in range(20):
    turns.append(f"user: Question {i+1} about account {i+1}.")
    turns.append(f"assistant: Answer {i+1}: Based on the data, account {i+1} shows strong signal.")

full_history = "\n".join(turns)

window_size = 6
sliding_window = "\n".join(turns[-window_size:])

summary = """SUMMARY: User asked about 20 accounts. Accounts 1-10 were tier-2 fit.
Accounts 11-18 were tier-1 fit with strong intent signals. Accounts 19-20 were
disqualified due to budget constraints."""
recent_turns = "\n".join(turns[-4:])
summarized = summary + "\n" + recent_turns

strategies = {
    "full_history": full_history,
    "sliding_window_6": sliding_window,
    "summarization": summarized,
}

print(f"{'Strategy':<22} {'Tokens':>8}  {'Turns Kept':>10}")
print("-" * 44)
for name, text in strategies.items():
    print(f"{name:<22} {count_tokens(text):>8}  {text.count('user:') + text.count('assistant:'):>10}")
```

The output is stark. Full history scales linearly — at 20 turns it is already expensive. Sliding window holds constant. Summarization costs slightly more than sliding window but preserves the thematic arc. The point is not that one is universally better — it is that the token cost is measurable and the tradeoff is explicit before the call.

## Use It

Dynamic context budget allocation is the mechanism behind every RAG-powered account-scoring workflow in GTM engineering. This slice shows a context assembler that checks headroom, truncates retrieved context when over budget, and returns the assembled prompt — the same pattern you would wrap around any Clay enrichment waterfall, Salesforce data pull, or Apollo intent lookup before sending results to an LLM for prioritization.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

def assemble_context(system, retrieved_chunks, history, query, window=128000, out_reserve=4096):
    budget = window - out_reserve
    sys_t = len(enc.encode(system))
    hist_t = len(enc.encode(history))
    query_t = len(enc.encode(query))
    remaining = budget - sys_t - hist_t - query_t

    kept = []
    retrieved_t = 0
    for chunk in retrieved_chunks:
        chunk_t = len(enc.encode(chunk))
        if retrieved_t + chunk_t > remaining:
            break
        kept.append(chunk)
        retrieved_t += chunk_t

    total = sys_t + hist_t + query_t + retrieved_t
    print(f"Budget: {budget} | Used: {total} | Headroom: {budget - total} | Chunks: {len(kept)}/{len(retrieved_chunks)}")
    return f"{system}\n\n" + "\n".join(kept) + f"\n\n{history}\n\n{query}"

prompt = assemble_context(
    system="Score this account for tier-1 ICP fit. Output JSON with score 1-5 and justification.",
    retrieved_chunks=[
        "Acme Corp: Series B, $18M ARR, 240 employees, uses Salesforce, HubSpot, dbt.",
        "Intent: elevated searches for 'revenue operations platform' (+340% MoM).",
        "Funding: $25M Series B led by Bessemer, announced 11 days ago.",
        "Competitor: currently using Outreach (your direct competitor).",
        "Decision maker: VP Sales Ops, active on LinkedIn, posted about RevOps tooling.",
        "Long chunk about historical account data that may or may not be relevant here.",
    ],
    history="user: Score Acme Corp\nassistant: Pulling firmographics now...",
    query="Finalize the score."
)
```

This is the Cluster 1.2 pattern (TAM Refinement & ICP Scoring). Every account-scoring enrichment that calls an LLM must pass through a budget gate. The assembler above is the minimum viable version: it measures, it truncates retrieved context first, it never silently fails on the LLM call. In production you would add reranking on the retrieved chunks, a sliding-window fallback for history, and an alert when headroom drops below a threshold. But the architecture — measure before you call, sacrifice retrieved context first, never sacrifice the output reservation — holds at every scale.

## Exercises

### Exercise 1: Overflow Diagnosis (Easy)

Take the Token Budget Reporter script. Change `context_window` from `128000` to `4000`. Run it. Now change `output_reservation` from `2048` to `0`. Run it again. Observe which segment the utilizer hits first. Then modify the script to print a warning when utilization exceeds 85% — a soft threshold you set before the hard boundary.

**Expected output:** The script prints a `WARNING: utilization at XX% — approaching context limit` message when the threshold is crossed.

### Exercise 2: Adaptive Memory Strategy (Medium)

Build on the Memory Architecture Comparison script. Write a function `adaptive_memory(turns, budget, strategy="auto")` that selects between full history, sliding window, and summarization based on available budget. When full history fits, use it. When it does not, try sliding window with N=6. If that still does not fit, fall back to a one-sentence summary plus the last two turns. The function returns the assembled history string and the strategy name it selected. Test it with budgets of 2000, 500, and 100 tokens on the same 20-turn conversation.

## Key Terms

**Context Window**: The maximum number of tokens a model can process in a single forward pass. Fixed per model. Not persistent across calls.

**Token Budget**: The allocation of available input tokens across prompt segments (system, tools, history, retrieved context, current message) plus reserved output tokens. Every segment competes for the same finite space.

**BPE (Byte Pair Encoding)**: The tokenization scheme used by modern LLMs. Frequently occurring subword sequences become single tokens; rare words fragment into multiple. JSON and code are token-dense.

**Sliding Window Memory**: A memory strategy that retains the last N conversation turns and drops older ones. Token cost is constant per call. Trades long-range recall for predictable budget.

**Summarization Memory**: A memory strategy that compresses older conversation turns into a summary and appends recent turns verbatim. Trades exact quotes for thematic recall at lower token cost.

**Lost in the Middle**: The observed attention degradation where models attend to content at the beginning and end of a prompt more reliably than content in the middle. Impacts retrieval ordering decisions.

**Output Reservation**: The token budget set aside for the model's response. If not reserved explicitly, the model may truncate its own output when the input fills the window.

## Sources

- OpenAI. *tiktoken: Fast BPE tokeniser for use with OpenAI's models.* GitHub repository, https://github.com/openai/tiktoken
- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* Transactions of the Association for Computational Linguistics, 11, 157–173. https://arxiv.org/abs/2307.03172
- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* Proceedings of ACL 2016. https://arxiv.org/abs/1508.07909
- [CITATION NEEDED — concept: token budget accounting across prompt segments in production systems]