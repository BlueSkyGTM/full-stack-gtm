## Ship It

Production summarization fails when the input is messy. Earnings call transcripts have speaker labels, crosstalk, and filler. HTML-stripped filings have broken sentences where navigation elements were removed. Concatenated multi-source data has conflicting information (two providers report different employee counts). Before the summarization call, you need a normalization pass: strip speaker labels, fix sentence boundaries, deduplicate sources, and flag conflicts for the model to resolve in the prompt. The summarization endpoint is cheap — the engineering around input normalization and output enforcement is where the time goes.

Token counting before the API call is non-negotiable. If you send 9,000 tokens to a model with an 8,000-token limit, the API will truncate the input silently (depending on the provider) or return an error. Either way, you lose the end of the document — and in earnings calls, the Q&A section at the end is often where the most valuable signals are. Count tokens, compare against the model's limit minus the space you need for the prompt and output, and chunk if necessary. The Anthropic SDK exposes a `count_tokens` method; OpenAI's `tiktoken` library does the same. Never estimate — count.

```python
import anthropic
import os
import re

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def count_tokens(text):
    return len(client.messages.count_tokens(
        model="claude-sonnet-4-22662050",
        messages=[{"role": "user", "content": text}]
    ).input_tokens)

def chunk_text(text, max_tokens_per_chunk=3000, overlap_tokens=300):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_tokens = 0
    overlap_text = ""
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens_per_chunk and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_sentences = []
            overlap_tok = 0
            for s in reversed(current_chunk):
                s_tok = count_tokens(s)
                if overlap_tok + s_tok > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_tok += s_tok
            current_chunk = list(overlap_sentences) if overlap_sentences else []
            current_tokens = sum(count_tokens(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_chunk(text):
    response = client.messages.create(
        model="claude-sonnet-4-22662050",
        max_tokens=500,
        messages=[{"role": "user", "content": f"Summarize the key facts, numbers, and named entities in this text. Be concise.\n\nTEXT:\n{text}"}]
    )
    return response.content[0].text

def reduce_summaries(summaries):
    combined = "\n\n".join(f"Section {i+1}:\n{s}" for i, s in enumerate(summaries))
    response = client.messages.create(
        model="claude-sonnet-4-22662050",
        max_tokens=300,
        messages=[{"role": "user", "content": f"Synthesize these section summaries into a single 3-bullet brief. Preserve all numbers and named entities.\n\n{combined}"}]
    )
    return response.content[0].text

def map_reduce_summarize(text):
    total_tokens = count_tokens(text)
    print(f"Document tokens: {total_tokens}")
    if total_tokens <= 4000:
        print("Within single-pass limit. No chunking needed.")
        return summarize_chunk(text)
    chunks = chunk_text(text, max_tokens_per_chunk=3000, overlap_tokens=300)
    print(f"Split into {len(chunks)} chunks with 300-token overlap")
    summaries = []
    for i, chunk in enumerate(chunks):
        chunk_tokens = count_tokens(chunk)
        print(f"  Chunk {i+1}: {chunk_tokens} tokens")
        summary = summarize_chunk(chunk)
        summaries.append(summary)
        print(f"  Summary {i+1}: {summary[:100]}...")
    print(f"\nReducing {len(summaries)} summaries...")
    final = reduce_summaries(summaries)
    return final

long_document = """
Acme Corp announced third quarter revenue of $42.3 million, up 12 percent year over year.
The enterprise segment drove growth with 18 percent increase in new logo acquisitions.
CEO Jane Chen highlighted EMEA expansion during the earnings call.
CFO David Park cautioned that supply chain disruptions could impact Q4 margins by 200 basis points.
Full-year guidance was raised to $170 million from the previous range of $165 to $168 million.
Morgan Stanley upgraded the stock from equal-weight to overweight.
The new AI-powered analytics platform launches in January with three Fortune 500 commitments.
Operating margins improved to 23 percent from 21 percent in the prior quarter.
The board authorized a $50 million share repurchase program.
Competitor NetSync Holdings reported a revenue decline of 5 percent, losing market share in the mid-market segment.
Acme's chief product officer Sarah Williams departed to join a startup, creating a leadership gap.
The company hired 340 new engineers in Q3, bringing total headcount to 2,100.
Research and development spend increased to $8.2 million, representing 19 percent of revenue.
International revenue now accounts for 31 percent of total, up from 27 percent a year ago.
Deferred revenue grew to $28 million, suggesting strong future contracted revenue.
The company's net retention rate stands at 118 percent, indicating existing customers are expanding.
""" * 3

final_summary = map_reduce_summarize(long_document)
print("\n" + "="*60)
print("FINAL REDUCED SUMMARY:")
print("="*60)
print(final_summary)
```

Output format enforcement is the last production concern. If your downstream system expects exactly three bullets, the model will sometimes return two or four. Structured prompts ("Return exactly 3 bullet points, each starting with a bullet character") help, but tool use is more reliable — define a schema that accepts an array of exactly three strings, and the model is constrained to match it. The latency budget matters too: map-reduce is N+1 sequential calls (N chunk summaries plus 1 reduce). If each call takes 4 seconds and you have 6 chunks, that's 28 seconds before the brief appears. For synchronous workflows (AE waiting for a brief), that's acceptable. For batch enrichment of thousands of accounts, parallelize the map step.