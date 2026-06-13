# RAG Evaluation: Precision, Recall, MRR, nDCG, Faithfulness, Answer Relevance

---

## Beat 1: Hook — Why This Lesson Exists

You built a RAG pipeline. A user asks a question. Your system retrieves documents, feeds them to an LLM, and returns an answer. How do you know if it's any good? Without evaluation metrics, you're eyeballing outputs and guessing. This lesson gives you six quantitative signals to replace those guesses. Half measure retrieval quality (did you fetch the right chunks?). Half measure generation quality (did the model produce a grounded, relevant answer?).

---

## Beat 2: Concept — The Six Metrics and What They Capture

Split the metrics into two groups and explain the mechanism of each:

**Retrieval Metrics** — evaluate the fetch step, before generation:
- **Precision@K**: Of the K chunks retrieved, what fraction are relevant? Mechanism: count true positives in the result set, divide by K.
- **Recall@K**: Of all relevant chunks that exist in the corpus, what fraction did you retrieve? Mechanism: count true positives in result set, divide by total relevant items.
- **MRR (Mean Reciprocal Rank)**: How high in the ranked list does the first relevant chunk appear? Mechanism: reciprocal of the rank position of the first relevant result, averaged across queries.
- **nDCG (Normalized Discounted Cumulative Gain)**: Does your ranking put the most relevant chunks first? Mechanism: sum relevance scores discounted by log of position, normalized against the ideal ranking.

**Generation Metrics** — evaluate the output step, after generation:
- **Faithfulness**: Does the answer contain claims not supported by the retrieved context? Mechanism: decompose answer into atomic claims, check each against source chunks.
- **Answer Relevance**: Does the answer actually address the question asked? Mechanism: compare semantic similarity between the question and the answer, or use an LLM judge to score relevance.

Explain that retrieval metrics require labeled relevant documents (ground truth). Generation metrics can use LLM-as-judge or heuristic approaches. Name tools that implement these: RAGAS (open-source framework implements faithfulness and answer relevance via LLM judging), DeepEval (implements all six with configurable judges).

---

## Beat 3: Demo — Computing All Six Metrics Against a Toy Corpus

Show a complete working Python script that:
1. Defines a toy document corpus (5-10 short text chunks)
2. Defines 3 queries with ground-truth relevant documents labeled
3. Simulates retrieval results (ranked lists of chunk IDs)
4. Computes Precision@K, Recall@K, MRR, and nDCG from the retrieval results vs. ground truth
5. Simulates a generated answer for each query
6. Implements a simple faithfulness check (does the answer contain claims not in retrieved chunks?)
7. Implements a simple answer relevance check (keyword overlap or embedding similarity)
8. Prints all metric values in a table

All code runs without modification. All output is printed to terminal. No browser. No API calls for the demo — use deterministic heuristics for faithfulness and relevance to keep it runnable.

---

## Beat 4: Use It — Apply Metrics to Your Own RAG Pipeline

Exercise hooks:

- **Easy**: Compute Precision@5 and Recall@5 for 3 queries against a provided mini-corpus. Print results. Given: corpus, queries, retrieval function, ground truth labels.
- **Medium**: Implement nDCG calculation for a ranked retrieval result. Compare two different retrieval strategies (keyword vs. embedding similarity) on the same queries and print which ranks better.
- **Hard**: Build a faithfulness checker that decomposes an LLM answer into claims and checks each against retrieved chunks. Use an LLM API call to decompose claims, then use string matching to verify support. Score 3 generated answers and print faithfulness scores.

GTM redirect: This is the evaluation layer for any Zone 3 AI solution that uses retrieval — sales enablement chatbots, support knowledge base assistants, account research tools. If your GTM stack includes a RAG-powered assistant (e.g., a Clay agent querying your knowledge base), these metrics tell you whether it's retrieving the right Playbook entries and generating grounded responses. Foundational for Zone 3 (AI Solutions) deployment confidence.

---

## Beat 5: Ship It — Build an Evaluation Harness for Production RAG

Wire the metrics into a repeatable evaluation script that:
1. Loads a test set (queries + ground truth relevant doc IDs + reference answers)
2. Runs your actual RAG pipeline (retrieve + generate) for each query
3. Computes all six metrics
4. Outputs a JSON report with per-query and aggregate scores
5. Fails a CI check if any metric drops below a threshold you set

Exercise hooks:

- **Easy**: Take the demo script and modify it to read queries from a JSON file instead of hardcoding them. Print the same metrics table.
- **Medium**: Add a threshold check: if faithfulness drops below 0.8 or answer relevance below 0.7, exit with code 1 (CI failure). Test it against a provided test set.
- **Hard**: Implement a full eval harness that compares two RAG configurations (e.g., different chunk sizes or embedding models) side by side. Print a comparison table showing which configuration wins on each metric. Run it against 10+ queries.

GTM redirect: When deploying a RAG-based sales assistant or support bot to your GTM team, this eval harness is what tells you "this version is better than the last version" before you push to production. Maps to Zone 3 AI solution reliability. If you're using Clay to surface account insights via retrieval, this ensures the surfaced insights are actually grounded.

---

## Beat 6: Quiz — Verify the Mechanisms

Five questions grounded in the metrics taught. Not trivia — mechanism questions. Each maps to an objective from Beat 2.

1. You retrieve 10 chunks. 4 are relevant. 6 relevant chunks exist in the corpus total. What is Precision@10? What is Recall@10?
2. Your retrieval system returns relevant chunks at positions 1, 5, and 8. What is the MRR for this query?
3. System A returns relevant chunks at positions [1, 2, 8]. System B returns them at [3, 4, 5]. Which has higher nDCG? Explain why position discounting causes this.
4. A generated answer contains the claim "Company X raised $50M in Series B." The retrieved chunks mention "Company X" and "Series B" but no dollar amount. Is this answer faithful? Explain the faithfulness violation.
5. You want to detect when your RAG system hallucinates. Which metric do you monitor? If you also want to catch when the system retrieves irrelevant chunks that waste context window space, which retrieval metric do you add?

---

*GTM cluster mapping: Foundational for Zone 3 (AI Solutions). Direct application to RAG-powered GTM tools — sales enablement chatbots, support assistants, account research agents. Evaluation is what separates a demo from a deployment.*