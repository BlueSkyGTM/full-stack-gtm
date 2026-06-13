# Norms and Distances

## Learning Objectives

1. Compute L1, L2, and L∞ norms on numeric vectors and explain when each is appropriate
2. Implement Euclidean, Manhattan, and cosine distance from first principles without library calls
3. Diagnose when cosine distance outperforms Euclidean for high-dimensional embedding similarity
4. Build a nearest-neighbor ranking function that returns sorted results with distance scores

---

## Beat 1: Hook

You have two embedding vectors from accounts in your CRM. How do you decide they're "similar"? Without a distance function, two 384-dimensional vectors are just arrays of numbers. Norms and distances are the bridge between raw vector output and actionable similarity scores — every lookalike model, clustering pipeline, and retrieval system depends on them.

---

## Beat 2: Concept

A **norm** measures the length or magnitude of a single vector. A **distance** measures how far apart two vectors are. Distance is derived from norms: Euclidean distance is the L2 norm of the difference vector. Cosine distance is derived from dot products normalized by L2 norms. Introduce L1 (Manhattan), L2 (Euclidean), and L∞ (max) norms with geometric intuition — L1 is grid walking, L2 is straight line, L∞ is chess king. Show that distance = norm(A - B) for the L1 and L2 cases.

---

## Beat 3: Mechanism

Implement all three norms and three distances (Euclidean, Manhattan, cosine) from scratch in Python using only `math` — no numpy, no scipy. Print outputs for three fixed 4-dimensional vectors so the practitioner can verify by hand. Then show the same computation using numpy one-liners (`np.linalg.norm`, `scipy.spatial.distance.cosine`) and confirm the outputs match. Demonstrates that the library functions are syntactic sugar over the same math.

**Exercise hooks:**
- **Easy:** Compute L1 and L2 norms of a given vector by hand, then verify with code
- **Medium:** Given three vectors, rank them by cosine distance from a query vector — predict the ranking before running code
- **Hard:** Implement a function that takes a query vector and a list of candidate vectors, returns the top-K nearest using a configurable distance metric (Euclidean or cosine)

---

## Beat 4: Use It

This maps to **Zone 2 (Enrichment)** and **Zone 3 (Scoring)** in the GTM topic map. When you embed company descriptions, tech stack tokens, or intent signals into vectors, cosine distance is the standard metric for ranking similarity — it ignores magnitude and compares direction only, which is what you want when comparing companies of different sizes on behavioral signals. Show a minimal example: embed three company descriptions using a sentence-transformer model, compute pairwise cosine distances, and identify the two closest matches. This is the same mechanism behind Clay's "lookalike" scoring and any waterfall enrichment step that scores fit.

---

## Beat 5: Ship It

Build a reusable `nearest_neighbors.py` script that: loads a JSON file of named entities with embedding vectors, accepts a query name, computes cosine distance against all others, and prints the top 3 matches with scores. Input file provided as a static fixture. Output is printed rankings — observable, testable, no browser required.

**Exercise hooks:**
- **Easy:** Run the script on the provided fixture, verify the output matches expected rankings
- **Medium:** Add a `--metric` flag that switches between cosine and Euclidean, observe how rankings change and explain why
- **Hard:** Replace the static fixture with live embeddings generated from a sentence-transformer model, embed five real company descriptions, and ship the full pipeline end-to-end

---

## Beat 6: Evaluate It

**Questions target:**
- Which norm is invariant to vector magnitude and why that matters for text embeddings
- Given two vectors, calculate L2 distance by hand and confirm against code output
- Why cosine distance can disagree with Euclidean ranking on high-dimensional sparse data
- Identify the failure mode: when Euclidean gives a "close" result that cosine rejects (magnitude-dominated case)

No quiz bank written here — quiz questions must be written against `docs/en.md` objectives per the rules. This section confirms the four testable angles listed above.

---

## GTM Redirect Rules Applied

- **Use It (Beat 4):** Explicitly names Zone 2/Zone 3 enrichment and scoring, references cosine similarity for account matching and lookalike modeling, cites the Clay waterfall mechanism where applicable
- **Ship It (Beat 5):** The nearest-neighbor script is the same retrieval pattern used in GTM enrichment waterfalls — rank candidates by embedding similarity before piping into scoring

---

## Citation Check

[CITATION NEEDED — concept: Clay waterfall uses cosine similarity for lookalike scoring in enrichment steps]