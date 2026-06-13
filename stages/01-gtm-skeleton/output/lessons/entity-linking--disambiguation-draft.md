# Entity Linking & Disambiguation

## Hook It
You extracted "Apple" from a blog post. Is it the fruit, the record label, or the company buying your competitor's product? Entity linking is the step that turns ambiguous text mentions into canonical, joinable identifiers. Without it, your enrichment pipelines fuse data to the wrong row.

## Ground It
Establish the three-stage pipeline: mention detection, candidate generation, and disambiguation. Cover the ambiguity problem classes (synonymy, polysemy, novelty) and the role of a knowledge base as the source of truth. Explain scoring mechanisms—string similarity, context overlap, and popularity priors—before introducing any library.

## Build It
Implement a minimal entity linker in Python that takes raw text, finds candidate matches in a small knowledge base using TF-IDF + cosine similarity, and resolves the best candidate. Observable output: print the input text, extracted mentions, candidate scores, and final linked entity IDs.

Exercises:
- **Easy**: Add a new entity to the knowledge base and confirm it appears in candidate output.
- **Medium**: Replace TF-IDF with sentence-transformer embeddings and compare rankings.
- **Hard**: Implement a coherence score that penalizes inconsistent entity types across multiple linked mentions in the same document.

## Use It
Redirect to **Zone 02 — Enrichment**. In a Clay waterfall, company names from intent signals are ambiguous strings. Entity linking maps those strings to canonical domains and LinkedIn slugs before the waterfall fires, preventing wrong-row enrichment. If the concept does not map to a live Clay feature, note it as foundational for Zone XX.

## Ship It
Cover latency budgets (candidate generation is the bottleneck), caching strategies for repeated surface forms, and handling NIL (no-match) entities. Explain evaluation: micro-precision/recall on a held-out set of annotated mentions. Flag the production risk of stale knowledge bases and the need for periodic rebuilds.

## Extend It
Point to neural end-to-end linkers (GENRE, REL) that replace the staged pipeline with a single sequence-to-sequence model. Mention cross-lingual linking, domain-specific KB construction, and the open problem of temporal entity drift—when the correct mapping changes over time.

---

**GTM Redirect:** Zone 02 — Enrichment. Specifically: canonicalizing company/person mentions before Clay waterfall execution. [CITATION NEEDED — concept: Clay entity resolution in enrichment waterfall]