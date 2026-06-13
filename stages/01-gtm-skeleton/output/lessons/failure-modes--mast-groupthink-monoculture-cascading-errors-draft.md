# Failure Modes — MAST, Groupthink, Monoculture, Cascading Errors

---

## Hook It

Opens with a scenario where an AI-powered GTM stack silently degrades: three agents fed the same training corpus all recommend the same ICP shift, nobody catches it, pipeline collapses. Names the four failure modes that caused it and previews the detection patterns.

---

## Map It

Explains each failure mechanism and how they compound:

- **MAST (Multi-Agent System Trust breakdown):** When agents trust each other's outputs without independent verification, trust becomes a vulnerability. [CITATION NEEDED — concept: MAST acronym origin and formal definition in multi-agent failure literature]
- **Groupthink:** Agents converge on the same answer because they share context windows, training data, or reward signals — not because the answer is correct.
- **Monoculture:** Homogeneous model selection (e.g., one provider for enrichment, scoring, copy generation, and routing) creates a single point of semantic failure.
- **Cascading Errors:** An upstream error (bad firmographic enrichment) propagates downstream through scoring, sequencing, and outreach — amplifying rather than attenuating.

Diagrams the compounding chain: Monoculture enables Groupthink, which accelerates Cascading Errors, which MAST prevents you from catching.

---

## Code It

**Easy hook:** Build a simulation where three agents vote on a classification. All three share the same training bias — demonstrate groupthink by showing 100% agreement on a wrong answer.

**Medium hook:** Inject a deliberate upstream error into a 3-step pipeline (enrich → score → route). Print intermediate outputs to show error amplification at each stage.

**Hard hook:** Implement a simple trust-weighted voting system where agents have independent verification. Compare the failure rate of the trusted-only chain vs. the verified chain across 1000 simulated runs. Print the delta.

All code runs in terminal, prints observable output, no browser dependency.

---

## Use It

Maps each failure mode to a GTM engineering context:

- **Monoculture → Vendor concentration risk:** Using one LLM provider for Clay enrichment, Apollo scoring, and Lavender copy generation. If that provider's semantic model shifts, everything shifts together.
- **Groupthink → ICP validation loops:** When your research agent, scoring agent, and copy agent all pull from the same CRM data without independent signal sources, they will agree — and they will be wrong together.
- **Cascading Errors → Waterfall pipeline fragility:** A Clay waterfall that passes bad data from step 1 into steps 2-5 without checkpoint validation. Redirect: this is the Clay waterfall failure pattern, mapped to Zone 2 (Enrichment) in the GTM topic map.

---

## Ship It

Provides a production-ready detection checklist:

- Run semantic diversity audits across agent outputs weekly — if cosine similarity between agent recommendations exceeds a threshold, flag monoculture risk.
- Add checkpoint assertions between pipeline stages: if enrichment confidence < threshold, halt before scoring.
- Maintain two independent model providers for any critical GTM decision path.
- Log trust chains: trace which agent influenced which downstream decision, so cascading errors are debuggable.

Exercise hook: given a failing pipeline output log, trace the cascading error back to its origin and write the checkpoint assertion that would have caught it.

---

## Cement It

Recaps the four failure modes and their signatures. Extends to open questions: How do you balance trust and verification without paralyzing throughput? When is monoculture acceptable (cost, latency constraints) vs. unacceptable (decision-critical paths)?

Suggests the practitioner audit their current GTM stack for each failure mode and document which zones carry the highest concentration risk. Redirects to Zone 2 (Enrichment) and Zone 4 (Activation) in the GTM topic map as the highest-risk zones for cascading failures.