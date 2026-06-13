# FinOps for LLMs — Unit Economics and Multi-Tenant Attribution

## Beat 1: Hook — The Margin That Vanished

A practitioner ships an LLM feature, sets pricing at $50/month per seat, and discovers at scale that one heavy-use tenant burns $47 in API costs alone while another burns $0.30. Flat-rate pricing for variable-cost infrastructure is a margin trap. This lesson builds the attribution machinery to make per-tenant unit economics visible and enforceable.

## Beat 2: Concept — Token-Level Cost Tracing

The mechanism is a two-layer cost model: (1) **direct cost** computed from token counts multiplied by per-model pricing, and (2) **attributed overhead** amortized across tenants using a chosen allocation strategy — equal split, proportional to usage, or weighted by tier. Multi-tenant attribution requires a tenant identifier propagated through every request, a usage ledger that accumulates token consumption per tenant per billing period, and a cost normalization function that maps raw token prices into a consistent unit (e.g., cost per 1M tokens). The key insight: without tenant-scoped aggregation, you have a blended cost number that hides which customers are profitable and which are subsidized.

## Beat 3: Implement — Building the Cost Engine

Working Python code that (a) wraps an LLM call with a decorator that captures input tokens, output tokens, and model ID, (b) writes each invocation to a SQLite ledger with tenant_id, timestamp, and computed cost, and (c) produces a per-tenant unit economics report showing revenue, direct LLM cost, attributed overhead, and gross margin. Observable output: a printed table of tenant-level P&L. Exercise hooks — Easy: modify the per-model pricing table; Medium: implement a proportional overhead allocation strategy instead of equal split; Hard: add a budget enforcement layer that blocks requests when a tenant exceeds their monthly LLM cost cap.

## Beat 4: Use It — GTM Redirect: Zone 03 (ICP & Account Intelligence)

Per-tenant unit economics directly informs ideal customer profile refinement: if 20% of tenants consume 80% of LLM spend, those usage patterns define who your *unprofitable* customers are and what your *ideal* customer is not. The cost ledger becomes an ICP signal — high-margin tenants share detectable usage signatures (low completion tokens, consistent prompt lengths, specific model routing). This is foundational for Zone 03 (ICP & Account Intelligence): account-level cost data feeds back into scoring and qualification. [CITATION NEEDED — concept: ICP refinement from cost-of-service data in GTM pipeline]

## Beat 5: Ship It — Production Safeguards

Production FinOps requires four components: (1) async flush of usage records so latency-sensitive paths aren't blocked by ledger writes, (2) a budget circuit-breaker that returns a graceful degradation response when a tenant hits their cap rather than failing open, (3) a reconciliation job that compares estimated costs against actual provider invoices (providers bill in discrete increments that may differ from real-time token counts), and (4) an alerting threshold that surfaces margin erosion before it compounds across a billing cycle. Exercise hooks — Easy: add a Slack webhook alert when any tenant's margin drops below 40%; Medium: implement reconciliation logic that flags variance > 5% between estimated and invoiced cost; Hard: design a tiered degradation strategy that downgrades model before blocking entirely.

## Beat 6: Review — Mechanism Check

Three diagnostic questions the practitioner should be able to answer after this lesson: (1) Given a raw token count and model, compute the direct cost — this tests the pricing layer. (2) Given three tenants with unequal usage, calculate attributed overhead under proportional allocation — this tests the allocation mechanism. (3) Given a tenant approaching their budget cap, describe the decision sequence for circuit-breaker vs. model downgrade vs. request block — this tests the enforcement layer. Each question maps directly to code built in Beat 3 and extended in Beat 5.