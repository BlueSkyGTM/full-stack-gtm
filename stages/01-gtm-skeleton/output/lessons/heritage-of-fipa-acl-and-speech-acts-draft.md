# Heritage of FIPA-ACL and Speech Acts

## Hook
You've written prompts that treat LLMs like text-in/text-out boxes. But agents that coordinate—negotiating, delegating, confirming—need a shared grammar of *intent*, not just content. Speech act theory is where that grammar comes from.

## Concept
Austin (1962) and Searle (1969) established that utterances *do* things: assert, direct, commit, declare, express. FIPA (1997) codified these into a formal Agent Communication Language with a fixed set of performatives. Trace the lineage from philosophical pragmatics to the ACL envelope every multi-agent framework still inherits.

## Mechanism
Deconstruct a FIPA-ACL message: sender, receiver, performative, content, language, ontology. Map the core performatives (`inform`, `request`, `propose`, `accept-proposal`, `reject-proposal`, `confirm`, `disconfirm`, `query-ref`) to their speech-act category. Show how the protocol layer (e.g., Contract Net, FIPA-Request) uses performative sequencing to enforce conversation state. [CITATION NEEDED — concept: FIPA Contract Net Protocol state machine specification]

## Use It
Build a minimal ACL message parser that classifies performatives into speech-act categories and traces a two-agent negotiation trace. This is foundational for Zone 3 (multi-agent orchestration in GTM workflows) — when your enrichment agents hand off accounts through structured proposals and acceptances rather than ad-hoc JSON. The same performative patterns underpin how Clay's waterfall stages pass decisions downstream.

## Ship It
Exercise hooks:
- **Easy**: Print a FIPA-ACL message, label its performative and speech-act class.
- **Medium**: Implement a conversation-state tracker that confirms a valid `request → agree → inform-done` sequence and rejects out-of-order messages.
- **Hard**: Simulate a three-agent Contract Net negotiation (initiator, participant-a, participant-b) with proposal/accept/reject cycling and observable protocol traces.

## Evaluate
- Classify any given performative into its Searlean speech-act category.
- Identify which performatives must follow which in a FIPA-Request protocol.
- Articulate why raw JSON exchange between agents fails where ACL-style performative semantics succeed.

---

**Learning Objectives**
1. Classify FIPA-ACL performatives by their Searlean speech-act category.
2. Parse and construct valid FIPA-ACL message envelopes.
3. Trace a Contract Net negotiation through its performative sequence.
4. Implement a conversation-state validator that accepts or rejects message order.
5. Explain why performative semantics matter for multi-agent coordination in GTM workflow orchestration.