# EchoLeak and the Emergence of CVEs for AI

## Hook (Beat 1)
In January 2025, researcher Samon Jovani disclosed EchoLeak (CVE-2025-30158), an indirect prompt injection vulnerability in Microsoft 365 Copilot that exfiltrated sensitive data through a zero-click markdown image tag. AI systems now have their own CVEs. This changes how practitioners evaluate, deploy, and defend AI-integrated tools.

## Concept (Beat 2)
EchoLeak exploits the architecture of retrieval-augmented AI assistants: an attacker-controlled document enters the retrieval corpus, the LLM processes it as a legitimate instruction, and output is exfiltrated via rendered markdown. The pattern is indirect prompt injection with out-of-band data exfiltration. What makes this a CVE rather than "expected behavior" is that the vendor explicitly claimed guardrails prevented it.

## Mechanism (Beat 3)
Walk through the EchoLeak attack chain step by step: (1) attacker sends email with embedded markdown image pointing to attacker-controlled URL, (2) Copilot retrieves and processes the email during a user query, (3) the LLM interprets the markdown as an instruction to embed sensitive context into the image URL parameters, (4) the rendered markdown triggers an HTTP request to the attacker's server with the exfiltrated data. Map this to the OWASP LLM Top 10 categories (LLM01: Prompt Injection, LLM06: Sensitive Information Disclosure). Contrast with direct prompt injection where the attacker has access to the input field.

## Code (Beat 4)
Build a minimal, safe demonstration of the pattern. Python script simulates three components: a "retrieval corpus" with a crafted document, an LLM call that processes user query plus retrieved document, and a mock HTTP listener that would receive the exfiltrated data. The script demonstrates the mechanism without targeting any real system. Observable output shows the full attack chain in terminal.

## Use It (Beat 5)
[CITATION NEEDED — concept: GTM cluster mapping for AI security evaluation in procurement workflows]

When evaluating AI-powered GTM tools (Clay, Apollo, ChatGPT plugins, custom GPTs), practitioners must now assess: does the tool retrieve and process external content? Does it render markdown or HTML in outputs? Does the vendor publish a security posture document addressing indirect prompt injection? Map this to Zone 03 (Enrichment) and Zone 05 (Orchestration) where AI tools most commonly process untrusted external data. Build a security evaluation checklist for any AI tool that handles prospect data.

## Ship It (Beat 6)
Produce a one-page AI Tool Security Evaluation template covering: data flow diagram (what enters the LLM, what leaves), indirect prompt injection surface area, data exfiltration vectors, vendor CVE history, and incident response plan if the tool's AI layer is compromised. This is now a standard part of vendor evaluation in any AI-augmented GTM stack.

---

**Exercise Hooks:**
- Easy: Trace the EchoLeak attack chain from a provided email body to the exfiltrated data endpoint.
- Medium: Modify the simulation script to demonstrate a different exfiltration vector (e.g., base64-encoded URL parameters instead of query strings).
- Hard: Audit a real AI-powered GTM tool's documentation and map its architecture to the OWASP LLM Top 10 categories, identifying which risks apply and whether the vendor addresses them.