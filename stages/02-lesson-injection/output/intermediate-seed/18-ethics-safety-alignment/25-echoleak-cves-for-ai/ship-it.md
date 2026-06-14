## Ship It

The deliverable is a one-page AI Tool Security Evaluation template. Print it, put it in your vendor evaluation folder, and run through it before signing any contract for an AI-powered GTM tool. This is not a comprehensive security audit — it is a first-pass filter that catches the vulnerability patterns demonstrated by EchoLeak, CamoLeak, and their analogues.

```python
TEMPLATE = """
AI TOOL SECURITY EVALUATION
================================================================================
Tool name:           _________________________________
Vendor:              _________________________________
Evaluated by:        _________________________________
Date:                _________________________________

1. DATA FLOW MAP
   What enters the LLM context window? (check all that apply)
   [ ] Prospect data from CRM
   [ ] Scraped website content
   [ ] Email bodies (inbound or outbound)
   [ ] Third-party API responses (Apollo, Clearbit, etc.)
   [ ] User-authored prompts
   [ ] System prompts with proprietary logic

   What leaves the system?
   [ ] LLM output rendered as markdown/HTML in browser
   [ ] LLM output stored in CRM fields
   [ ] LLM output sent via email/messaging
   [ ] API calls to external endpoints
   [ ] Logs containing prompt content

2. INDIRECT PROMPT INJECTION SURFACE
   Does the tool retrieve external content?          [ ] Yes  [ ] No
   Is retrieved content passed to an LLM?             [ ] Yes  [ ] No
   Does the LLM distinguish instructions from data?   [ ] Yes  [ ] No  [ ] Unknown
   Can the LLM make external HTTP requests?           [ ] Yes  [ ] No  [ ] Unknown
   Does output render markdown or HTML?               [ ] Yes  [ ] No
   Are URLs in output redacted or sanitized?          [ ] Yes  [ ] No  [ ] Unknown

3. VENDOR SECURITY POSTURE
   Published prompt injection policy?                 [ ] Yes  [ ] No
   CVE history:                                       ________________
   Bug bounty program?                                [ ] Yes  [ ] No
   Data processing agreement covers AI layer?         [ ] Yes  [ ] No
   Disclosure timeline for past incidents:            ________________

4. INCIDENT RESPONSE PLAN
   What data is in the LLM context window during
   normal operation?                                  ________________
   What prompts/logic would be exposed if the
   AI layer is compromised?                           ________________
   What external endpoints could receive
   exfiltrated data?                                  ________________
   Who is notified if the AI layer is compromised?    ________________
   Rollback procedure (disable AI features):          ________________

5. RISK ASSESSMENT
   Injection surface rating:
     [ ] LOW    (LLM processes only internal data, no external retrieval)
     [ ] MEDIUM (LLM retrieves external data but output is text-only)
     [ ] HIGH   (LLM retrieves external data AND renders markdown/HTML)

   Recommendation:
     [ ] Deploy with monitoring
     [ ] Deploy with output sanitization layer
     [ ] Do not deploy until vendor addresses gaps
================================================================================
"""

print(TEMPLATE)
```

This template exists because the GTM engineering stack in 2025 is not just a set of productivity tools — it is a distributed system where LLMs process untrusted data at scale. Every enrichment waterfall, every AI-personalized outreach sequence, every research agent that scrapes a prospect's website is a node in that system. EchoLeak proved that the vulnerability pattern is real, it is exploitable in production, and it bypasses vendor-claimed guardrails. The evaluation template is how you apply that knowledge to your own stack before someone else demonstrates it for you.