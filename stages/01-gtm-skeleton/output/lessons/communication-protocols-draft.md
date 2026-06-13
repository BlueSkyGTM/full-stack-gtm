# Communication Protocols

## Hook

Real AI systems fail at the seams, not the center. A model that reasons perfectly but can't receive input or return output in a structured, reliable way is a demo, not a product. This lesson covers the wire-level patterns that let agents, APIs, and humans exchange information without losing meaning.

## Concept

Covers four protocol primitives — request/response, streaming, pub/sub, and RPC — then maps each to serialization formats (JSON, JSONL, protobuf) and transport assumptions (stateless HTTP, persistent WebSocket, message broker). Emphasizes the trade-off: tighter coupling enables richer interaction but increases coordination cost.

## Demo

Three runnable scripts: a synchronous JSON-over-HTTP client/server pair, a streaming JSONL exchange over stdio simulating agent-to-agent output, and a minimal pub/sub pattern using file-based queues. Each script prints observable output confirming message receipt, order, and schema validation.

## Use It

Clay enrichment waterfalls pass data between enrichment steps using structured JSON payloads with defined schemas — each step reads specific fields, writes new fields, and forwards the combined payload downstream. Map the protocol primitives from this lesson to the waterfall's request/response chaining pattern. [GTM cluster: Data Enrichment — Zone 1/2]

## Ship It

**Easy:** Extend the JSON-over-HTTP server to validate incoming payloads against a schema dict and return structured error responses.  
**Medium:** Convert the stdio streaming demo into a WebSocket server that streams agent tokens to a connected client.  
**Hard:** Build a two-agent pipeline where Agent A writes structured tasks to a file-based queue and Agent B consumes, processes, and writes results back using a defined message protocol with acknowledgment.

## Review

Five forced-recall prompts covering: which protocol primitive fits which coupling level, why streaming uses JSONL instead of JSON, what breaks when a pub/sub consumer crashes mid-message, how schema validation differs from type coercion, and where the Clay waterfall maps to request/response vs. pub/sub.