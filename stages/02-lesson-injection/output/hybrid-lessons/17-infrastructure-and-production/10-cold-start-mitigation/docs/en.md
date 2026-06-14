# Cold Start Mitigation for Serverless LLMs

## Learning Objectives

- Enumerate the five layers of cold-start mitigation and name one implementation at each layer.
- Compute total cold-start time as a sum of node provision, weight download, VRAM loading, and engine initialization for a given model size.
- Build a keep-alive scheduler that maintains warm instances below a platform's idle timeout.
- Compare the cost-latency tradeoffs of provisioned concurrency, keep-alive pings, weight caching, and warm pools.
- Trace a cold start event through each phase and identify which mitigation layer would eliminate the latency spike.

## The Problem

Your enrichment endpoint returns in 200ms on request 47. Request 1 took 9 seconds. That gap is a cold start — the container spinning up, the model weights loading into GPU memory, the runtime initializing. In a GTM pipeline scoring 10,000 accounts overnight, one