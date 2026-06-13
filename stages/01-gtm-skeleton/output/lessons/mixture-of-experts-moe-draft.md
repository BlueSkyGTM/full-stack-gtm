# Mixture of Experts (MoE)

## Hook It
Most scaling strategies make every parameter do work on every token. MoE breaks that contract — a gating network decides which subset of parameters (which "experts") earn the right to compute. You get a 8x parameter count with only 2x the FLOPs. This is the architecture behind Mixtral, DeepSeek, and rumored GPT-4.

## Map It
Sparse MoE vs dense models. The three components: expert networks (feed-forward layers, not separate models), the router/gating network (usually a linear projection + softmax), and the load-balancing auxiliary loss that prevents the router from collapsing onto one expert. Top-k routing (typically top-2) as the standard selection mechanism. Token-choice routing vs expert-choice routing.

## Build It
Implement a minimal MoE layer from scratch: a router that assigns tokens to experts, expert feed-forward networks that process only their assigned tokens, and the merge step that recombines outputs. Print routing distributions, expert utilization, and output shapes at every stage.

- **Easy**: Build a router that assigns 8 tokens to 4 experts using top-2 routing. Print the assignment mask.
- **Medium**: Complete the MoE layer with expert FFNs and sparse dispatch. Print per-expert token counts and output tensor shape.
- **Hard**: Add an auxiliary load-balancing loss. Train the MoE on a toy classification task and print expert utilization per epoch.

## Use It
The routing mechanism in MoE mirrors the Clay waterfall in GTM enrichment: a signal enters, a routing decision determines which enrichment provider (expert) handles it, and the result merges back. When you configure conditional branching in a Clay waterfall — "if company size > 500, route to ZoomInfo; else route to Apollo" — you are implementing a hand-coded gating function. MoE makes this routing learnable. [CITATION NEEDED — concept: Clay waterfall conditional routing as gating mechanism]

## Ship It
MoE models ship with specific operational constraints: uneven GPU memory usage across devices (expert parallelism requires careful placement), higher VRAM for the same active parameter count (all experts must live in memory even if inactive), and routing instability under distribution shift. Deploy Mixtral-8x7B via vLLM or Ollama with expert parallelism configured. Monitor expert utilization metrics to detect routing collapse in production.

- **Easy**: Load Mixtral-8x7B-Instruct via Ollama. Run three prompts from different domains and log which tokens the router likely activated (via logits or generation metadata).
- **Medium**: Deploy Mixtral with vLLM using tensor parallelism across 2 GPUs. Print throughput and compare to a dense 7B model on the same hardware.
- **Hard**: Benchmark expert utilization on a domain-shifted dataset. Identify and print the ratio of tokens routed to top-1 vs top-2 experts across domains.

## Push It
Expert-choice routing reverses the decision: instead of tokens choosing experts, experts choose their top-k tokens. This solves load imbalance by construction. Sparse upcycling: converting a dense checkpoint into an MoE by duplicating the FFN and adding noise to the router so experts specialize during fine-tuning. Fine-grained experts (splitting each FFN into more, smaller experts) and shared experts (always-active experts that capture common patterns). DeepSeek-V2's approach: shared + routed experts with top-6 routing from 64 experts.

- **Easy**: Modify the Build It MoE to use expert-choice routing. Print per-expert token counts before and after the switch.
- **Medium**: Implement sparse upcycling. Take a trained dense FFN checkpoint, create an MoE with 8 experts initialized from it, and fine-tune on the same data. Compare loss curves.
- **Hard**: Implement shared + routed experts (DeepSeek-style). Ablate the shared expert count and plot downstream task accuracy vs expert utilization entropy.