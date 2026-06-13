# Capstone 07 — End-to-End Fine-Tuning Pipeline (Data to SFT to DPO to Serve)

## Hook

You've done SFT in isolation. You've done DPO in isolation. Now you string the full pipeline together: raw data ingestion → supervised fine-tuning → preference alignment → inference endpoint. This is the loop that ships custom models to production.

## Concept

Four stages, each with a distinct optimization objective and data contract. SFT teaches the model *what* to say via maximum likelihood on labeled examples. DPO teaches it *what not to say* by contrasting preferred vs rejected completions against a reference policy. The output of SFT becomes the reference model for DPO. The output of DPO becomes the model you serve. Break the ordering and you get a model that is either unaligned or has forgotten its task instruction.

## Demo

Single Python script that: (1) loads a JSONL dataset of GTM-relevant prompt/completion pairs, (2) runs SFT using `trl`'s `SFTTrainer` on a small open model (Qwen2.5-0.5B), (3) pairs the SFT checkpoint with a preference dataset to run DPO via `DPOTrainer`, (4) loads the final adapter and runs inference, printing side-by-side outputs from the base vs SFT vs DPO model on the same prompt. All observable via printed output.

## Use It

**GTM Redirect — Zone 3 (Engagement): personalized outbound at scale.**

Fine-tuning a small model on your best-performing outreach sequences gives you a model that writes in your voice, for your ICP, without per-prompt token costs or context window constraints. Run the pipeline on historical email data where the "preferred" examples are replies-booked and "rejected" examples are ignored or unsubscribed.

- *Easy:* Prepare a JSONL file of 20 outreach prompts with chosen/rejected completions pulled from your CRM.
- *Medium:* Run the full SFT→DPO pipeline on email personalization data and compare base vs aligned model outputs on 5 held-out prospects.
- *Hard:* Instrument the pipeline with Weights & Biases logging, run a hyperparameter sweep on learning rate and DPO beta, and document which configuration produces the highest win-rate on held-out preference pairs.

## Ship It

Package the final DPO-aligned adapter into a vLLM or Ollama serve configuration. Write a shell script that: starts the server, sends a batch of 10 GTM prompts via curl, captures latency and output, and validates that responses conform to a schema (JSON mode or constrained decoding). Print the full batch results and p50/p95 latency.

## Evaluate

Three prompts that test whether the practitioner can diagnose pipeline failures: (1) identify why DPO output degraded because SFT was skipped, (2) trace a data formatting bug in the preference JSONL that causes a silent trainer failure, (3) explain why serving the base model instead of the DPO adapter produces higher-loss but "safer" outputs. Each scenario requires the practitioner to reason about the mechanism, not the API call.