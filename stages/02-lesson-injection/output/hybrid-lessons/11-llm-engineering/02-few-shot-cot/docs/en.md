# Few-Shot, Chain-of-Thought, Tree-of-Thought Prompting

## Learning Objectives

1. Compare few-shot, chain-of-thought, and tree-of-thought prompting by their mechanism of action and tradeoffs in token cost, latency, and accuracy.
2. Implement few-shot classification with a fixed example budget and verify output consistency across multiple runs.
3. Construct chain-of-thought prompts that produce visible intermediate reasoning steps for multi-step tasks.
4. Evaluate whether tree-of-thought exploration yields measurably better outputs than linear chain-of-thought for a given task.
5. Diagnose which prompting strategy fits a task based on task properties — classification, multi-step reasoning, or open-ended planning.

## The Problem

You ship a reply classifier for your sales development reps. The prompt says: "Classify this reply as INTERESTED, OBJECTION, or UNSUBSCRIBE." You test it on ten replies. It gets seven right. Your SDR manager says that is not good enough.

So you add three labeled examples to the prompt — actual replies your team has manually categorized. Accuracy climbs to nine out of ten. Same model, same API, same cost per call. The only change is that the prompt now contains demonstrations the model can pattern-match against. That is few-shot prompting, and the gain is real and measurable.

But then you hit a wall. A reply comes in: "We are evaluating three vendors and your pricing is 20% above the median. Send me your security docs and I will circulate them internally." That reply is simultaneously an objection (pricing), a request for information (security docs), and a buying signal (evaluating vendors). A single label is wrong. What you need is reasoning — the model should work through the reply step by step and assign a primary category with secondary signals flagged. That is chain-of-thought, and it costs you more output tokens in exchange for accuracy on harder cases.

Now scale up. You want the model to plan a five-touch outbound sequence for a CISO at a Series B fintech. A single chain of thought produces one plan. But what if the first approach — leading with compliance — is weaker than an alternative — leading with peer proof? You would not know unless you generated both and compared. That is tree-of-thought: generate multiple reasoning branches, evaluate them, prune the weak ones, and expand the strong ones. The cost is multiple API calls per step. The benefit is that you search the space of possible outputs instead of accepting the first one.

These three techniques form an escalation ladder. Few-shot teaches the model *what* by example. Chain-of-thought teaches it *how* by forcing visible reasoning. Tree-of-thought teaches it *where* by exploring multiple paths and selecting the best. Your job is picking the right rung — because each step up costs more tokens, more latency, and more orchestration code.

## The Concept

**Few-shot mechanism.** You place K input-output pairs in the prompt before the target input. The model performs in-context learning: it adjusts its output distribution based on the examples without any weight updates. The examples function as a task specification — they show the format, the category boundaries, and the expected output shape. The decisions you make are how many shots to include (typically 2–5; more hits context limits with diminishing returns), what distribution of examples to use (balanced across categories, or weighted toward hard cases), and what to do when the model drifts off-pattern at inference time. Research on in-context learning suggests the model uses the examples to locate a task in its pretrained distribution rather than learning new behaviors from scratch [CITATION NEEDED — concept: in-context learning as latent task inference, Xie et al. 2021].

**Chain-of-thought mechanism.** You append reasoning traces to the examples, or you append the phrase "Let's think step by step" to a zero-shot prompt. The model generates intermediate tokens — "The reply mentions pricing, which maps to OBJECTION, but also requests security docs, which is a buying signal" — before producing the final answer. Those intermediate tokens are not commentary. They are computation. Each generated token becomes part of the context window for the next token's prediction, which means the model can use earlier reasoning steps to constrain later ones. The mechanism matters: a transformer performs a fixed amount of computation per token (one forward pass). Multi-step problems that require more computation than one forward pass can provide will fail without intermediate tokens to spread the work across. Chain-of-thought does not make the model smarter — it gives it more forward passes to arrive at the same answer a smarter system might produce in one step [CITATION NEEDED — concept: CoT inference-time compute scaling, Wei et al. 2022].

**Tree-of-thought mechanism.** You extend chain-of-thought by branching. At each reasoning step, you generate multiple candidate next steps (typically 3–5). You evaluate each candidate — either by prompting the model to score its own outputs or by applying an external scoring function. You prune low-scoring branches and expand high-scoring ones. This is beam search over language model reasoning. The critical implementation detail is that tree-of-thought is not a single-prompt technique. It requires an orchestrator loop: generate candidates, score, prune, repeat. Each step is a separate API call. For a depth-3 tree with 3 branches per node and a beam width of 2, that is roughly 9–18 model calls depending on implementation, compared to 1 call for few-shot or chain-of-thought.

```mermaid
flowchart TD
    A[Input task] --> B{What does the task need?}
    
    B -->|Pattern matching<br/>Classification| C[Few-Shot]
    B -->|Multi-step reasoning<br/>Single solution path| D[Chain-of-Thought]