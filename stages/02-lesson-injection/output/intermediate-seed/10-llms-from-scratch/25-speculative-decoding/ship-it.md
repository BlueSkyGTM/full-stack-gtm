## Ship It

To deploy speculative decoding in production, you need three things: a target model worth accelerating (typically 7B+ parameters, where memory bandwidth dominates), a correlated draft model or head, and an inference server that supports the draft-verify loop natively.

**vLLM** implements speculative decoding with several draft strategies. For the classic two-model approach, you provide a draft model checkpoint. For EAGLE specifically, you provide a trained EAGLE head — a small autoregressive network that operates on the target model's hidden states. The configuration is a single parameter in the server's launch command:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
)

sampling = SamplingParams(temperature=0.0, max_tokens=100)
output = llm.generate(["Write a one-sentence value proposition for a B2B SaaS CRM."], sampling)
print(output[0].outputs[0].text)
```

The `[ngram]` draft model uses n-gram lookup from the prompt context — no trained draft model required. It works best for tasks with repetitive structure (code generation, structured data extraction). For higher acceptance rates on open-ended generation, you'd swap `[ngram]` for an EAGLE checkpoint trained on the target model's hidden states.

The GTM production concern is cost-per-prospect, not raw tokens-per-second. When you're running multi-agent orchestration at scale — say, a pipeline that generates personalized outreach for 50,000 accounts — the inference cost dominates. Speculative decoding with a 4× speedup means 4× the throughput on the same GPU, which means 4× the prospects processed per dollar. For Zone 10 agent squads where a draft router proposes and a verifier confirms, the same math applies: every rejected proposal is wasted draft compute, and every accepted proposal is a verification you didn't have to run separately.

The deployment risk is tail latency. Speculative decoding adds variance: when the draft is accepted, you process tokens quickly; when it's rejected, you fall back to single-token mode. In a synchronous GTM pipeline where a human is waiting for a generated email, this variance is usually invisible (a 200ms vs 800ms difference in a 30-second interaction). In a batch pipeline processing millions of records, it averages out. The risk case is real-time agent orchestration where a verifier timeout cascades into a failed workflow — monitor acceptance rates and fall back to non-speculative decoding if they drop below ~1.5 tokens per pass.