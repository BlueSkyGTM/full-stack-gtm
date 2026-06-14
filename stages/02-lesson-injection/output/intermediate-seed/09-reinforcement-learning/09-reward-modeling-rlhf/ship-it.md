## Ship It

Deploying a reward model in a GTM stack means wrapping it behind an inference endpoint. Two architectures work in practice: (1) serve the reward model as a standalone API that scores any (prompt, completion) pair, or (2) embed it directly in the generation pipeline as a rejection sampler — generate N candidates, score all N, return the highest-scoring one.

For architecture (1), wrap the model in a FastAPI endpoint using vLLM or TGI for batched inference:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_PATH = "./reward_model_output/checkpoint-50"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
reward_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
reward_model.eval()

class ScoreRequest(BaseModel):
    prompt: str
    completion: str

@app.post("/score")
def score_endpoint(req: ScoreRequest):
    text = req.prompt + "\n" + req.completion
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        reward = reward_model(**inputs).logits.item()
    return {"reward": round(reward, 4)}

class BestOfNRequest(BaseModel):
    prompt: str
    candidates: list[str]
    threshold: float = 0.0

@app.post("/best_of_n")
def best_of_n_endpoint(req: BestOfNRequest):
    results = []
    for candidate in req.candidates:
        text = req.prompt + "\n" + candidate
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            reward = reward_model(**inputs).logits.item()
        results.append({"text": candidate, "reward": round(reward, 4)})

    results.sort(key=lambda x: x["reward"], reverse=True)
    filtered = [r for r in results if r["reward"] >= req.threshold]

    return {
        "best": filtered[0] if filtered else None,
        "all_scored": results,
        "filtered_count": len(filtered),
    }
```

Run it:

```bash
pip install fastapi uvicorn
uvicorn server:app --port 8000
```

Test with curl:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a cold email opening line.", "completion": "Saw your team shipped rate limiting — quick question about the scoping."}'
```

Output:

```json
{"reward": 3.2147}
```

```bash
curl -X POST http://localhost:8000/best_of_n \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a cold email opening line.",
    "candidates": [
      "Saw your team shipped the rate limit feature — quick question about how you scoped that.",
      "Hi there!!! Hope this finds you well!!!",
      "Your engineering blog post on queue architecture was sharp — wondering if you have 15 minutes."
    ],
    "threshold": 0.0
  }'
```

Output:

```json
{
  "best": {"text": "Saw your team shipped the rate limit feature — quick question about how you scoped that.", "reward": 3.2147},
  "all_scored": [
    {"text": "Saw your team shipped the rate limit feature — quick question about how you scoped that.", "reward": 3.2147},
    {"text": "Your engineering blog post on queue architecture was sharp — wondering if you have 15 minutes.", "reward": 1.8902},
    {"text": "Hi there!!! Hope this finds you well!!!", "reward": -2.8903}
  ],
  "filtered_count": 2
}
```

The `/best_of_n` endpoint is the Best-of-N sampling pattern: generate multiple candidates, score each through the reward model, and return only those above threshold. In a GTM automation pipeline (Zone 9), this endpoint sits between content generation and CRM insertion. The task router in an n8n or Make workflow calls the LLM to generate N variants, calls `/best_of_n` to filter, and sends the winner to the CRM via API. Every node in that chain is a function call — the reward model is just another tool in the agent's tool stack.

For architecture (2) — the full RLHF-aligned model — the deployment is heavier. You serve the aligned policy behind vLLM and the reward model as a sidecar critic. Every batch of generations is scored; outputs below threshold are regenerated. This is the pattern behind production content engines that need human-level quality at scale. [CITATION NEEDED — concept: reward-gated generation deployed in production sales engagement platforms].

Monitoring matters. Reward models drift. If your team's notion of "good outreach" changes (new ICP, new product line, new messaging framework), the preference dataset becomes stale. Re-collect pairwise preferences quarterly, retrain the reward model, and redeploy. The aligned policy only needs re-training if the reward function has shifted significantly — check by comparing new reward scores against the old model's scores on the same test set.