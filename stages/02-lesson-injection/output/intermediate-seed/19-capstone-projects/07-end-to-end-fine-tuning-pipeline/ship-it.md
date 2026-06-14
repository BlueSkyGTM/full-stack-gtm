## Ship It

Packaging the DPO-aligned model for production serving means choosing between two paths. vLLM gives you continuous batching, PagedAttention, and high throughput — appropriate when you are generating emails at scale (thousands per hour). Ollama gives you a simpler REST API on a single machine — appropriate for development, low-volume production, or local SDR tools.

First, export the merged DPO model in a format your server understands. For vLLM, you need merged full weights (not a separate adapter):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SFT_MERGED = "./capstone_pipeline/sft/merged"
DPO_ADAPTER = "./capstone_pipeline/dpo"
SERVE_DIR = "./capstone_pipeline/serve_merged"

base = AutoModelForCausalLM.from_pretrained(
    SFT_MERGED, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base, DPO_ADAPTER)
merged = model.merge_and_unload()
merged.save_pretrained(SERVE_DIR, safe_serialization=True)

tok = AutoTokenizer.from_pretrained(SFT_MERGED)
tok.save_pretrained(SERVE_DIR)

print(f"Merged model saved to {SERVE_DIR}")
print(f"Files:")
import os
for f in os.listdir(SERVE_DIR):
    size = os.path.getsize(os.path.join(SERVE_DIR, f)) / (1024*1024)
    print(f"  {f}: {size:.1f} MB")
```

Now serve with vLLM and validate with a batch of GTM prompts:

```bash
#!/bin/bash
# serve_and_validate.sh
# Requires: pip install vllm
# Run after the Python merge script above

set -e

MODEL_PATH="./capstone_pipeline/serve_merged"
PORT=8000

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --dtype float16 \
    --gpu-memory-utilization 0.8 &

SERVER_PID=$!

echo "Waiting for server to initialize..."
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 2
done

PROMPTS=(
    "Write a cold email to a VP of Sales at a B2B SaaS company about reducing sales cycle length."
    "Write a follow-up email after no response to a cold outreach."
    "Write a cold email to a Head of Marketing about content personalization."
    "Write a cold email to a CFO about reducing SaaS spend."
    "Write a cold email to a Head of People Ops about onboarding automation."
    "Write a value proposition email to a Director of Engineering."
    "Write a re-engagement email to a cold prospect who went dark."
    "Write a break-up email to an unresponsive prospect."
    "Write a cold email to a CTO about developer productivity."
    "Write a referral request email to a happy customer."
)

RESULTS_FILE="./capstone_pipeline/batch_results.json"
echo "[" > $RESULTS_FILE

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    START=$(python3 -c "import time; print(time.time())")

    RESPONSE=$(curl -s http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_PATH\",
            \"prompt\": \"### Prompt:\n$PROMPT\n\n### Response:\n\",
            \"max_tokens\": 200,
            \"temperature\": 0.7
        }")

    END=$(python3 -c "import time; print(time.time())")
    LATENCY=$(python3 -c "print(f'{$END - $START:.3f}')")

    TEXT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])")

    WORD_COUNT=$(echo "$TEXT" | wc -w | tr -d ' ')
    HAS_GREETING=$(echo "$TEXT" | grep -qiE "^(hi|hello|hey|dear)" && echo "true" || echo "false")
    HAS_CTA=$(echo "$TEXT" | grep -qiE "(call|meet|schedule|demo|chat|talk|connect)" && echo "true" || echo "false")

    echo "--- Prompt $((i+1))/10 ---"
    echo "Latency: ${LATENCY}s | Words: $WORD_COUNT | Greeting: $HAS_GREETING | CTA: $HAS_CTA"
    echo "Output: $(echo "$TEXT" | head -c 120)..."
    echo ""

    python3 -c "
import json
entry = {
    'prompt': '''$PROMPT''',
    'output': '''$(echo "$TEXT" | sed "s/'/\\\\'/g")''',
    'latency_seconds': float('$LATENCY'),
    'word_count': int('$WORD_COUNT'),
    'has_greeting': $HAS_GREETING,
    'has_cta': $HAS_CTA
}
print(json.dumps(entry))
" >> $RESULTS_FILE