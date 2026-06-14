## Ship It

Deploy LiteLLM as a Docker container with a production configuration. The dev config above used SQLite and a single worker. Production needs Postgres for the spend log (SQLite corrupts under concurrent writes), multiple workers for throughput, and environment-injected secrets.

Create `docker-compose.yml`:

```yaml
version: "3.9"

services:
  litellm-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: litellm
      POSTGRES_USER: litellm
      POSTGRES_PASSWORD: ${LITELLM_DB_PASSWORD}
    volumes:
      - litellm_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U litellm"]
      interval: 5s
      timeout: 5s
      retries: 5

  litellm-proxy:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      LITELLM_DB_PASSWORD: ${LITELLM_DB_PASSWORD}
    volumes:
      - ./litellm_prod.yaml:/app/config.yaml
    command: --config /app/config.yaml --port 4000 --num_workers 8
    depends_on:
      litellm-db:
        condition: service_healthy

volumes:
  litellm_pgdata:
```

Create `litellm_prod.yaml`:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
      rpm: 500
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      rpm: 100
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
      rpm: 400

router_settings:
  fallbacks:
    - gpt-4o-mini:
      - claude-sonnet
    - gpt-4o:
      - claude-sonnet
  routing_strategy: usage-based-routing-v2

litellm_settings:
  success_callback: ["lite_llm_logger"]
  failure_callback: ["lite_llm_logger"]
  cache: true
  cache_params:
    type: redis
    host: redis
    port: 6379
  budget_settings:
    max_budget: 500.0
    budget_duration: "1d"

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: postgresql://litellm:$(LITELLM_DB_PASSWORD)@litellm-db:5432/litellm
```

Deploy:

```bash
export LITELLM_DB_PASSWORD="$(openssl rand -hex 16)"
export LITELLM_MASTER_KEY="sk-litellm-$(openssl rand -hex 16)"
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"

docker compose up -d

sleep 20

curl -s http://localhost:4000/health/readiness | python3 -m json.tool
```

The `/health/readiness` endpoint confirms the proxy started, connected to Postgres, and loaded all model configs. Verify the spend log is writing to Postgres:

```bash
docker exec litellm-proxy bash -c "pip install psycopg2-binary -q && python3 -c \"
import psycopg2, os
conn = psycopg2.connect(host='litellm-db', dbname='litellm', user='litellm', password=os.environ['LITELLM_DB_PASSWORD'])
cur = conn.cursor()
cur.execute('SELECT count(*) FROM lite_llm_spend_logs;')
print(f'Spend log rows: {cur.fetchone()[0]}')
cur.execute('SELECT model, round(sum(response_cost), 4) FROM lite_llm_spend_logs GROUP BY model ORDER BY model;')
for row in cur.fetchall():
    print(f'  {row[0]}: \${row[1]}')
conn.close()
\""
```

In a GTM context, this deployment is the infrastructure under your Clay enrichment waterfall — the same pattern from Zone 17. You version the config file alongside your Clay table definitions. When you change the prompt template or swap the primary model, you commit the config change. The spend log in Postgres gives you per-model cost attribution, so you can detect when a prompt regression causes spend to spike — the commercial equivalent of detecting scoring model drift.