## Ship It

Deploy the agent as a GitHub Action triggered by the `issue_comment` event with the substring `/fix`. The Action receives the webhook payload, extracts the issue URL, invokes the agent script, and enforces two safety rails: a three-call timeout and a circuit breaker that comments on the issue if the agent fails.

```yaml
name: Issue-to-PR Agent

on:
  issue_comment:
    types: [created]

jobs:
  run-agent:
    if: contains(github.event.comment.body, '/fix')
    runs-on: ubuntu-latest
    permissions:
      issues: write
      pull-requests: write
      contents: write
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install deps
        run: pip install openai requests

      - name: Run agent
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ISSUE_URL: ${{ github.event.issue.html_url }}
          REPO: ${{ github.repository }}
          ISSUE_NUMBER: ${{ github.event.issue.number }}
          MAX_CALLS: "3"
          DRY_RUN: "false"
        run: python agent.py

      - name: Circuit breaker on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.payload.issue.number,
              body: '⚠️ Agent failed after exhausting budget. Please review manually.'
            });
```

After merging this workflow file, verify deployment by opening a test issue in the repo, commenting `/fix`, and checking the Actions tab for the run. The Action run URL confirms the deployment is live.

The `permissions` block is the credential-scoping mechanism from the problem statement. Fine-grained permissions (`issues: write`, `pull-requests: write`, `contents: write`) grant the agent exactly what it needs — no more. The `GITHUB_TOKEN` provided by Actions is automatically scoped to the single repo, preventing cross-repo access. If you need the agent to operate across repos, replace the Actions token with a GitHub App that has explicit installation targets.

The budget ceiling (`MAX_CALLS: "3"`) is the safety rail that prevents runaway costs. Three calls covers one plan + one edit + one retry if tests fail. A more generous budget (five to seven calls) allows the agent to iterate on failing tests, but doubles or triples the API cost per issue. The circuit breaker (`if: failure()`) ensures the issue author always gets feedback — silence is worse than a failure message.