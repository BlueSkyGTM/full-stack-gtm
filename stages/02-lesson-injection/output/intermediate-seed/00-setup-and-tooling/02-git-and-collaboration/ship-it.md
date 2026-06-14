## Ship It

A shared GTM operations repository needs three things to be safe for team collaboration: branch protection on `main` (nobody pushes directly — all changes go through pull requests), a PR template that forces the author to state the hypothesis behind a config change, and a `README` that documents the folder structure so a new revops hire knows where enrichment configs live versus scoring models.

The remote setup (GitHub branch protection rules, PR templates) happens in the GitHub UI or API. But the local scaffolding — the files that make the repo self-documenting — is terminal-only:

```bash
#!/bin/bash

WORKDIR=$(mktemp -d /tmp/gtm-ops-repo.XXXXXX)
cd "$WORKDIR"

git init -q
git config user.name "GTM Engineer"
git config user.email "gtm@example.com"

mkdir -p configs/clay configs/scoring configs/outbound templates docs

cat > README.md << 'EOF'
# GTM Operations Repository