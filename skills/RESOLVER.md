# Skill Routing — BlueSkyGTM Engineering

This file tells gbrain which skill to invoke for a given trigger phrase.
See `references/runtime-guide.md` for the full gstack skill list.

## Triggers

| Trigger | Skill | Notes |
|---------|-------|-------|
| spec / backlog / issue | /spec | Author a backlog-ready spec |
| document / docs / missing doc | /document-generate | Generate missing documentation |
| autoplan / full plan review | /autoplan | CEO+Eng+DX review gate |
| review / diff check | /review | Code review |
| ship / pr / release | /ship | Ship / create PR |
| bug / error / broken | /investigate | Debug and fix |
| design / visual / site | /design-review | Visual site changes |
| pivot / scope / direction | /office-hours | Direction and scope questions |
| deploy / monitoring | /canary | Post-deploy monitoring |
| save / checkpoint | /context-save | Save progress |
| resume / restore context | /context-restore | Resume prior context |
| security / audit | /cso | Security audit |
| health / codebase health | /health | Codebase health check |
| upgrade gstack | /gstack-upgrade | Upgrade gstack skill suite |
| setup gbrain | /setup-gbrain | Configure gbrain |
| sync gbrain | /sync-gbrain | Sync gbrain artifacts |
| taskmaster / tasks | /taskmaster | Visual task tracker |
