# Dev Environment

## Hook

You're a practitioner who needs to ship, not a student setting up a playground. Every minute fighting tooling is a minute not spent on pipeline logic. This beat establishes the minimum viable environment: shell, language runtime, API key management, and HTTP client. Nothing more.

## Concept

Three mechanisms, in order of dependency:

1. **Path resolution** — how your shell locates executables via `$PATH`, why `which python3` matters, and what happens when multiple runtimes collide.
2. **Environment variable isolation** — the process-level scope of `export`, why `.env` files exist (dotenv pattern: key=value pairs loaded into process memory, never committed to VCS), and the difference between a variable visible to one terminal session vs. inherited by child processes.
3. **HTTP as the control plane** — every AI API and GTM tool (Clay, Apollo, Smartlead) speaks REST. The `curl` → `requests`/`fetch` progression is the only transport layer you need.

Tool introduction order follows the mechanism: `curl` proves the API works, `python3 -c` proves the runtime works, `.env` proves secrets stay local.

## Demo

Three working scripts, each self-contained with observable output:

1. **`verify_env.sh`** — prints `$SHELL`, `python3 --version`, `which curl`, and confirms a `.env` file exists or tells you it doesn't. Outputs a single pass/fail line per dependency.
2. **`test_api_key.py`** — loads `.env` via `python-dotenv`, sends one `GET` to `https://httpbin.org/headers` with an `Authorization: Bearer $OPENAI_API_KEY` header (masked in output), prints status code and masked key. Uses `requests`. No API credits consumed — httpbin mirrors headers back.
3. **`http_check.sh`** — `curl -s -o /dev/null -w "%{http_code}" https://httpbin.org/get` — confirms outbound HTTPS works. Prints the status code.

All three run without modification. No browser required.

## Use It

**GTM Redirect — Zone 01 (Data Foundation) / Zone 02 (Enrichment)**

The dev environment is the prerequisite for every GTM pipeline. Specifically:

- **Clay's API** requires a Bearer token in an `Authorization` header. You'll pass it from `.env` → `requests.Session.headers`. This is the same pattern as `test_api_key.py`, just pointed at `https://api.clay.com/v3/contacts` instead of httpbin.
- **Waterfall enrichment** in Clay calls multiple data providers sequentially. You'll reproduce that waterfall in Python in a later lesson. Right now, you prove you can make *one* authenticated call.
- **Apollo, Hunter, Dropcontact** — same pattern, different base URL and auth header.

If the three demo scripts pass, you're ready for the enrichment waterfall. If any fail, stop here and fix before moving on.

## Ship It

Three exercises, escalating:

- **Easy**: Run `verify_env.sh`. Fix any failures. Paste the output (with masked API keys) into your notes. Confirm all lines say `PASS`.
- **Medium**: Create a `.env` file with a dummy `TEST_KEY=abc123`. Write a one-liner Python script that loads it with `dotenv`, prints the key, and exits with code 0. Run it; confirm output is `abc123`.
- **Hard**: Write `enrichment_ping.py` that takes a provider name as a CLI argument (`python enrichment_ping.py clay`), loads the correct API key from `.env` (pattern: `CLAY_API_KEY`, `APOLLO_API_KEY`), hits the provider's health endpoint or any `GET` endpoint, and prints `{provider: STATUS_CODE, latency_ms: X}`. Handle the case where the key is missing — print `{provider: MISSING_KEY, latency_ms: 0}` and exit 1.

## Pitfalls

- **`python` vs `python3`** — macOS ships a stub `python` that opens the App Store. Always use `python3`. The demo scripts hardcode `python3`.
- **`.env` in the wrong directory** — `python-dotenv` looks in the current working directory. If you run the script from `~/` but `.env` lives in `~/projects/gtm-pipe/`, it won't load. The fix is explicit: `load_dotenv("path/to/.env")`.
- **API key in shell history** — `export OPENAI_API_KEY=sk-...` is now in your `~/.bash_history` or `~/.zsh_history` in plaintext. Use `.env` files instead. Add `.env` to `.gitignore` immediately — before your first commit.
- **Multiple Python versions** — `which -a python3` may show `/usr/bin/python3` (system) and `/opt/homebrew/bin/python3` (Homebrew). If `pip3` installs a package into one but you run the other, `import` fails. Verify with `python3 -c "import requests; print(requests.__file__)"` to see which site-packages you're hitting.
- **Corporate proxy / VPN** — if `curl https://httpbin.org/get` returns a connection error, you're behind a proxy. Set `HTTPS_PROXY` in your environment. This is not a code problem; it's a network problem. Fix it before proceeding.

---

**Learning Objectives (testable, action-verb):**

1. Configure a local shell environment with `python3`, `curl`, and `dotenv`, and verify each component produces expected output.
2. Execute an authenticated HTTP request to a REST API using credentials loaded from a `.env` file.
3. Diagnose and resolve `PATH` conflicts between multiple Python runtimes.
4. Implement a CLI script that accepts a provider name, loads the corresponding API key, and reports connection status with latency.
5. Prevent credential exposure by excluding `.env` from version control and avoiding `export` in shell history.