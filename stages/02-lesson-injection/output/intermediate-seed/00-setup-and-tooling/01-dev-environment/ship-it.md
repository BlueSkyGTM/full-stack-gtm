## Ship It

**Definition of done:** all three scripts print `PASS` on every line. No `FAIL` output. A `.env` file exists in your project root with at least one key defined. A `.gitignore` file exists and contains `.env` so secrets never enter version control.

Create the `.gitignore` now if you have not already:

```bash
echo '.env' >> .gitignore
echo '.venv/' >> .gitignore
```

**GTM Redirect — Zone 01 (Data Foundation):** This environment is the launchpad for your first SerperDev call. When you can run `test_api_key.py` and see `PASS`, you have proven the exact mechanism that SerperDev, Clay, and Apollo depend on: a secret loaded from disk, injected into a process, and transmitted as an HTTP header. The next lesson replaces `httpbin.org` with a real GTM endpoint. Do not proceed until this one is clean.