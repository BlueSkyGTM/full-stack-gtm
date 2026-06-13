# Terminal & Shell

## Learning Objectives

1. Navigate the file system using shell commands (`pwd`, `ls`, `cd`, `mkdir`)
2. Create, read, and modify files from the terminal (`cat`, `echo`, `touch`, `>>`)
3. Chain commands using pipes (`|`) and redirection (`>`, `>>`)
4. Write and execute a shell script that accepts arguments
5. Inspect and modify environment variables, including `PATH`

---

## Beat 1: Hook

Why the terminal still matters when every SaaS tool has a dashboard. CLI is the only interface that composes — GUIs do one thing at a time, shells let you chain operations, automate repetitive work, and inspect what graphical tools hide. If you're going to run enrichment scripts, pipe CSVs between tools, or debug API responses, the shell is the control plane.

---

## Beat 2: Core Concept

Three layers demystified: **terminal emulator** (the window — e.g., Terminal.app, iTerm2, Windows Terminal), **shell** (the command interpreter — bash, zsh, sh), and **the operating system** (what actually executes the commands). The shell is a REPL: read input, evaluate, print result, loop. Everything else builds on this.

### Mechanism: command parsing
The shell reads a line, splits on whitespace, treats the first token as the program name, and passes remaining tokens as arguments. Knowing this explains quoting rules, glob expansion (`*.csv`), and why spaces in filenames break things.

### Mechanism: pipes and redirection
Every process has three streams: stdin (file descriptor 0), stdout (fd 1), stderr (fd 2). The `|` operator connects stdout of one process to stdin of the next. The `>` operator redirects stdout to a file. The `2>` operator redirects stderr. This is how you build data pipelines without writing a program.

### Mechanism: environment variables
Key-value pairs inherited by every child process. `PATH` tells the shell where to find executables. `HOME` points to your user directory. You set them with `export VAR=value`. They're how you pass configuration (API keys, environment names) to scripts without hardcoding.

---

## Beat 3: Use It

**GTM Redirect:** Foundational for Zone 0 (Data Infrastructure) and Zone 1 (Prospect Data Engineering). The terminal is how you invoke `curl` against Apollo's API, pipe JSON through `jq` to extract company names, redirect output to a CSV, and chain that into a Clay import. The waterfall enrichment pattern in Clay is a multi-step data pipeline — the shell's pipe model is the mental model for how data flows between enrichment steps.

[CITATION NEEDED — concept: terminal/shell as prerequisite for Clay workflow execution and API-based enrichment pipelines]

---

## Beat 4: Build It

**Easy:** Navigate to a directory, list its contents, create a subdirectory, create a file inside it, write a line to the file, read the file back, then clean up. Observable output: the file's contents printed to terminal.

**Medium:** Write a shell script that takes a CSV filename as an argument, reads the header row, prints the column count, and prints the total row count. Observable output: "Columns: N" and "Rows: N" printed to terminal.

**Hard:** Write a shell script that accepts a company domain as an argument, uses `curl` to fetch the company's clearbit enrichment data (or a mock JSON file if no API key), pipes the response through a parser to extract the company name and employee count, and appends the result as a CSV row. Observable output: a CSV line like `"acme.com","Acme Corp",500` printed to terminal and appended to a file.

---

## Beat 5: Ship It

Take the medium exercise and make it a reusable script in `~/bin/` (or any directory on your `PATH`). Make it executable with `chmod +x`. Confirm it runs from anywhere by typing just the script name. This is how CLI tools become part of your daily workflow — the same pattern tools like `jq`, `csvkit`, and `curl` follow.

---

## Beat 6: Extend It

**Shell scripting has limits.** When you need error handling, HTTP libraries, or structured data manipulation beyond what `awk`/`sed` provide, you've hit the ceiling. That's where Python enters — next lesson. 

**Further reading:** `man bash` (the actual manual, searchable with `/`), the difference between `.bashrc` and `.bash_profile`, and how `#!/usr/bin/env bash` works (the shebang tells the OS which interpreter to use, `env` finds it on PATH so it's portable across machines).