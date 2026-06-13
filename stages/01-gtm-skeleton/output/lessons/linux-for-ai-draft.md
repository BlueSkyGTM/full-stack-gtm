# Linux for AI

## Hook

You will spend most of your AI engineering career inside a terminal. Every GPU you rent, every inference server you deploy, every log you debug — Linux. This lesson covers the command-line mechanics that sit underneath every AI tool you'll use.

## Concept

Four primitives that compose into everything else: the filesystem hierarchy, process lifecycle, environment variables, and I/O redirection. Each solved a specific Unix design problem; each still dictates how AI tooling behaves today. Mechanism first, commands second.

## Demonstration

Working terminal sessions showing: navigating to a model directory, checking if a Python process is running, exporting a variable and confirming a script reads it, piping inference output through grep and tee. Every example prints observable confirmation.

## Use It

GTM cluster redirect: **Zone 01 — Infrastructure & Compute**. Specifically: SSH into a cloud GPU instance, verify the runtime environment, and confirm a model serving endpoint responds with `curl`. This is the same workflow used to host inference APIs that feed GTM enrichment pipelines.

## Ship It

Write a shell script that: activates a virtual environment, runs a Python inference script, captures stdout and stderr to separate log files, and exits with a nonzero status code if inference fails. This is the pattern used in production cron jobs and CI pipelines.

## Review

Checklist of the four primitives, how they compose, and which GTM infra task each maps to. One connection forward: the next lesson covers containerization, which is just these same primitives packaged into a frozen filesystem.

---

**Exercise Hooks:**
- *Easy:* Navigate to a directory, list files filtered by extension, print the count
- *Medium:* Write a script that checks if a process is running and restarts it if not, logging the event
- *Hard:* Build a wrapper script that accepts a Python script path, runs it inside a virtual environment, captures exit code, timestamps output, and sends a summary to a log file — mimicking a production inference runner

---

**Learning Objectives:**
1. Navigate the Linux filesystem and manipulate files from the terminal
2. Configure environment variables and virtual environments for Python-based AI tooling
3. Execute and manage long-running processes (inference servers, training jobs)
4. Write shell scripts that chain commands with pipes, redirection, and exit-code handling
5. Diagnose process and network issues using `ps`, `top`, `ss`, and `curl`

**GTM Redirect:** Zone 01 — Infrastructure & Compute. Linux proficiency is the prerequisite for deploying any GTM-AI tooling (Clay agents, inference endpoints, enrichment webhooks). No forced application beyond this — it is foundational.