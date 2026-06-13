# Python Environments

## Hook It
You install a package for one project and break another. This beat frames dependency isolation as the core problem: why one global Python install is a liability once you're running multiple GTM scripts against different APIs.

## Ground It
The mechanism is `sys.path` resolution and `site-packages` isolation. Explain how Python locates modules, what a virtual environment actually does (modifies `sys.prefix` and adds a local `site-packages`), then name `venv`, `virtualenv`, and `conda` as tools that implement this pattern. Cover `pyenv` as a version manager, not an environment manager.

## Show It
Create a virtual environment, install a package into it, demonstrate isolation by showing the package is unavailable outside the environment. Show `pip freeze > requirements.txt` for reproducibility. Show a `.env` file loaded with `python-dotenv` and `os.getenv`. Every code block prints observable output confirming the mechanism worked.

## Try It
- **Easy:** Create a venv, install `requests`, print `requests.__version__`, deactivate, attempt `import requests` in the global interpreter, observe the `ModuleNotFoundError`.
- **Medium:** Build two environments with conflicting versions of the same package. Print version from each to confirm isolation.
- **Hard:** Write a script that reads an API key from `.env`, makes a live HTTP call, and prints the response status. Package it with a `requirements.txt` that a reviewer can install in one command.

## Use It
GTM Cluster 01 — TAM Mapping / Signal Machine + Score & Qualify. This Python environment is where you'll run Clay webhook receivers, Apollo API calls, and enrichment scripts. Every integration in the Signal Machine pipeline starts with a reproducible `requirements.txt` and a clean venv. If you can't recreate your environment, you can't hand off your enrichment script to a teammate.

## Ship It
Checkpoint: the practitioner delivers a directory containing a `requirements.txt`, a `.env.example` (keys redacted), and a script that runs end-to-end after `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. The GTM redirect: this is the exact scaffold for any Clay webhook or Apollo enrichment function you'll deploy in later lessons.