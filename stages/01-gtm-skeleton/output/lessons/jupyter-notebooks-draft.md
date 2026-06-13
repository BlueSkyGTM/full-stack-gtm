# Jupyter Notebooks: Interactive Computing for GTM Data Work

---

## Beat 1: Hook

The edit-run-inspect loop breaks when your workflow is "write script, run script, check output, repeat." Notebooks solve a specific problem: keeping interpreter state alive between code edits so you don't re-run expensive operations (API calls, data loads) every time you change a downstream calculation. This is the same pattern used when prototyping enrichment logic before moving it into a Clay waterfall or production pipeline.

---

## Beat 2: Learn It

**Cell execution model.** A notebook is a sequence of cells — code cells and markdown cells — executed against a persistent kernel. The kernel maintains state across cells. Variables defined in cell 1 are available in cell 5. This is the core mechanism and the core danger.

**Kernel state vs. document order.** Cells can be executed in any order. The kernel tracks execution order via `In [n]` / `Out [n]` counters. Document order (top to bottom) is cosmetic. Hidden state accumulates when you re-run cells non-linearly. This is the number one source of "works on my machine" notebook bugs.

**The .ipynb format.** A notebook file is a JSON document containing cell source, outputs, and metadata. This makes notebooks version-control-hostile (outputs change, metadata shifts) but also means they're inspectable and convertible.

**When notebooks, when scripts.** Notebooks are for exploration, prototyping, and documentation of analysis. Scripts are for production. The boundary: if you're scheduling it or deploying it, it's a script.

---

## Beat 3: See It

Demonstrate the execution model by showing kernel state persistence across cells, then demonstrate the danger by executing cells out of order and observing the result.

```
import json
nb = {
 "cells": [
  {"cell_type": "code", "source": ["accounts = ['Acme', 'Globex', 'Initech']"], "execution_count": 1},
  {"cell_type": "code", "source": ["filtered = [a for a in accounts if len(a) > 4]"], "execution_count": 2},
  {"cell_type": "code", "source": ["print(filtered)"], "execution_count": 3}
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
print(json.dumps(nb, indent=2))
```

This is the actual structure of a `.ipynb` file. Every notebook is this JSON under the hood.

---

## Beat 4: Practice It

**Easy:** Create a notebook with three code cells that build on each other's state. Restart the kernel and observe what breaks.

**Medium:** Open a `.ipynb` file as raw JSON. Identify the execution counts and outputs. Manually edit the JSON to change an output, then reopen the notebook.

**Hard:** Reproduce a hidden-state bug: define a variable in cell 5, use it in cell 2 (by executing 5 first, then 2). Document the exact steps to reproduce and explain why the notebook appears to work but would fail if run top-to-bottom via "Restart & Run All."

---

## Beat 5: Use It

When prototyping a **Clay waterfall** (Zone 2: Enrichment), you need to test each enrichment step — company lookup, employee count, tech stack — against real API responses before wiring it into Clay. A notebook lets you cache the API response in cell 1, then iterate on parsing logic in cell 3 without re-calling the API. Once the parsing logic is correct, port it to Clay's formula columns.

[CITATION NEEDED — concept: Clay waterfall prototyping workflow in notebooks]

**Exercise hook:** Load a sample enrichment API response (provided JSON) into a notebook. Write parsing logic across multiple cells to extract company name, employee count, and tech stack. Verify against expected output. Describe how each cell maps to a step in a Clay enrichment column.

---

## Beat 6: Ship It

**Deliverable:** A notebook that loads a prospect dataset (CSV), performs three transformation steps across separate cells, documents each step with markdown cells, and exports the result. Include a cell at the top that demonstrates the notebook is reproducible by running "Restart & Run All." The notebook must contain a markdown cell explaining which transformations would move into a Clay waterfall and which require a script.

---

## Learning Objectives

1. **Execute** code cells in a notebook and predict kernel state after each execution.
2. **Diagnose** hidden-state bugs caused by non-linear cell execution.
3. **Inspect** a `.ipynb` file as raw JSON and identify cell structure, execution counts, and outputs.
4. **Prototyping enrichment logic** in a notebook before porting to a Clay waterfall.
5. **Determine** when a workflow belongs in a notebook vs. a script based on scheduling and deployment requirements.