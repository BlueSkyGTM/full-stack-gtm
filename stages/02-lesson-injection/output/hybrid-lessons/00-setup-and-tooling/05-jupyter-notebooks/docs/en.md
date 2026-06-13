# Jupyter Notebooks: Interactive Computing for GTM Data Work

## Learning Objectives

- Launch a Jupyter server and execute cells in both document order and non-linear order
- Trace kernel state across cells to predict which variables are available at each execution point
- Reproduce a hidden-state bug by running cells out of order, then diagnose it via "Restart & Run All"
- Extract cell logic from a `.ipynb` file and assemble a standalone Python script
- Benchmark cell execution time to estimate cost at GTM data scale

## The Problem

The standard Python development loop is: write a script, run the script, inspect the output, edit the script, run the whole thing again. This works fine for small scripts. It falls apart when your script makes an API call to pull 5,000 company records from your CRM, and then you want to test a filtering function that operates on those records. Every edit to the filter means re-running the API call. Twenty iterations of the filter means twenty API calls, twenty rate-limit waits, and twenty minutes of dead time.

Notebooks solve this specific problem by keeping a Python interpreter alive between code edits. The 5,000 records stay in memory while you iterate on the filter twenty times. The expensive operation runs once. The cheap operations run twenty times. This is the same pattern you use when prototyping enrichment logic before moving it into a Clay waterfall — you pull the data once, test your transformations against it, then ship the working logic elsewhere.

But notebooks introduce their own failure mode. Because the interpreter is stateful and cells can run in any order, the notebook on your screen may not represent what the kernel actually holds in memory. Someone else — or your future self, after a kernel restart — runs the same notebook top-to-bottom and gets a `NameError` on cell 2. This is the "works on my machine" problem, ported to interactive computing.

## The Concept

A notebook is a JSON document containing an ordered list of cells. Each cell is either code (executable Python) or markdown (formatted text). A kernel — a persistent Python process — sits behind the notebook UI and executes code cells on demand. When you run a cell, the notebook sends that cell's source to the kernel, the kernel executes it against its in-memory namespace, and any output gets displayed beneath the cell.

The kernel maintains that namespace across cells. A variable defined in cell 1 is available in cell 5, cell 10, cell 50 — until you restart the kernel or delete the variable. This persistence is the entire reason notebooks exist. It is why you can load data once and iterate on transformations without reloading.

```mermaid
flowchart TD
    A["Cell 1: companies = load_csv('accounts.csv')"] -->|"companies in kernel memory"| B["Cell 2: filtered = filter by industry"]
    B -->|"filtered in kernel memory"| C["Cell 3: enriched = call provider API"]
    C -->|"enriched in kernel memory"| D["Cell 4: export to CSV"]
    A -.->|"still available"| C
    B -.->|"still available"| D
    E["Kernel (Python process)<br/>holds all variables until restart"] -.-> A
    E -.-> B
    E -.-> C
    E -.-> D
```

The danger lives in the execution model: cells can run in any order. The notebook UI shows `In [n]` and `Out [n]` counters that track the order the kernel actually received commands. The visual order of cells in the document — top to bottom — is cosmetic. If you run cell 5, then scroll up and run cell 2, the kernel executes them in that sequence: 5, then 2. State from cell 5 is available to cell 2, even though cell 2 sits above cell 5 in the document.

This creates hidden state. You define a variable in cell 5, use it in cell 2, and everything works because the kernel has both in memory. But someone opening your notebook and clicking "Restart & Run All" gets a `NameError` because cell 2 runs before cell 5 defines the variable. This is the number one source of notebook bugs, and it is entirely a human-behavior problem: the tool lets you execute things out of order, and you do.

The `.ipynb` file is a JSON document. It stores cell sources, cell outputs, execution counts, and kernel metadata. Outputs are baked into the file itself, which means every time you run a notebook and save it, the file changes — even if your code didn't. A git diff on a notebook is dominated by output changes and metadata shifts, not source edits. Tools like `nbstripout` strip outputs before commit, but the underlying format is the root cause. When reviewing a PR that includes notebook changes, you are reading JSON diffs of output text, not clean code changes.

The boundary between notebooks and scripts is straightforward: if you are scheduling it, deploying it, or running it unattended, it is a script. Notebooks are for exploration, prototyping, and documenting an analysis with inline results. The workflow: prototype in a notebook, verify the logic, then extract the working cells into a `.py` file and ship that.

## Build It

Let's construct a notebook from scratch — not through a UI, but by writing the underlying JSON. This strips away the magic and shows you exactly what a `.ipynb` file contains.

```python
import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Company Enrichment Prototype\n",
                "Testing filter logic before moving to Clay."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["['Acme Corp', 'Globex', 'Initech', 'Umbrella']\n"]
                }
            ],
            "source": [
                "companies = ['Acme Corp', 'Globex', 'Initech', 'Umbrella']\n",
                "print(companies)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["['Acme Corp', 'Umbrella']\n"]
                }
            ],
            "source": [
                "long_names = [c for c in companies if len(c) > 7]\n",
                "print(long_names)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("enrichment_prototype.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Wrote enrichment_prototype.ipynb")
print(f"Total cells: {len(notebook['cells'])}")
print(f"Code cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')}")
print(f"Markdown cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')}")
print()
print("First 200 chars of the raw file:")
with open("enrichment_prototype.ipynb") as f:
    print(f.read()[:200])
```

That JSON structure — the `cells` array, the `metadata` block, the `nbformat` version — is exactly what lives inside every `.ipynb` file on disk. The `execution_count` field is the `In [n]` counter. The `outputs` array stores what the cell printed when it last ran. When you open this file in Jupyter, it renders the cells visually. When you view it as raw text, you see this JSON.

Now let's simulate the kernel itself to make the state-persistence mechanism concrete:

```python
class FakeKernel:
    def __init__(self):
        self.globals_dict = {"__builtins__": __builtins__}
        self.exec_count = 0

    def run_cell(self, source):
        self.exec_count += 1
        print(f"In [{self.exec_count}]: {source.strip()}")
        exec(source, self.globals_dict)
        visible = {k: type(v).__name__
                   for k, v in self.globals_dict.items()
                   if not k.startswith('_')}
        if visible:
            print(f"  >> kernel memory: {visible}")
        print()

kernel = FakeKernel