# Lesson Outline: Graph Theory for Machine Learning

---

## Hook It

Tables assume rows are independent. Accounts aren't. When your ICP scoring ignores that Company A shares a tech stack with Company B, or that Contact X reports to Contact Y, you're leaving signal on the table. Graph theory is the mathematics of relationships — and most B2B data is relational.

---

## Ground It

Core mechanisms in sequence:

1. **Graph primitives**: nodes, edges, directed vs. undirected, weighted vs. unweighted. Why a social graph (undirected) differs from an org chart (directed, acyclic).
2. **Adjacency representations**: matrix, list, edge list. Why adjacency matrices fail at scale and sparse representations win.
3. **Traversal algorithms**: BFS and DFS as the "for loops" of graph computation. When each is appropriate.
4. **Centrality measures**: degree, betweenness, eigenvector. What each captures about node importance.
5. **The message-passing abstraction**: how GNNs aggregate neighborhood information. This is the mechanism behind node classification, link prediction, and graph classification.

---

## Show It

Working code demonstrations (all terminal-runnable, all print observable output):

1. **Build and query a graph** using NetworkX. Print adjacency, degree, shortest path.
2. **Compute centrality** on a small B2B org chart. Print ranked output.
3. **Implement manual message passing** on a toy graph (no framework — just numpy operations on an adjacency matrix). Print before/after node representations. This makes the GNN mechanism legible before touching PyG or DGL.

Exercise hooks:
- *Easy*: Modify edge weights and observe centrality rank changes.
- *Medium*: Add nodes/edges and recompute shortest paths.
- *Hard*: Implement weighted message passing with a learnable aggregation function.

---

## Use It

**GTM Redirect → Zone 1 (ICP & Targeting), Zone 2 (Enrichment)**

Graph methods apply directly to GTM problems that tabular methods mis-handle:

1. **Account similarity graphs**: Companies sharing technologies, investors, or hiring patterns form a weighted graph. Node similarity on this graph surfaces lookalike accounts that firmographic filters miss. [CITATION NEEDED — concept: graph-based lookalike modeling in account scoring]
2. **Org chart traversal**: Multi-threaded outbound requires mapping reporting lines. This is a directed graph traversal — BFS from a VP to find directs, skip-level directs, and peer VPs.
3. **Technology adjacency**: Bipartite graph (companies × technologies). Edge weight = adoption strength. Recommend accounts based on tech stack overlap with your best customers.
4. **Enrichment as graph propagation**: A waterfall enrichment sequence is a path through data provider nodes. Each provider is a node; the edges are "fallback" relationships. Modeling it as a graph lets you optimize the traversal order.

Exercise hooks:
- *Easy*: Build a company-technology bipartite graph and find companies sharing ≥3 technologies.
- *Medium*: Compute eigenvector centrality on a prospect relationship graph to identify the most "connected" accounts.
- *Hard*: Implement label propagation to classify prospects as likely/unlikely ICP based on graph neighbors.

---

## Ship It

Build a runnable artifact:

**Account Scoring via Graph Features**

Take a list of companies with known tech stacks. Construct a company similarity graph. Compute graph-based features (centrality, community membership, neighbor ICP ratio). Score accounts using these features alongside traditional firmographics. Print ranked output.

Constraints:
- Input: CSV or hardcoded company data
- Output: Ranked account list with graph-derived scores
- Must run in terminal with `python script.py`

Exercise hooks:
- *Medium*: Ship the basic scorer above.
- *Hard*: Add community detection (Louvain or label propagation) and use community ID as a categorical feature. Compare scoring with and without graph features.

---

## Prove It

Assessment targets (must align to learning objectives in `docs/en.md`):

1. **Given an adjacency matrix, compute node degree** — tests representation literacy.
2. **Given a directed graph, determine if a path exists between two nodes** — tests traversal comprehension.
3. **Compare BFS vs. DFS for org chart traversal** — tests algorithm selection, not memorization.
4. **Explain why eigenvector centrality differs from degree centrality** in a GTM context (e.g., an account with few connections but those connections are well-connected).
5. **Describe the message-passing mechanism** in one paragraph without naming a framework.

Objectives (3–5, action verbs, testable):
1. Represent relational data as a graph using adjacency matrices and edge lists.
2. Implement BFS and DFS traversal on a directed graph.
3. Compute and interpret degree, betweenness, and eigenvector centrality.
4. Implement single-step message passing on a toy graph using matrix operations.
5. Construct a company similarity graph and extract graph-based features for account scoring.

---

*Note: Full exercise text and `docs/en.md` will be generated in subsequent passes. This outline establishes scope and sequence only.*