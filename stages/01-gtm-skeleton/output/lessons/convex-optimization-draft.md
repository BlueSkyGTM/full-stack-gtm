# Convex Optimization

## Hook
You have 12 outbound reps, 4 territories, and a quota. You need to assign reps to territories while minimizing travel cost and balancing pipeline. That is a constrained optimization problem — and if the constraints are convex, you can solve it exactly, not heuristically.

## Concept
Define convex sets, convex functions, and the critical property: every local minimum is a global minimum. Show the geometric intuition (bowl vs. saddle) and explain why this one property makes a problem tractable vs. NP-hard.

## Mechanism
Walk through the solver pipeline: formulate the objective, express constraints, check convexity via composition rules, then hand to a solver. Cover linear programs (LP), quadratic programs (QP), and second-order cone programs (SOCP) — each is a strict subset of the one after it. Explain interior-point methods at the algorithm level before naming CVXPY or SciPy as implementations.

## Code
Implement a working constrained optimization: allocate a fixed budget across N outbound channels to maximize expected pipeline, subject to per-channel minimums and a total spend constraint. Use `cvxpy` with `print` statements confirming feasibility, objective value, and the optimal allocation vector.

## Use It
This is the engine behind GTM budget allocation and territory rebalancing. [CITATION NEEDED — concept: GTM budget optimization via convex programming]. The redirect is **Zone 2 — Outbound & Sequences**: channel spend allocation and rep-territory assignment are convex programs when formulated correctly (linear costs, linear or quadratic objectives).

## Ship It
Exercise hooks:
- **Easy**: Re-run the code example with modified constraints (lower one channel's minimum) and confirm the solver finds a new optimum.
- **Medium**: Formulate a territory-balancing LP — equal pipeline targets, travel-cost objective — and solve it. Print the assignment matrix.
- **Hard**: Add a quadratic risk term to the budget allocation objective (penalize variance in expected returns). Confirm the problem remains convex. Compare objective values: linear vs. quadratic.