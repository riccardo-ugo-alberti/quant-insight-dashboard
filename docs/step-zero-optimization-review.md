# Step 0 — Code Optimization Review

This document captures a **step-zero** optimization assessment for the current codebase.  
Goal: agree on the highest-impact improvements before larger refactors.

## What we should improve first

1. **Normalize optimizer API and remove compatibility debt**
   - `src/optimizer.py` currently mixes a modern tuple-based API with late-file compatibility wrappers.
   - The wrappers hide mismatches (`returns` vs `prices`, `allow_short` vs `shorting`) and make behavior harder to reason about.
   - Action: lock a single public signature and enforce it in tests/app code.

2. **Reduce repeated heavy optimization calls in frontier generation**
   - `compute_frontier` solves a constrained optimization independently for each target return.
   - This is robust, but expensive as `points` grows.
   - Action: warm-start each solve with previous optimal weights and cache reusable arrays (`mu.values`, `cov.values`) once per run.

3. **Tighten numerical robustness and diagnostics**
   - Failures from SLSQP are surfaced as raw runtime errors with little context.
   - Action: include mode/constraints summary in error messages and validate `max_weight`, feasible return range, and covariance PSD assumptions before solve.

4. **Clarify data contract (prices vs returns)**
   - Optimizer functions generally assume prices then call `.pct_change()`, while tests pass synthetic returns.
   - Action: split entry points into explicit `optimize_from_prices` and `optimize_from_returns`, avoiding ambiguous inference.

5. **Add performance regression checks**
   - Current test coverage focuses on correctness only.
   - Action: add lightweight benchmark-style test gates (e.g., frontier 25 points under threshold on fixed synthetic data) to catch accidental slowdowns.

## Success criteria for Step 0

- One canonical optimizer API used consistently by app and tests.
- Frontier runtime reduced measurably (target: 20–40% faster on same synthetic dataset).
- Better failure messages for infeasible or unstable inputs.
- Clear separation between price-based and return-based workflows.

## Progress update

- [x] Step 3 implemented: added input validation and richer optimizer/frontier diagnostics.

## Proposed immediate next tasks

- [x] Refactor optimizer signatures and remove compatibility wrappers.
- [x] Update tests to match canonical API and expected return types.
- [x] Implement warm-start frontier loop + micro-benchmark test.
- [ ] Update README developer notes with optimization conventions.
