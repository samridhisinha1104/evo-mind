"""Structured mutation operators for strategy evolution.

Instead of asking the LLM to free-form rewrite a strategy (which often just
renames it), mutations are explicit combinatorial operators applied to the
strategy's step list and params.  The LLM is consulted to *choose* which
operator(s) to apply, but the mechanics are deterministic.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from evomind.state import StepResult, Strategy


def _pick_random(items: list, exclude: set | None = None) -> str | None:
    """Pick a random item not in *exclude*.  Returns None if nothing to pick."""
    candidates = [x for x in items if x not in (exclude or set())]
    return random.choice(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Individual operators
# ---------------------------------------------------------------------------


def add_step(strategy: Strategy, available_steps: list[str], **_kw: Any) -> Strategy:
    """Insert a random step that isn't already in the strategy."""
    s = deepcopy(strategy)
    existing = set(s["steps"])
    new_step = _pick_random(available_steps, exclude=existing)
    if new_step:
        pos = random.randint(0, len(s["steps"]))
        s["steps"].insert(pos, new_step)
    s["name"] = f"{s['name']}_add"
    return s


def remove_step(
    strategy: Strategy,
    step_results: list[StepResult] | None = None,
    **_kw: Any,
) -> Strategy:
    """Drop the least-useful step.  Falls back to random if no results supplied."""
    s = deepcopy(strategy)
    if len(s["steps"]) <= 1:
        s["name"] = f"{s['name']}_noremove"
        return s

    if step_results:
        # Remove the step with worst signal (error or empty data)
        worst = None
        for r in step_results:
            if r["step"] in s["steps"] and (r["error"] or not r["data"]):
                worst = r["step"]
                break
        if worst:
            s["steps"].remove(worst)
        else:
            s["steps"].pop(random.randint(0, len(s["steps"]) - 1))
    else:
        s["steps"].pop(random.randint(0, len(s["steps"]) - 1))

    s["name"] = f"{s['name']}_remove"
    return s


def swap_step(strategy: Strategy, available_steps: list[str], **_kw: Any) -> Strategy:
    """Replace one step with a different one from the registry."""
    s = deepcopy(strategy)
    if not s["steps"]:
        return s
    idx = random.randint(0, len(s["steps"]) - 1)
    existing = set(s["steps"])
    replacement = _pick_random(available_steps, exclude=existing)
    if replacement:
        s["steps"][idx] = replacement
    s["name"] = f"{s['name']}_swap"
    return s


def tune_params(strategy: Strategy, **_kw: Any) -> Strategy:
    """Perturb a numeric param by ±20%."""
    s = deepcopy(strategy)
    params = s.get("params", {})
    numeric_keys = [k for k, v in params.items() if isinstance(v, (int, float))]
    if numeric_keys:
        key = random.choice(numeric_keys)
        factor = random.uniform(0.8, 1.2)
        original = params[key]
        if isinstance(original, int):
            params[key] = max(1, int(original * factor))
        else:
            params[key] = round(original * factor, 4)
        s["params"] = params
    s["name"] = f"{s['name']}_tune"
    return s


def reorder_steps(strategy: Strategy, **_kw: Any) -> Strategy:
    """Swap two steps in the ordering."""
    s = deepcopy(strategy)
    if len(s["steps"]) < 2:
        return s
    i, j = random.sample(range(len(s["steps"])), 2)
    s["steps"][i], s["steps"][j] = s["steps"][j], s["steps"][i]
    s["name"] = f"{s['name']}_reorder"
    return s


def crossover(
    strategy: Strategy,
    partner: Strategy | None = None,
    **_kw: Any,
) -> Strategy:
    """Merge two strategies: take the first half of one, second half of the other."""
    s = deepcopy(strategy)
    if partner is None or not partner.get("steps"):
        s["name"] = f"{s['name']}_nocross"
        return s

    mid_a = len(s["steps"]) // 2 or 1
    mid_b = len(partner["steps"]) // 2 or 1
    combined = s["steps"][:mid_a] + partner["steps"][mid_b:]
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for step in combined:
        if step not in seen:
            seen.add(step)
            deduped.append(step)
    s["steps"] = deduped
    # Merge params (partner overrides)
    merged_params = {**s.get("params", {}), **partner.get("params", {})}
    s["params"] = merged_params
    s["name"] = f"{s['name']}_cross"
    return s


# ---------------------------------------------------------------------------
# Registry & meta-mutation
# ---------------------------------------------------------------------------

MUTATION_OPERATORS = {
    "add_step": add_step,
    "remove_step": remove_step,
    "swap_step": swap_step,
    "tune_params": tune_params,
    "reorder_steps": reorder_steps,
    "crossover": crossover,
}


def apply_mutation(
    strategy: Strategy,
    operator_name: str,
    available_steps: list[str],
    step_results: list[StepResult] | None = None,
    partner: Strategy | None = None,
) -> tuple[Strategy, str]:
    """Apply a named mutation operator.  Returns (new_strategy, operator_name_used)."""
    fn = MUTATION_OPERATORS.get(operator_name)
    if fn is None:
        raise ValueError(f"Unknown mutation operator: {operator_name!r}")
    mutated = fn(
        strategy,
        available_steps=available_steps,
        step_results=step_results,
        partner=partner,
    )
    mutated["description"] = f"Mutated via {operator_name} from {strategy['name']}"
    return mutated, operator_name


def random_mutation(
    strategy: Strategy,
    available_steps: list[str],
    step_results: list[StepResult] | None = None,
    partner: Strategy | None = None,
    n_mutations: int = 1,
) -> tuple[Strategy, list[str]]:
    """Apply *n_mutations* random operators in sequence.  Returns (result, operators_used)."""
    ops_used: list[str] = []
    current = strategy
    operators = list(MUTATION_OPERATORS.keys())
    for _ in range(n_mutations):
        op = random.choice(operators)
        current, _ = apply_mutation(
            current,
            op,
            available_steps=available_steps,
            step_results=step_results,
            partner=partner,
        )
        ops_used.append(op)
    return current, ops_used
