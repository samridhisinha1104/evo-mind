"""Builds the LangGraph StateGraph that runs the evolve-and-retain loop.

    planner -> executor -> evaluator -> reflector -+-> planner   (if should_continue)
                                                    +-> END       (otherwise)

Node functions in evomind.nodes take extra arguments (llm, memory, df) beyond
`state`, so build_graph() binds those via closures before registering them
with LangGraph — LangGraph itself only ever calls `node(state)`.
"""

from __future__ import annotations

import pandas as pd
from langgraph.graph import END, StateGraph

from evomind.llm import LLMClient
from evomind.loader import load_dataset, validate_dataset
from evomind.memory import StrategyMemory
from evomind.nodes import evaluator_node, executor_node, planner_node, reflector_node
from evomind.state import EvoState
from evomind.telemetry import RunTracker, log_event


def _route_after_reflection(state: EvoState) -> str:
    return "planner" if state.get("should_continue") else END


def build_graph(
    df: pd.DataFrame,
    llm: LLMClient | None = None,
    memory: StrategyMemory | None = None,
):
    """Compile the EvoMind graph, bound to one dataset/LLM client/memory store."""
    memory = memory or StrategyMemory()

    graph = StateGraph(EvoState)

    graph.add_node("planner", lambda state: planner_node(state, llm=llm, memory=memory))
    graph.add_node("executor", lambda state: executor_node(state, df=df))
    graph.add_node("evaluator", lambda state: evaluator_node(state, llm=llm))
    graph.add_node("reflector", lambda state: reflector_node(state, memory=memory))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "evaluator")
    graph.add_edge("evaluator", "reflector")
    graph.add_conditional_edges("reflector", _route_after_reflection, {"planner": "planner", END: END})

    return graph.compile()


def run_evomind(
    dataset_path: str,
    task_description: str,
    max_iterations: int = 5,
    score_threshold: float = 0.8,
    llm: LLMClient | None = None,
    memory: StrategyMemory | None = None,
    tracker: RunTracker | None = None,
    max_dataset_mb: int = 500,
) -> EvoState:
    """Convenience entrypoint: load the dataset, build the graph, run it to completion."""
    from evomind.nodes import make_task_signature, summarize_dataset

    tracker = tracker or RunTracker()
    tracker.record_event("run_start", task=task_description, dataset=dataset_path)

    # Multi-format data loading
    df = load_dataset(dataset_path)
    validate_dataset(df, max_mb=max_dataset_mb)
    log_event("dataset_loaded", rows=len(df), cols=len(df.columns), path=dataset_path)

    dataset_summary = summarize_dataset(df)
    task_signature = make_task_signature(task_description, dataset_summary)

    memory = memory or StrategyMemory()

    # Save task embedding for similar task recall
    try:
        from evomind.llm import get_task_embedding
        emb = get_task_embedding(task_description)
        memory.save_task_embedding(task_signature, task_description, emb)
    except Exception as e:
        print(f"[Warning] Failed to save task embedding in CLI: {e}")

    compiled = build_graph(df, llm=llm, memory=memory)

    initial_state: EvoState = {
        "task_description": task_description,
        "dataset_path": dataset_path,
        "dataset_summary": dataset_summary,
        "task_signature": task_signature,
        "history": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "score_threshold": score_threshold,
        "best_score": -1.0,
        "should_continue": True,
    }

    # recursion_limit: 4 graph-steps per loop iteration, plus headroom
    final_state = compiled.invoke(initial_state, config={"recursion_limit": max_iterations * 4 + 10})

    tracker.record_event(
        "run_complete",
        stop_reason=final_state.get("stop_reason"),
        best_score=final_state.get("best_score"),
        iterations=len(final_state.get("history", [])),
    )

    return final_state  # type: ignore[return-value]
