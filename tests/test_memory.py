from evomind.memory import StrategyMemory


def make_strategy(name="s1"):
    return {
        "name": name, "description": "d", "steps": ["describe_data"], "params": {},
        "parent_name": "", "mutation_applied": "", "generation": 0,
    }


def test_save_and_get_best(tmp_path):
    mem = StrategyMemory(tmp_path / "test.db")
    mem.save_strategy("sig_a", make_strategy("low"), 0.2)
    mem.save_strategy("sig_a", make_strategy("high"), 0.9)
    mem.save_strategy("sig_b", make_strategy("other_sig"), 0.99)

    best = mem.get_best_strategies("sig_a", top_k=2)
    assert len(best) == 2
    assert best[0]["name"] == "high"
    assert best[0]["score"] == 0.9
    assert best[1]["name"] == "low"


def test_get_best_strategies_empty_for_unknown_signature(tmp_path):
    mem = StrategyMemory(tmp_path / "test.db")
    assert mem.get_best_strategies("never_seen") == []


def test_record_and_read_run_history(tmp_path):
    mem = StrategyMemory(tmp_path / "test.db")
    mem.record_run("sig_a", 0, make_strategy("s0"), 0.1)
    mem.record_run("sig_a", 1, make_strategy("s1"), 0.4)

    history = mem.history_for("sig_a")
    assert [h["iteration"] for h in history] == [0, 1]
    assert history[1]["score"] == 0.4


def test_global_best_spans_signatures(tmp_path):
    mem = StrategyMemory(tmp_path / "test.db")
    mem.save_strategy("sig_a", make_strategy("a"), 0.5)
    mem.save_strategy("sig_b", make_strategy("b"), 0.95)

    top = mem.get_global_best(top_k=1)
    assert top[0]["name"] == "b"


def test_save_with_lineage(tmp_path):
    """Test that lineage info is persisted correctly."""
    mem = StrategyMemory(tmp_path / "test.db")
    parent_id = mem.save_strategy(
        "sig_a", make_strategy("parent"), 0.5,
        mutation_type="initial", generation=0,
    )
    child_id = mem.save_strategy(
        "sig_a", make_strategy("child"), 0.7,
        parent_id=parent_id, mutation_type="add_step", generation=1,
    )

    lineage = mem.get_lineage(child_id)
    assert len(lineage) == 2
    assert lineage[0]["name"] == "child"
    assert lineage[0]["mutation_type"] == "add_step"
    assert lineage[1]["name"] == "parent"


def test_evolution_tree(tmp_path):
    """Test that we can get the full evolution tree for a task."""
    mem = StrategyMemory(tmp_path / "test.db")
    id1 = mem.save_strategy("sig_a", make_strategy("gen0"), 0.3, generation=0)
    id2 = mem.save_strategy("sig_a", make_strategy("gen1"), 0.5, parent_id=id1, mutation_type="add_step", generation=1)
    id3 = mem.save_strategy("sig_a", make_strategy("gen2"), 0.7, parent_id=id2, mutation_type="swap_step", generation=2)

    tree = mem.get_evolution_tree("sig_a")
    assert len(tree) == 3
    assert tree[0]["generation"] == 0
    assert tree[2]["generation"] == 2


def test_task_embeddings_and_similarity(tmp_path):
    """Test embedding storage and cosine similarity search."""
    mem = StrategyMemory(tmp_path / "test.db")
    mem.save_strategy("sig_a", make_strategy("a"), 0.8)
    mem.save_strategy("sig_b", make_strategy("b"), 0.9)

    # Store embeddings
    mem.save_task_embedding("sig_a", "find patterns in sales data", [1.0, 0.0, 0.0])
    mem.save_task_embedding("sig_b", "discover trends in revenue", [0.9, 0.1, 0.0])

    # Query similar tasks
    similar = mem.find_similar_tasks([1.0, 0.0, 0.0], top_k=2)
    assert len(similar) == 2
    assert similar[0]["task_signature"] == "sig_a"  # exact match should be most similar

    # Cross-task transfer
    results = mem.get_strategies_for_similar_tasks(
        [0.95, 0.05, 0.0],
        top_k_tasks=2,
        top_k_strategies=1,
    )
    assert len(results) >= 1


def test_integration_lineage_propagation(tmp_path):
    """Test that reflector_node saves and updates strategy IDs and planner_node propagates parent_id."""
    from evomind.nodes import planner_node, reflector_node
    from evomind.state import EvoState
    from unittest.mock import MagicMock

    mem = StrategyMemory(tmp_path / "test.db")

    # 1. First iteration (iteration 0)
    # Reflector node runs and saves the initial strategy, returning ID
    state: EvoState = {
        "task_description": "Analyze housing price trends",
        "task_signature": "sig_housing",
        "strategy": {
            "name": "initial_strat",
            "description": "Baseline",
            "steps": ["describe_data"],
            "params": {},
            "parent_name": "",
            "mutation_applied": "initial",
            "generation": 0,
        },
        "evaluation": {
            "score": 0.5,
            "rationale": "Okay baseline",
            "insight_depth": 0.5,
            "coverage": 0.5,
            "efficiency": 0.5,
            "novelty": 0.5,
            "signal_count": 1,
        },
        "iteration": 0,
        "history": [],
    }

    reflector_out = reflector_node(state, memory=mem)
    state.update(reflector_out)

    assert "id" in state["strategy"]
    first_id = state["strategy"]["id"]
    assert first_id > 0

    # 2. Second iteration (iteration 1)
    # Planner node runs to propose a mutated strategy. It should set the parent_id to first_id
    mock_llm = MagicMock()
    mock_llm.complete_json.return_value = {"operators": ["add_step"], "reasoning": "Let's add missing values."}

    planner_out = planner_node(state, llm=mock_llm, memory=mem)
    state.update(planner_out)

    assert state["strategy"]["parent_id"] == first_id
    assert state["strategy"]["generation"] == 1
    assert state["strategy"]["parent_name"] == "initial_strat"

    # Reflector node runs again for the mutated strategy
    state["evaluation"] = {
        "score": 0.75,
        "rationale": "Better coverage",
        "insight_depth": 0.7,
        "coverage": 0.8,
        "efficiency": 0.75,
        "novelty": 0.6,
        "signal_count": 3,
    }
    state["iteration"] = 1

    reflector_out = reflector_node(state, memory=mem)
    state.update(reflector_out)

    second_id = state["strategy"]["id"]
    assert second_id > first_id

    # Verify lineage path in DB
    lineage = mem.get_lineage(second_id)
    assert len(lineage) == 2
    assert lineage[0]["id"] == second_id
    assert lineage[0]["parent_id"] == first_id
    assert lineage[1]["id"] == first_id

