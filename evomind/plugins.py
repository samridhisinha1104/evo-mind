"""Plugin architecture for EvoMind.

Users drop a `.py` file into the `plugins/` directory. Each plugin can register:
- Analysis steps (functions the executor can call)
- Custom evaluators (alternative scoring functions)
- Custom mutation operators

Plugins are auto-discovered at startup. Each plugin file must define a
`register(registry)` function that receives the PluginRegistry.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from evomind.state import StepResult

# Default plugin directory (next to the evomind package)
DEFAULT_PLUGIN_DIR = Path(__file__).resolve().parent.parent / "plugins"


class PluginRegistry:
    """Central registry that plugins use to register their extensions."""

    def __init__(self) -> None:
        self._steps: dict[str, Callable[[pd.DataFrame, dict], StepResult]] = {}
        self._evaluators: dict[str, Callable] = {}
        self._mutation_operators: dict[str, Callable] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    # --- Registration methods (called by plugins) ---

    def register_step(
        self,
        name: str,
        fn: Callable[[pd.DataFrame, dict], StepResult],
        description: str = "",
        required_dtypes: list[str] | None = None,
    ) -> None:
        """Register a new analysis step."""
        self._steps[name] = fn
        self._metadata[f"step:{name}"] = {
            "type": "step",
            "name": name,
            "description": description,
            "required_dtypes": required_dtypes or [],
        }

    def register_evaluator(
        self,
        name: str,
        fn: Callable,
        description: str = "",
    ) -> None:
        """Register a custom evaluator function."""
        self._evaluators[name] = fn
        self._metadata[f"evaluator:{name}"] = {
            "type": "evaluator",
            "name": name,
            "description": description,
        }

    def register_mutation_operator(
        self,
        name: str,
        fn: Callable,
        description: str = "",
    ) -> None:
        """Register a custom mutation operator."""
        self._mutation_operators[name] = fn
        self._metadata[f"mutation:{name}"] = {
            "type": "mutation_operator",
            "name": name,
            "description": description,
        }

    # --- Accessors ---

    @property
    def steps(self) -> dict[str, Callable]:
        return dict(self._steps)

    @property
    def evaluators(self) -> dict[str, Callable]:
        return dict(self._evaluators)

    @property
    def mutation_operators(self) -> dict[str, Callable]:
        return dict(self._mutation_operators)

    @property
    def step_names(self) -> list[str]:
        return list(self._steps.keys())

    def get_metadata(self, key: str) -> dict[str, Any]:
        return self._metadata.get(key, {})

    def all_metadata(self) -> dict[str, dict[str, Any]]:
        return dict(self._metadata)


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

def discover_plugins(plugin_dir: str | Path | None = None) -> list[Path]:
    """Find all .py files in the plugin directory."""
    plugin_dir = Path(plugin_dir) if plugin_dir else DEFAULT_PLUGIN_DIR
    if not plugin_dir.exists():
        return []
    return sorted(plugin_dir.glob("*.py"))


def load_plugins(
    plugin_dir: str | Path | None = None,
    registry: PluginRegistry | None = None,
) -> PluginRegistry:
    """Discover and load all plugins, returning a populated registry."""
    registry = registry or PluginRegistry()
    plugin_files = discover_plugins(plugin_dir)

    for plugin_path in plugin_files:
        _load_single_plugin(plugin_path, registry)

    return registry


def _load_single_plugin(plugin_path: Path, registry: PluginRegistry) -> None:
    """Import a single plugin file and call its register() function."""
    module_name = f"evomind_plugin_{plugin_path.stem}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, "register"):
            module.register(registry)
    except Exception as exc:
        # Don't crash the whole system for a bad plugin
        import logging
        logging.getLogger("evomind").warning(
            f"Failed to load plugin {plugin_path.name}: {exc}"
        )


# ---------------------------------------------------------------------------
# Integration helper
# ---------------------------------------------------------------------------

def merge_plugin_steps(
    base_registry: dict[str, Callable],
    base_available: list[str],
    plugin_registry: PluginRegistry,
) -> tuple[dict[str, Callable], list[str]]:
    """Merge plugin steps into the base step registry and available steps list."""
    merged_registry = {**base_registry, **plugin_registry.steps}
    merged_available = list(base_available) + [
        name for name in plugin_registry.step_names if name not in base_available
    ]
    return merged_registry, merged_available
