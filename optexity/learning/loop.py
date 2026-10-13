"""Iteratively compile an automation down to zero LLM calls.

Each iteration runs the automation and folds back what the run learned:
- a node that ran agentically (an agentic_task node, or a deterministic node that fell back to an agent)
  left an action cache, which is compiled and spliced in place of that node;
- a node whose locator failed, but whose element the index predictor found, left that element,
  which becomes the node's new locator.
The loop stops once a run succeeds without either, i.e. with no LLM calls at runtime.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx
from browser_use.agent.action_cache import ActionCache, CachedElement

from optexity.learning.compiler import (
    Pruner,
    automation_to_json,
    compile_automation,
    locator_command,
)
from optexity.learning.files import ACTION_CACHE_FILE, PREDICTED_ELEMENT_FILE
from optexity.schema.actions.interaction_action import BaseAction
from optexity.schema.automation import ActionNode, Automation
from optexity.schema.task import Task

STATUS_LINE = re.compile(r"completed with status (\w+)")
STEP_DIRECTORY = re.compile(r"step_(\d+)$")


@dataclass
class RunResult:
    task_id: str
    status: str | None
    # Keyed by node index: what an agentic run of that node did, and what the index predictor chose.
    caches: dict[int, ActionCache] = field(default_factory=dict)
    predicted: dict[int, CachedElement] = field(default_factory=dict)
    tokens: int = 0
    seconds: float | None = None

    @property
    def llm_calls(self) -> int:
        # Each index prediction is one call; agentic runs report their own count.
        return sum(cache.llm_calls for cache in self.caches.values()) + len(self.predicted)


def _agent_succeeded(cache: ActionCache) -> bool:
    return any(action.is_done and action.params.get("success") for action in cache.actions)


def read_run(task_id: str, logs_directory: Path) -> RunResult:
    log = (logs_directory / "optexity.log").read_text(errors="replace")
    statuses = STATUS_LINE.findall(log)
    run = RunResult(task_id=task_id, status=statuses[-1] if statuses else None)

    started, finished = None, None
    memory_tokens = 0
    for step_directory in logs_directory.iterdir():
        match = STEP_DIRECTORY.search(step_directory.name)
        if not match:
            continue
        index = int(match.group(1))
        if (cache_file := step_directory / ACTION_CACHE_FILE).exists():
            run.caches[index] = ActionCache.load(cache_file)
        if (element_file := step_directory / PREDICTED_ELEMENT_FILE).exists():
            run.predicted[index] = CachedElement.model_validate_json(element_file.read_text())
        if (state_file := step_directory / "state.json").exists():
            state = json.loads(state_file.read_text())
            started = datetime.fromisoformat(state["started_at"])
            completed = datetime.fromisoformat(state["completed_at"])
            finished = max(finished, completed) if finished else completed
            # Memory's usage is cumulative and covers index predictions, not agentic runs.
            memory_tokens = max(memory_tokens, state.get("token_usage", {}).get("total_tokens", 0))

    run.tokens = memory_tokens + sum(c.usage.total_tokens for c in run.caches.values() if c.usage)
    if started and finished:
        run.seconds = (finished - started).total_seconds()
    return run


def with_locator(node: ActionNode, element: CachedElement) -> ActionNode:
    interaction = node.interaction_action
    if interaction is None:
        return node
    for name in type(interaction).model_fields:
        action = getattr(interaction, name)
        if isinstance(action, BaseAction):
            updated = action.model_copy(update={"command": locator_command(element), "skip_command": False})
            return node.model_copy(update={"interaction_action": interaction.model_copy(update={name: updated})})
    return node


def learn_from_run(automation: Automation, run: RunResult, prune: Pruner | None) -> Automation | None:
    """The automation with this run's lessons folded in, or None if the run taught nothing."""
    if not all(isinstance(node, ActionNode) for node in automation.nodes):
        raise ValueError("The learning loop only supports flat automations, where step index equals node index")

    nodes = list(automation.nodes)
    learned = False
    # Splice from the back so earlier indices stay valid.
    for index in sorted(set(run.caches) | set(run.predicted), reverse=True):
        cache = run.caches.get(index)
        if cache is not None and _agent_succeeded(cache):
            compiled = compile_automation(cache, prune).nodes
            if compiled:
                nodes[index : index + 1] = compiled
                learned = True
                continue
        if index in run.predicted:
            nodes[index] = with_locator(nodes[index], run.predicted[index])
            learned = True

    return automation.model_copy(update={"nodes": nodes}) if learned else None


@dataclass
class InferenceServer:
    """Runs automations on a local `optexity inference` server started with OPTEXITY_LOCAL_AUTOMATION=automation_file."""

    url: str
    endpoint_name: str
    input_parameters: dict[str, list[str]]
    automation_file: Path
    timeout_seconds: float = 900

    def run(self, automation: Automation) -> RunResult:
        self.automation_file.write_text(automation_to_json(automation))
        response = httpx.post(
            f"{self.url}/inference",
            json={"endpoint_name": self.endpoint_name, "input_parameters": self.input_parameters},
            timeout=60,
        )
        response.raise_for_status()
        task_id = response.json()["task_id"]
        logs_directory = Task.model_fields["save_directory"].default / task_id / "logs"

        deadline = time.monotonic() + self.timeout_seconds
        log_file = logs_directory / "optexity.log"
        while time.monotonic() < deadline:
            if log_file.exists() and STATUS_LINE.search(log_file.read_text(errors="replace")):
                return read_run(task_id, logs_directory)
            time.sleep(2)
        raise TimeoutError(f"Task {task_id} did not finish within {self.timeout_seconds}s")


def learn(
    automation: Automation,
    run: Callable[[Automation], RunResult],
    prune: Pruner | None = None,
    max_iterations: int = 4,
    on_iteration: Callable[[int, Automation, RunResult], None] = lambda *_: None,
) -> tuple[Automation, list[RunResult]]:
    results: list[RunResult] = []
    for iteration in range(max_iterations):
        result = run(automation)
        results.append(result)
        on_iteration(iteration, automation, result)
        if result.status == "success" and result.llm_calls == 0:
            break
        improved = learn_from_run(automation, result, prune)
        if improved is None:
            break
        automation = improved
    return automation, results
