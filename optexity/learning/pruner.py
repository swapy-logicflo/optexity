"""LLM pass that removes semantically redundant actions the compiler's rules cannot see.

Rules catch errors and steps the agent itself judged failures, but not a retry the agent judged a
success, or a navigation it made only to re-check the page. The model only chooses which cached
actions to keep: it cannot add actions or change locators, which stay as resolved during the run.
"""

import json
import logging

from browser_use.agent.action_cache import CachedAction
from pydantic import BaseModel, Field

from optexity.inference.models import get_llm_model_with_fallback
from optexity.learning.compiler import describe

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You turn a browser agent's execution trace into the shortest sequence of actions "
    "that accomplishes the task when replayed deterministically."
)

INSTRUCTIONS = """Replay starts on the first action's page in a fresh session and runs the kept actions in order.
Keep exactly the actions the task needs. Remove:
- retries and alternative attempts at something an earlier kept action already achieved
- navigations or scripts the agent used only to check or recover its view of the page
- anything after the task's goal was reached
Never remove an action that enters data or triggers an effect the task asked for."""


class PruneDecision(BaseModel):
    keep: list[int] = Field(description="Ids of the actions to keep, in their original order")
    reasoning: str = Field(description="Why each removed action is not needed, one short sentence each")


def _render(action_id: int, action: CachedAction) -> str:
    # Typed values may be secrets and don't decide whether a step is needed.
    params = {k: v for k, v in action.params.items() if not (action.name == "input" and k == "text")}
    target = describe(action.element) if action.element else "-"
    return (
        f"[{action_id}] {action.name} target={target} params={json.dumps(params)} page={action.url}\n"
        f"     agent's goal: {action.goal}\n"
        f"     agent's verdict afterwards: {action.outcome}"
    )


def build_prompt(task: str, actions: list[CachedAction], feedback: str | None = None) -> str:
    rendered = "\n".join(_render(i, action) for i, action in enumerate(actions))
    prompt = f"Task: {task}\n\nActions the agent executed, in order:\n{rendered}\n\n{INSTRUCTIONS}"
    if feedback:
        prompt += f"\n\nYour previous answer was invalid: {feedback}"
    return prompt


def invalid_ids(keep: list[int], action_count: int) -> str | None:
    if not keep:
        return "keep at least one action"
    unknown = sorted({i for i in keep if not 0 <= i < action_count})
    if unknown:
        return f"ids {unknown} do not exist; valid ids are 0 to {action_count - 1}"
    return None


def llm_prune(
    task: str,
    actions: list[CachedAction],
    provider: str | None = None,
    model_name: str | None = None,
) -> list[CachedAction]:
    """Keep only the actions the model says the task needs, falling back to all of them on failure."""
    try:
        model = get_llm_model_with_fallback(provider, model_name, True)
        feedback = None
        for _ in range(2):
            decision, usage = model.get_model_response_with_structured_output(
                prompt=build_prompt(task, actions, feedback),
                response_schema=PruneDecision,
                system_instruction=SYSTEM_INSTRUCTION,
            )
            feedback = invalid_ids(decision.keep, len(actions))
            if feedback is None:
                kept = sorted(set(decision.keep))
                logger.info(
                    f"LLM pruning kept {len(kept)}/{len(actions)} actions "
                    f"({usage.total_tokens} tokens): {decision.reasoning}"
                )
                return [actions[i] for i in kept]
            logger.warning(f"LLM pruning returned invalid ids: {feedback}")
    except Exception as e:
        logger.warning(f"LLM pruning failed: {e}")
    logger.warning("Keeping all rule-filtered actions")
    return actions
