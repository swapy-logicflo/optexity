"""Compile a browser-use action cache into a deterministic Optexity automation.

The agentic run already resolved every element it touched, so each replayable action becomes a node
with a Playwright locator command and needs no LLM on the next run. An action that can't be compiled
safely stays agentic, scoped down to that action's goal instead of the whole task.
"""

import json
import re
from collections.abc import Callable

from browser_use.agent.action_cache import ActionCache, CachedAction, CachedElement

from optexity.schema.actions.interaction_action import (
    AgenticTask,
    ClickElementAction,
    GoBackAction,
    GoToUrlAction,
    InputTextAction,
    InteractionAction,
    KeyPressAction,
    SelectOptionAction,
)
from optexity.schema.actions.keyboard_keys import KEY_NAMES
from optexity.schema.automation import ActionNode, Automation, Parameters

# Actions that only serve the agent's own reasoning; replaying them changes nothing on the page.
# Scrolling is included because Playwright scrolls an element into view before acting on it.
NON_REPLAYED_ACTIONS = {
    "done",
    "scroll",
    "wait",
    "extract",
    "screenshot",
    "find_text",
    "dropdown_options",
    "read_file",
    "write_file",
    "replace_file",
}

# Playwright's get_by_role only accepts ARIA roles; Chrome's accessibility tree also reports internal ones.
ARIA_ROLES = {
    "button",
    "checkbox",
    "combobox",
    "link",
    "listbox",
    "menuitem",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
}

# Ids that look framework-generated are regenerated on each render, so they are not stable identities.
GENERATED_ID = re.compile(r"\d{3,}|[0-9a-f]{8,}|^:|^(ember|react|mui|radix)", re.IGNORECASE)

# browser-use's system prompt has the agent close each evaluation with "Verdict: Success|Failure|Uncertain".
UNSUCCESSFUL_VERDICT = re.compile(r"verdict:\s*(failure|uncertain)", re.IGNORECASE)

FALLBACK_MAX_STEPS = 5

# Narrows the rule-filtered actions further, e.g. by asking an LLM which ones the task needed.
Pruner = Callable[[str, list[CachedAction]], list[CachedAction]]


def _css_attribute(tag: str, attribute: str, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"{tag}[{attribute}='{escaped}']"


def _short_text(text: str | None) -> str | None:
    if text and "\n" not in text and len(text) <= 80:
        return text
    return None


def locator_command(element: CachedElement) -> str:
    """Most stable Playwright locator for the element, as the command string Optexity evaluates.

    CSS and role/text locators pierce open shadow roots; XPath does not, so it is the last resort.
    Values come from the page and the command is passed to eval, so every literal goes through
    json.dumps to stay a plain string.
    """
    attributes = element.attributes
    element_id = attributes.get("id")
    if element_id and not GENERATED_ID.search(element_id):
        return f"locator({json.dumps(_css_attribute(element.tag, 'id', element_id))}).first"
    for attribute in ("name", "data-testid", "aria-label", "placeholder"):
        if value := attributes.get(attribute):
            return f"locator({json.dumps(_css_attribute(element.tag, attribute, value))}).first"
    if element.ax_role in ARIA_ROLES and element.ax_name:
        return (
            f"get_by_role({json.dumps(element.ax_role)}, "
            f"name={json.dumps(element.ax_name)}, exact=True).first"
        )
    if text := _short_text(element.text):
        return f"get_by_text({json.dumps(text)}, exact=True).first"
    return f"locator({json.dumps('xpath=/' + element.x_path)}).first"


def describe(element: CachedElement) -> str:
    """Human-readable target, used as prompt_instructions when the locator stops matching."""
    attributes = element.attributes
    label = (
        element.ax_name
        or _short_text(element.text)
        or attributes.get("aria-label")
        or attributes.get("placeholder")
        or attributes.get("name")
        or attributes.get("id")
    )
    kind = element.ax_role or element.tag
    return f"{kind} '{label}'" if label else kind


def _same_target(a: CachedAction, b: CachedAction) -> bool:
    if a.element is None or b.element is None or a.url != b.url:
        return False
    return a.element.element_hash == b.element.element_hash or a.element.x_path == b.element.x_path


def replayable_actions(cache: ActionCache) -> list[CachedAction]:
    """Drop failed, reasoning-only and superseded actions; what remains changed the page."""
    actions = [
        action
        for action in cache.actions
        if action.error is None
        and action.name not in NON_REPLAYED_ACTIONS
        # The agent judged the step ineffective and retried; replaying it would repeat the dead end.
        and not (action.outcome and UNSUCCESSFUL_VERDICT.search(action.outcome))
    ]
    # The automation already opens its start url, so the agent navigating there first is redundant.
    if actions and actions[0].name == "navigate" and actions[0].params.get("url") == cache.start_url:
        actions = actions[1:]

    kept = []
    for i, action in enumerate(actions):
        # A later clearing write to the same field overrides this one.
        overridden = action.name == "input" and any(
            later.name == "input" and later.params.get("clear", True) and _same_target(action, later)
            for later in actions[i + 1 :]
        )
        if not overridden:
            kept.append(action)
    return kept


def _agentic_fallback(action: CachedAction, task: str) -> InteractionAction:
    return InteractionAction(
        agentic_task=AgenticTask(task=action.goal or task, max_steps=FALLBACK_MAX_STEPS)
    )


def compile_action(action: CachedAction, task: str) -> InteractionAction:
    params = action.params
    element = action.element

    if action.name == "navigate":
        return InteractionAction(
            go_to_url=GoToUrlAction(url=params["url"], new_tab=params.get("new_tab", False))
        )
    if action.name == "go_back":
        return InteractionAction(go_back=GoBackAction())
    if action.name == "send_keys" and params.get("keys") in KEY_NAMES:
        return InteractionAction(key_press=KeyPressAction(type=params["keys"]))

    if element is None:
        return _agentic_fallback(action, task)

    command = locator_command(element)
    target = describe(element)
    if action.name == "input":
        return InteractionAction(
            input_text=InputTextAction(
                command=command,
                prompt_instructions=f"Enter text into the {target}",
                input_text=params["text"],
            )
        )
    if action.name == "click":
        return InteractionAction(
            click_element=ClickElementAction(
                command=command, prompt_instructions=f"Click the {target}"
            )
        )
    if action.name == "select_dropdown":
        return InteractionAction(
            select_option=SelectOptionAction(
                command=command,
                prompt_instructions=f"Select '{params['text']}' in the {target}",
                select_values=[params["text"]],
            )
        )
    return _agentic_fallback(action, task)


def parameter_name(element: CachedElement, taken: set[str]) -> str:
    """Identifier for the value typed into this field, e.g. name=04fullname -> fullname."""
    attributes = element.attributes
    raw = attributes.get("name") or attributes.get("id") or element.ax_name or attributes.get("placeholder") or ""
    base = re.sub(r"^[\d_]+", "", re.sub(r"\W+", "_", raw).strip("_").lower()) or "value"
    name, suffix = base, 2
    while name in taken:
        name, suffix = f"{base}_{suffix}", suffix + 1
    return name


def compile_automation(cache: ActionCache, prune: Pruner | None = None, parameterize: bool = False) -> Automation:
    """With parameterize, typed values become input parameters defaulting to the recorded values."""
    if cache.start_url is None:
        raise ValueError("Action cache has no start url; the agent never observed a page")

    actions = replayable_actions(cache)
    if prune is not None and actions:
        actions = prune(cache.task, actions)

    nodes: list[ActionNode] = []
    input_parameters: dict[str, list[str | int | float | bool]] = {}
    for action in actions:
        interaction = compile_action(action, cache.task)
        previous = nodes[-1].interaction_action if nodes else None
        # Uncompilable actions from the same step share one goal, so one scoped agentic node covers them.
        if (
            interaction.agentic_task
            and previous is not None
            and previous.agentic_task == interaction.agentic_task
        ):
            continue
        if parameterize and interaction.input_text and action.element is not None:
            name = parameter_name(action.element, set(input_parameters))
            input_parameters[name] = [interaction.input_text.input_text]
            interaction.input_text.input_text = f"{{{name}[0]}}"
        nodes.append(ActionNode(type="action_node", interaction_action=interaction))

    return Automation(
        url=cache.start_url,
        parameters=Parameters(input_parameters=input_parameters, generated_parameters={}),
        nodes=nodes,
    )


def automation_to_json(automation: Automation) -> str:
    return json.dumps(automation.model_dump(mode="json", exclude_defaults=True), indent=2)
