"""Compile a browser-use action cache into a deterministic Optexity automation.

The agentic run already resolved every element it touched, so each replayable action becomes a node
that needs no LLM to find its target on the next run: a Playwright locator command for interactions,
an extraction node for data the agent read off the page. An action with no registered compiler stays
agentic, scoped down to that action's goal instead of the whole task.
"""

import json
import re
from collections.abc import Callable

from browser_use.agent.action_cache import ActionCache, CachedAction, CachedElement

from optexity.schema.actions.extraction_action import ExtractionAction, LLMExtraction
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
# Scrolling and waiting are included because Playwright scrolls to an element and waits for it before
# acting, and an agent's waits mostly pace its own LLM loop.
NON_REPLAYED_ACTIONS = {
    "done",
    "scroll",
    "wait",
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
# A digit run only counts against an id when it's set off by a separator or a known prefix (mui-48213,
# ember1234, field_9931) — a bare "\d{3,}" would also reject real ids like "zip94105" or "order2024".
GENERATED_ID = re.compile(
    r"^(ember|react|mui|radix)[-_]?\d|^[0-9a-f]{8,}$|^:|[-_:]\d{3,}$", re.IGNORECASE
)

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


# Compilers keyed by browser-use action name. A compiler returns None when this particular action
# can't be made deterministic; compile_action then keeps it agentic. Element compilers run only when
# the recorder resolved an element for the action.
ActionCompiler = Callable[[CachedAction], ActionNode | None]
ElementCompiler = Callable[[CachedAction, CachedElement], ActionNode]
ACTION_COMPILERS: dict[str, ActionCompiler] = {}
ELEMENT_COMPILERS: dict[str, ElementCompiler] = {}


def compiles(name: str) -> Callable[[ActionCompiler], ActionCompiler]:
    def register(compiler: ActionCompiler) -> ActionCompiler:
        ACTION_COMPILERS[name] = compiler
        return compiler

    return register


def compiles_on_element(name: str) -> Callable[[ElementCompiler], ElementCompiler]:
    def register(compiler: ElementCompiler) -> ElementCompiler:
        ELEMENT_COMPILERS[name] = compiler
        return compiler

    return register


def _interaction_node(**action) -> ActionNode:
    return ActionNode(type="action_node", interaction_action=InteractionAction(**action))


@compiles("navigate")
def _navigate(action: CachedAction) -> ActionNode:
    params = action.params
    return _interaction_node(
        go_to_url=GoToUrlAction(url=params["url"], new_tab=params.get("new_tab", False))
    )


@compiles("go_back")
def _go_back(action: CachedAction) -> ActionNode:
    return _interaction_node(go_back=GoBackAction())


@compiles("send_keys")
def _send_keys(action: CachedAction) -> ActionNode | None:
    keys = action.params.get("keys")
    return _interaction_node(key_press=KeyPressAction(type=keys)) if keys in KEY_NAMES else None


@compiles("extract")
def _extract(action: CachedAction) -> ActionNode:
    # browser-use registers extract without a param_model, so its params are recorded nested under
    # "params"; accept the flat shape too in case that's fixed upstream.
    params = action.params.get("params", action.params)
    # The agent's query is free text with no schema, so the extracted data comes back as one string.
    return ActionNode(
        type="action_node",
        extraction_action=ExtractionAction(
            llm=LLMExtraction(
                extraction_format={"result": "str"},
                extraction_instructions=params["query"],
            )
        ),
    )


@compiles_on_element("input")
def _input(action: CachedAction, element: CachedElement) -> ActionNode:
    return _interaction_node(
        input_text=InputTextAction(
            command=locator_command(element),
            prompt_instructions=f"Enter text into the {describe(element)}",
            input_text=action.params["text"],
        )
    )


@compiles_on_element("click")
def _click(action: CachedAction, element: CachedElement) -> ActionNode:
    return _interaction_node(
        click_element=ClickElementAction(
            command=locator_command(element),
            prompt_instructions=f"Click the {describe(element)}",
        )
    )


@compiles_on_element("select_dropdown")
def _select(action: CachedAction, element: CachedElement) -> ActionNode:
    option = action.params["text"]
    return _interaction_node(
        select_option=SelectOptionAction(
            command=locator_command(element),
            prompt_instructions=f"Select '{option}' in the {describe(element)}",
            select_values=[option],
        )
    )


def _agentic_fallback(action: CachedAction, task: str) -> ActionNode:
    return _interaction_node(
        agentic_task=AgenticTask(task=action.goal or task, max_steps=FALLBACK_MAX_STEPS)
    )


def compile_action(action: CachedAction, task: str) -> ActionNode:
    node = None
    if action.name in ELEMENT_COMPILERS:
        if action.element is not None:
            node = ELEMENT_COMPILERS[action.name](action, action.element)
    elif action.name in ACTION_COMPILERS:
        node = ACTION_COMPILERS[action.name](action)
    return node or _agentic_fallback(action, task)


def _agentic_task(node: ActionNode) -> AgenticTask | None:
    return node.interaction_action.agentic_task if node.interaction_action else None


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
        node = compile_action(action, cache.task)
        # Uncompilable actions from the same step share one goal, so one scoped agentic node covers them.
        agentic = _agentic_task(node)
        if agentic is not None and nodes and _agentic_task(nodes[-1]) == agentic:
            continue
        input_text = node.interaction_action.input_text if node.interaction_action else None
        if parameterize and input_text and action.element is not None:
            name = parameter_name(action.element, set(input_parameters))
            input_parameters[name] = [input_text.input_text]
            input_text.input_text = f"{{{name}[0]}}"
        nodes.append(node)

    return Automation(
        url=cache.start_url,
        parameters=Parameters(input_parameters=input_parameters, generated_parameters={}),
        nodes=nodes,
    )


def automation_to_json(automation: Automation) -> str:
    return json.dumps(automation.model_dump(mode="json", exclude_defaults=True), indent=2)
