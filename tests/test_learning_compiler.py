import ast
import json

from browser_use.agent.action_cache import ActionCache, CachedAction, CachedElement

from optexity.learning.compiler import (
    automation_to_json,
    compile_automation,
    locator_command,
    replayable_actions,
)
from optexity.schema.automation import Automation

URL = "https://www.roboform.com/filling-test-all-fields"


def element(tag="input", attributes=None, x_path="html/body/form/input", text=None, ax_role=None, ax_name=None):
    return CachedElement(
        tag=tag,
        attributes=attributes or {},
        x_path=x_path,
        element_hash=hash((tag, x_path, json.dumps(attributes, sort_keys=True))),
        text=text,
        ax_role=ax_role,
        ax_name=ax_name,
    )


def action(name, params=None, el=None, url=URL, goal="goal", error=None, is_done=False, outcome=None):
    return CachedAction(
        step_number=1,
        name=name,
        params=params or {},
        element=el,
        url=url,
        goal=goal,
        outcome=outcome,
        error=error,
        is_done=is_done,
    )


def cache(*actions):
    return ActionCache(task="task", start_url=URL, actions=list(actions), llm_calls=1, duration_seconds=1.0, usage=None)


def called_with(command):
    """The single string argument of the locator call, proving the command is one call with a literal."""
    call = ast.parse(command, mode="eval").body.value  # strip the trailing .first
    assert isinstance(call, ast.Call) and len(call.args) == 1
    assert isinstance(call.args[0], ast.Constant)
    return call.func.id, call.args[0].value


class TestLocatorCommand:
    def test_prefers_stable_id(self):
        el = element(attributes={"id": "email", "name": "user_email"})
        assert locator_command(el) == "locator(\"input[id='email']\").first"

    def test_skips_generated_id_for_name(self):
        el = element(attributes={"id": "mui-48213", "name": "04fullname"})
        assert locator_command(el) == "locator(\"input[name='04fullname']\").first"

    def test_role_and_accessible_name_for_links(self):
        el = element(tag="a", attributes={"href": "/download"}, ax_role="link", ax_name="File Download")
        assert locator_command(el) == 'get_by_role("link", name="File Download", exact=True).first'

    def test_ignores_non_aria_roles(self):
        el = element(tag="span", ax_role="StaticText", ax_name="Hello", text="Hello")
        assert locator_command(el) == 'get_by_text("Hello", exact=True).first'

    def test_xpath_is_last_resort(self):
        el = element(tag="div", x_path="html/body/div[3]")
        assert locator_command(el) == 'locator("xpath=/html/body/div[3]").first'

    def test_page_controlled_values_stay_a_single_string_literal(self):
        hostile = 'x\'"]) or __import__("os").system("echo pwned") #\\'
        method, selector = called_with(locator_command(element(attributes={"name": hostile})))
        assert method == "locator"
        assert selector.startswith("input[name='") and "echo pwned" in selector


class TestReplayableActions:
    def test_drops_failed_reasoning_only_and_done_actions(self):
        fill = action("input", {"text": "a"}, element(attributes={"name": "a"}))
        kept = replayable_actions(
            cache(
                action("scroll", {"down": True}),
                action("input", {"text": "a"}, element(attributes={"name": "b"}), error="not found"),
                fill,
                action("done", {"text": "ok"}, is_done=True),
            )
        )
        assert kept == [fill]

    def test_drops_navigation_to_start_url(self):
        click = action("click", el=element(tag="a", ax_role="link", ax_name="Next"))
        kept = replayable_actions(cache(action("navigate", {"url": URL}), click))
        assert kept == [click]

    def test_later_clearing_write_overrides_earlier(self):
        field = element(attributes={"name": "city"})
        final = action("input", {"text": "SF", "clear": True}, field)
        kept = replayable_actions(cache(action("input", {"text": "LA", "clear": True}, field), final))
        assert kept == [final]

    def test_drops_steps_the_agent_judged_failed_or_uncertain(self):
        logout = element(tag="a", ax_role="link", ax_name="Logout")
        confirmed = action("click", el=logout, outcome="Logged out. Verdict: Success")
        kept = replayable_actions(
            cache(
                action("click", el=logout, outcome="Still on the secure area. Verdict: Uncertain/Failure"),
                action("navigate", {"url": f"{URL}/logout"}, outcome="Verdict: Failure"),
                confirmed,
            )
        )
        assert kept == [confirmed]

    def test_appending_write_keeps_earlier(self):
        field = element(attributes={"name": "city"})
        first = action("input", {"text": "San", "clear": True}, field)
        append = action("input", {"text": " Francisco", "clear": False}, field)
        assert replayable_actions(cache(first, append)) == [first, append]


class TestCompileAutomation:
    def test_form_fill_compiles_to_deterministic_inputs(self):
        fields = [("04fullname", "myname"), ("10address1", "xyz"), ("11address2", "abc"), ("13adr_city", "SF")]
        automation = compile_automation(
            cache(
                *[action("input", {"text": t, "clear": True}, element(attributes={"name": n}, x_path=f"html/{n}")) for n, t in fields],
                action("done", {"text": "ok"}, is_done=True),
            )
        )
        inputs = [node.interaction_action.input_text for node in automation.nodes]
        assert [(i.command, i.input_text) for i in inputs] == [
            (f"locator(\"input[name='{n}']\").first", t) for n, t in fields
        ]
        assert not any(node.interaction_action.agentic_task for node in automation.nodes)

    def test_output_json_validates_as_automation(self):
        automation = compile_automation(cache(action("input", {"text": "a"}, element(attributes={"name": "a"}))))
        assert Automation.model_validate_json(automation_to_json(automation)) == automation

    def test_uncompilable_actions_from_one_step_share_one_agentic_node(self):
        automation = compile_automation(
            cache(
                action("upload_file", {"path": "a.pdf"}, goal="Upload the resume"),
                action("evaluate", {"code": "1"}, goal="Upload the resume"),
            )
        )
        assert len(automation.nodes) == 1
        assert automation.nodes[0].interaction_action.agentic_task.task == "Upload the resume"
