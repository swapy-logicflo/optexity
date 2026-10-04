from browser_use.agent.action_cache import ActionCache, CachedAction, CachedElement

from optexity.learning.loop import RunResult, learn, learn_from_run
from optexity.schema.actions.interaction_action import AgenticTask, InputTextAction, InteractionAction
from optexity.schema.automation import ActionNode, Automation, Parameters

URL = "https://www.roboform.com/filling-test-all-fields"


def field(name):
    return CachedElement(tag="input", attributes={"name": name}, x_path=f"html/{name}", element_hash=hash(name))


def fill(name, text, outcome="Verdict: Success"):
    return CachedAction(
        step_number=1, name="input", params={"text": text, "clear": True}, element=field(name),
        url=URL, goal="fill", outcome=outcome, error=None, is_done=False,
    )


def done(success=True):
    return CachedAction(
        step_number=2, name="done", params={"text": "ok", "success": success}, element=None,
        url=URL, goal="finish", error=None, is_done=True,
    )


def agent_cache(*actions):
    return ActionCache(task="fill the form", start_url=URL, actions=list(actions), llm_calls=3, duration_seconds=10, usage=None)


def node(interaction):
    return ActionNode(type="action_node", interaction_action=interaction)


def automation(*nodes):
    return Automation(url=URL, parameters=Parameters(input_parameters={}, generated_parameters={}), nodes=list(nodes))


AGENTIC = node(InteractionAction(agentic_task=AgenticTask(task="fill the form", max_steps=15)))
STALE_INPUT = node(
    InteractionAction(input_text=InputTextAction(command="locator(\"input[name='old']\").first", prompt_instructions="Enter full name", input_text="myname"))
)


class TestLearnFromRun:
    def test_agentic_node_is_replaced_by_its_compiled_steps(self):
        run = RunResult(task_id="t", status="success", caches={0: agent_cache(fill("04fullname", "myname"), fill("13adr_city", "SF"), done())})
        learned = learn_from_run(automation(AGENTIC), run, prune=None)
        commands = [n.interaction_action.input_text.command for n in learned.nodes]
        assert commands == ["locator(\"input[name='04fullname']\").first", "locator(\"input[name='13adr_city']\").first"]

    def test_splices_keep_surrounding_nodes_in_place(self):
        run = RunResult(task_id="t", status="success", caches={1: agent_cache(fill("10address1", "xyz"), done())})
        learned = learn_from_run(automation(STALE_INPUT, AGENTIC, STALE_INPUT), run, prune=None)
        assert [bool(n.interaction_action.input_text) for n in learned.nodes] == [True, True, True]
        assert learned.nodes[1].interaction_action.input_text.input_text == "xyz"

    def test_predicted_element_becomes_the_new_locator(self):
        run = RunResult(task_id="t", status="success", predicted={0: field("04fullname")})
        learned = learn_from_run(automation(STALE_INPUT), run, prune=None)
        action = learned.nodes[0].interaction_action.input_text
        assert action.command == "locator(\"input[name='04fullname']\").first"
        assert (action.input_text, action.prompt_instructions) == ("myname", "Enter full name")

    def test_does_not_learn_from_an_agent_that_did_not_succeed(self):
        run = RunResult(task_id="t", status="failed", caches={0: agent_cache(fill("04fullname", "myname"), done(success=False))})
        assert learn_from_run(automation(AGENTIC), run, prune=None) is None

    def test_returns_none_when_nothing_was_learned(self):
        assert learn_from_run(automation(STALE_INPUT), RunResult(task_id="t", status="success"), prune=None) is None


class TestLearn:
    def test_iterates_until_a_run_needs_no_llm(self):
        def run(current):
            if current.nodes[0].interaction_action.agentic_task:
                return RunResult(task_id="t0", status="success", caches={0: agent_cache(fill("04fullname", "myname"), done())})
            return RunResult(task_id="t1", status="success")

        final, results = learn(automation(AGENTIC), run, max_iterations=4)
        assert [r.llm_calls for r in results] == [3, 0]
        assert final.nodes[0].interaction_action.input_text.input_text == "myname"

    def test_stops_when_a_run_teaches_nothing(self):
        results = learn(automation(STALE_INPUT), lambda _: RunResult(task_id="t", status="failed"), max_iterations=4)[1]
        assert len(results) == 1
