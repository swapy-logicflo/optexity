from browser_use.agent.action_cache import CachedAction, CachedElement

from optexity.learning import pruner
from optexity.learning.pruner import PruneDecision, build_prompt, invalid_ids, llm_prune
from optexity.schema.token_usage import TokenUsage

URL = "https://the-internet.herokuapp.com/login"


def action(name, params=None, el=None):
    return CachedAction(
        step_number=1, name=name, params=params or {}, element=el, url=URL, goal="g", error=None, is_done=False
    )


def links(n):
    return [
        action("click", el=CachedElement(tag="a", attributes={}, x_path=f"html/a[{i}]", element_hash=i, ax_role="link", ax_name=f"Link {i}"))
        for i in range(n)
    ]


class FakeModel:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def get_model_response_with_structured_output(self, prompt, response_schema, system_instruction):
        self.prompts.append(prompt)
        return PruneDecision(keep=self.answers.pop(0), reasoning="r"), TokenUsage()


def use_model(monkeypatch, model):
    monkeypatch.setattr(pruner, "get_llm_model_with_fallback", lambda *args: model)


class TestPrompt:
    def test_typed_values_are_not_sent_to_the_llm(self):
        password = action(
            "input",
            {"text": "SuperSecretPassword!", "clear": True},
            CachedElement(tag="input", attributes={"name": "password"}, x_path="html/input", element_hash=1),
        )
        prompt = build_prompt("log in", [password])
        assert "SuperSecretPassword!" not in prompt
        assert '"clear": true' in prompt


class TestInvalidIds:
    def test_accepts_known_ids(self):
        assert invalid_ids([0, 2], 3) is None

    def test_rejects_empty_and_unknown_ids(self):
        assert invalid_ids([], 3) == "keep at least one action"
        assert "[3, 7]" in invalid_ids([0, 3, 7], 3)


class TestLlmPrune:
    def test_keeps_selected_actions_in_original_order(self, monkeypatch):
        candidates = links(4)
        use_model(monkeypatch, FakeModel([3, 0, 0]))
        assert llm_prune("task", candidates) == [candidates[0], candidates[3]]

    def test_retries_with_feedback_after_invalid_ids(self, monkeypatch):
        candidates = links(2)
        model = FakeModel([5], [1])
        use_model(monkeypatch, model)
        assert llm_prune("task", candidates) == [candidates[1]]
        assert "ids [5] do not exist" in model.prompts[1]

    def test_falls_back_to_all_actions_when_the_llm_keeps_failing(self, monkeypatch):
        candidates = links(2)
        use_model(monkeypatch, FakeModel([9], [9]))
        assert llm_prune("task", candidates) == candidates
