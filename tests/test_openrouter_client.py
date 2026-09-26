from types import SimpleNamespace

from openrouter_client import OpenRouterClient


class FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=5, cached_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cached_tokens = cached_tokens


def _fake_response(content, usage=None):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage or FakeUsage())


class FakeOpenAI:
    """Stands in for the openai.OpenAI client, shaped like it for analyze_one."""

    def __init__(self, responses=None, error_then_success=None):
        self._responses = responses or []
        self._error_then_success = error_then_success
        self.call_count = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, model, messages):
        self.call_count += 1
        if self._error_then_success is not None and self.call_count == 1:
            raise ConnectionError("simulated transient failure")
        response = self._responses[min(self.call_count - 1, len(self._responses) - 1)]
        return response


def test_analyze_one_parses_valid_json_opinion():
    fake = FakeOpenAI(responses=[_fake_response('{"opinion": 1}')])
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )
    result = client.analyze_one("some text")
    assert result["opinion"] == 1
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 5


def test_analyze_one_parses_cannot_tell():
    fake = FakeOpenAI(responses=[_fake_response('{"opinion": "cannot tell"}')])
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )
    result = client.analyze_one("some text")
    assert result["opinion"] == "cannot tell"


def test_analyze_one_returns_none_opinion_on_malformed_json():
    fake = FakeOpenAI(responses=[_fake_response("not json at all")])
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )
    result = client.analyze_one("some text")
    assert result["opinion"] is None


def test_analyze_one_retries_then_succeeds():
    fake = FakeOpenAI(
        responses=[_fake_response('{"opinion": 2}')], error_then_success=True
    )
    client = OpenRouterClient(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        max_retries=2,
        client=fake,
    )
    result = client.analyze_one("some text")
    assert result["opinion"] == 2
    assert fake.call_count == 2


def test_analyze_one_gives_up_after_max_retries():
    class AlwaysFails:
        def __init__(self):
            self.call_count = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model, messages):
            self.call_count += 1
            raise ConnectionError("always fails")

    fake = AlwaysFails()
    client = OpenRouterClient(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        max_retries=2,
        client=fake,
    )
    result = client.analyze_one("some text")
    assert result["opinion"] is None
    assert fake.call_count == 3  # initial attempt + 2 retries


import pandas as pd

from openrouter_client import analyze_many


class CountingFakeClient:
    def __init__(self, opinion_by_text=None):
        self.calls = []
        self._opinion_by_text = opinion_by_text or {}

    def analyze_one(self, text):
        self.calls.append(text)
        opinion = self._opinion_by_text.get(text, 1)
        return {
            "opinion": opinion,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
        }


def test_analyze_many_writes_results_csv(tmp_path):
    df = pd.DataFrame({"id": ["a", "b"], "text": ["text a", "text b"]})
    results_path = tmp_path / "results.csv"
    client = CountingFakeClient()

    summary = analyze_many(client, df, str(results_path), max_workers=2)

    assert summary == {"total": 2, "skipped": 0, "completed": 2, "failed": 0}
    saved = pd.read_csv(results_path, dtype=str)
    assert set(saved["id"]) == {"a", "b"}
    assert set(saved["opinion"]) == {"1"}


def test_analyze_many_skips_already_processed_ids(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text("id,opinion,prompt_tokens,completion_tokens,cached_tokens\na,1,1,1,0\n")

    df = pd.DataFrame({"id": ["a", "b"], "text": ["text a", "text b"]})
    client = CountingFakeClient()

    summary = analyze_many(client, df, str(results_path), max_workers=2)

    assert summary["skipped"] == 1
    assert summary["completed"] == 1
    assert client.calls == ["text b"]
    saved = pd.read_csv(results_path, dtype=str)
    assert len(saved) == 2


def test_analyze_many_counts_failed_when_opinion_is_none(tmp_path):
    df = pd.DataFrame({"id": ["a"], "text": ["text a"]})
    client = CountingFakeClient(opinion_by_text={"text a": None})
    results_path = tmp_path / "results.csv"

    summary = analyze_many(client, df, str(results_path), max_workers=1)

    assert summary == {"total": 1, "skipped": 0, "completed": 0, "failed": 1}
    saved = pd.read_csv(results_path)
    assert saved["opinion"].isna().all()
