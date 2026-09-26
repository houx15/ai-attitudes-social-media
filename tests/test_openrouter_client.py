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
