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

    def _create(self, model, messages, **kwargs):
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

        def _create(self, model, messages, **kwargs):
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


class ExceptionRaisingClient:
    """A client that raises an exception for specific texts."""

    def __init__(self, raise_on_text=None):
        self.raise_on_text = raise_on_text or set()
        self.calls = []

    def analyze_one(self, text):
        self.calls.append(text)
        if text in self.raise_on_text:
            raise RuntimeError(f"Injected exception for {text}")
        return {
            "opinion": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
        }


def test_analyze_many_guards_against_exception_from_client(tmp_path):
    """Verify that exceptions from client.analyze_one don't crash analyze_many."""
    df = pd.DataFrame(
        {"id": ["a", "b", "c"], "text": ["text a", "text b", "text c"]}
    )
    # Client raises exception only for "text b"
    client = ExceptionRaisingClient(raise_on_text={"text b"})
    results_path = tmp_path / "results.csv"

    summary = analyze_many(client, df, str(results_path), max_workers=2)

    # All 3 rows should have been attempted (all 3 texts should be in calls)
    assert set(client.calls) == {"text a", "text b", "text c"}
    # Summary should show 2 completed, 1 failed (not crashed)
    assert summary["total"] == 3
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    # All 3 rows should be written to CSV: a with opinion 1, b with NaN, c with opinion 1
    saved = pd.read_csv(results_path)
    assert len(saved) == 3
    assert set(saved["id"]) == {"a", "b", "c"}
    # Row b should have NaN opinion (from the exception)
    b_row = saved[saved["id"] == "b"]
    assert b_row["opinion"].isna().all()


# --- Final-review fix wave ---

import pytest

from openrouter_client import load_processed_ids


def test_analyze_many_retries_row_with_empty_opinion_on_resume(tmp_path):
    # (B) a row that failed on a previous run must be re-sent, not skipped
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "id,opinion,prompt_tokens,completion_tokens,cached_tokens\n"
        "a,,0,0,0\n"
        "b,1,1,1,0\n"
    )
    df = pd.DataFrame({"id": ["a", "b"], "text": ["text a", "text b"]})
    client = CountingFakeClient()

    summary = analyze_many(client, df, str(results_path), max_workers=1)

    assert client.calls == ["text a"]
    assert summary["skipped"] == 1
    assert summary["completed"] == 1
    assert load_processed_ids(str(results_path)) == {"a", "b"}


def test_load_processed_ids_only_counts_valid_opinions(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "id,opinion,prompt_tokens,completion_tokens,cached_tokens\n"
        "a,,0,0,0\n"
        "b,2,1,1,0\n"
        "c,cannot tell,1,1,0\n"
        "d,-1.0,1,1,0\n"
        "e,garbage,1,1,0\n"
        "f,7,1,1,0\n"
    )
    assert load_processed_ids(str(results_path)) == {"b", "c", "d"}


def test_backoff_base_seconds_is_configurable(monkeypatch):
    # (C) backoff is (attempt + 1) * backoff_base_seconds
    import openrouter_client

    sleeps = []
    monkeypatch.setattr(openrouter_client.time, "sleep", lambda s: sleeps.append(s))

    class AlwaysFails:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model, messages, **kwargs):
            raise ConnectionError("always fails")

    client = OpenRouterClient(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        max_retries=3,
        backoff_base_seconds=2.0,
        client=AlwaysFails(),
    )
    client.analyze_one("x")
    assert sleeps == [2.0, 4.0, 6.0]


def test_backoff_default_stays_small(monkeypatch):
    import openrouter_client

    sleeps = []
    monkeypatch.setattr(openrouter_client.time, "sleep", lambda s: sleeps.append(s))
    fake = FakeOpenAI(responses=[_fake_response('{"opinion": 2}')], error_then_success=True)
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )
    client.analyze_one("x")
    assert sleeps == [pytest.approx(0.01)]


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"opinion": 2}', 2),
        ('{"opinion": 2.0}', 2),
        ('{"opinion": "2"}', 2),
        ('{"opinion": "-1"}', -1),
        ('{"opinion": 0}', 0),
        ('{"opinion": "cannot tell"}', "cannot tell"),
        ('{"opinion": [1, 2]}', None),
        ('{"opinion": 99}', None),
        ('{"opinion": 1.5}', None),
        ('{"opinion": "positive"}', None),
        ('{"opinion": {"x": 1}}', None),
        ('{"opinion": true}', None),
        ('{"opinion": null}', None),
        ('{"other": 1}', None),
    ],
)
def test_analyze_one_validates_opinion(content, expected):
    # (D) only -2..2 or "cannot tell" survive; anything else is a parse failure
    fake = FakeOpenAI(responses=[_fake_response(content)])
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )
    assert client.analyze_one("x")["opinion"] == expected


def test_malformed_model_output_does_not_corrupt_results_csv(tmp_path):
    # (D) end-to-end: a real OpenRouterClient fed an array / out-of-range opinion
    # must not produce a broken CSV, and later reads must still work.
    from prepare_data import clean

    class ByTextOpenAI:
        def __init__(self, content_by_text):
            self._content_by_text = content_by_text
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model, messages, **kwargs):
            user_text = messages[-1]["content"]
            for key, content in self._content_by_text.items():
                if key in user_text:
                    return _fake_response(content)
            raise AssertionError(user_text)

    fake = ByTextOpenAI({
        "post-a": '{"opinion": 1}',
        "post-b": '{"opinion": [1, 2]}',
        "post-c": '{"opinion": 1000}',
        "post-d": '{"opinion": "a,\\"quoted\\"\\nvalue"}',
    })
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )

    input_dir = tmp_path / "weibo"
    input_dir.mkdir()
    pd.DataFrame({
        "weibo_id": ["a", "b", "c", "d"],
        "user_id": ["u1", "u2", "u3", "u4"],
        "weibo_content": ["post-a", "post-b", "post-c", "post-d"],
        "zan": [0, 0, 0, 0],
    }).to_parquet(input_dir / "2024-03-01.parquet", index=False)

    results_path = tmp_path / "results.csv"
    df = pd.DataFrame({"id": ["a", "b", "c", "d"], "text": ["post-a", "post-b", "post-c", "post-d"]})
    summary = analyze_many(client, df, str(results_path), max_workers=2)
    assert summary["completed"] == 1
    assert summary["failed"] == 3

    saved = pd.read_csv(results_path, dtype=str)
    assert list(saved.columns) == ["id", "opinion", "prompt_tokens", "completion_tokens", "cached_tokens"]
    assert len(saved) == 4
    assert load_processed_ids(str(results_path)) == {"a"}

    result = clean(
        platform="weibo",
        input_dir=str(input_dir),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(results_path),
        output_path=str(tmp_path / "daily.parquet"),
    )
    assert result.iloc[0]["avg_opinion"] == pytest.approx(1.0)


def test_analyze_many_csv_write_is_quoted_even_for_unvalidated_values(tmp_path):
    # (D) even if an injected client bypasses validation, csv.writer must keep
    # the file parseable (commas / quotes / newlines are quoted, not raw).
    df = pd.DataFrame({"id": ["a", "b"], "text": ["text a", "text b"]})
    client = CountingFakeClient(opinion_by_text={"text b": 'x,"y"\nz', "text a": [1, 2]})
    results_path = tmp_path / "results.csv"

    analyze_many(client, df, str(results_path), max_workers=1)

    saved = pd.read_csv(results_path, dtype=str)
    assert len(saved) == 2
    assert saved.set_index("id").loc["b", "opinion"] == 'x,"y"\nz'
    assert load_processed_ids(str(results_path)) == set()


def test_analyze_one_turns_reasoning_off():
    captured = {}

    class RecordingOpenAI:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model, messages, **kwargs):
            captured.update(kwargs)
            return _fake_response('{"opinion": 1}')

    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=RecordingOpenAI()
    )
    client.analyze_one("some text")

    assert captured["extra_body"] == {"reasoning": {"effort": "none"}}


def test_analyze_many_shows_progress_and_resume_counts(tmp_path, capsys):
    results_path = tmp_path / "results.csv"
    results_path.write_text("id,opinion,prompt_tokens,completion_tokens,cached_tokens\na,1,1,1,0\n")
    df = pd.DataFrame({"id": ["a", "b", "c"], "text": ["ta", "tb", "tc"]})

    analyze_many(CountingFakeClient(), df, str(results_path), max_workers=2, desc="weibo")

    captured = capsys.readouterr()
    assert "3 posts: 1 already done, 2 to analyze" in captured.out
    assert "weibo" in captured.err and "2/2" in captured.err


def test_analyze_many_reports_token_usage_live_and_at_the_end(tmp_path, capsys):
    results_path = tmp_path / "results.csv"
    results_path.write_text("id,opinion,prompt_tokens,completion_tokens,cached_tokens\na,1,400,6,100\n")
    df = pd.DataFrame({"id": ["a", "b", "c"], "text": ["ta", "tb", "tc"]})

    analyze_many(CountingFakeClient(), df, str(results_path), max_workers=2, desc="weibo")

    captured = capsys.readouterr()
    # CountingFakeClient reports 1 prompt + 1 completion token per call
    assert "weibo tokens this run: 2 prompt (0 cached) + 2 output" in captured.out
    assert "weibo tokens all runs: 402 prompt (100 cached) + 8 output" in captured.out
    assert "in=2" in captured.err and "out=2" in captured.err


def test_format_tokens_is_compact():
    from openrouter_client import format_tokens

    assert format_tokens(950) == "950"
    assert format_tokens(45_678) == "45.7k"
    assert format_tokens(1_234_567) == "1.23M"


def test_analyze_one_reads_cached_tokens_from_prompt_tokens_details():
    # OpenAI-compatible APIs (incl. OpenRouter) nest it: usage.prompt_tokens_details.cached_tokens
    usage = SimpleNamespace(
        prompt_tokens=400,
        completion_tokens=6,
        prompt_tokens_details=SimpleNamespace(cached_tokens=384),
    )
    fake = FakeOpenAI(responses=[_fake_response('{"opinion": 1}', usage=usage)])
    client = OpenRouterClient(
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m", client=fake
    )

    assert client.analyze_one("some text")["cached_tokens"] == 384
