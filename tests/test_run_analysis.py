import pandas as pd

from run_analysis import analyze


class FakeClient:
    def __init__(self):
        self.calls = []

    def analyze_one(self, text):
        self.calls.append(text)
        return {
            "opinion": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
        }


def test_analyze_weibo_end_to_end(tmp_path):
    day1 = pd.DataFrame({
        "weibo_id": ["w1"],
        "user_id": ["u1"],
        "weibo_content": ["AI is great"],
        "zan": [5],
    })
    day1.to_parquet(tmp_path / "2024-03-01.parquet", index=False)
    output_path = tmp_path / "out" / "weibo_opinion_results.csv"
    client = FakeClient()

    summary = analyze(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        output_path=str(output_path),
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        workers=2,
        client=client,
    )

    assert summary["completed"] == 1
    assert client.calls == ["AI is great"]
    assert output_path.exists()


def test_analyze_returns_zero_summary_when_no_input(tmp_path):
    output_path = tmp_path / "out" / "twitter_opinion_results.csv"
    client = FakeClient()

    summary = analyze(
        platform="twitter",
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        output_path=str(output_path),
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        client=client,
    )

    assert summary == {"total": 0, "skipped": 0, "completed": 0, "failed": 0}
    assert client.calls == []


def test_analyze_rejects_unknown_platform(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        analyze(
            platform="reddit",
            input_dir=str(tmp_path),
            filename_pattern="{date}.parquet",
            output_path=str(tmp_path / "out.csv"),
            api_key="k",
            base_url="https://openrouter.ai/api/v1",
            model="m",
            start_date="2024-03-01",
            end_date="2024-03-05",
            target_days=[1, 10, 20],
        )


# --- Final-review fix wave ---

import sys
import types

import pytest


def _write_weibo_day(tmp_path):
    pd.DataFrame({
        "weibo_id": ["w1"],
        "user_id": ["u1"],
        "weibo_content": ["AI is great"],
        "zan": [5],
    }).to_parquet(tmp_path / "2024-03-01.parquet", index=False)


def test_analyze_passes_timeout_and_backoff_to_constructed_client(tmp_path, monkeypatch):
    # (C) when no client is injected, analyze() must build OpenRouterClient with
    # the given timeout and backoff instead of silently using SDK defaults.
    import run_analysis

    _write_weibo_day(tmp_path)
    constructed = []

    class RecordingClient(FakeClient):
        def __init__(self, **kwargs):
            super().__init__()
            constructed.append(kwargs)

    monkeypatch.setattr(run_analysis, "OpenRouterClient", RecordingClient)

    analyze(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        output_path=str(tmp_path / "out.csv"),
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        model="m",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        timeout=45,
        backoff_base_seconds=2.0,
    )

    assert constructed[0]["timeout"] == 45
    assert constructed[0]["backoff_base_seconds"] == 2.0


def _install_fake_config(monkeypatch, tmp_path):
    fake_config = types.ModuleType("config")
    fake_config.WEIBO_INPUT_DIR = str(tmp_path / "weibo")
    fake_config.WEIBO_FILENAME_PATTERN = "{date}.parquet"
    fake_config.TWITTER_INPUT_DIR = str(tmp_path / "twitter")
    fake_config.TWITTER_FILENAME_PATTERN = "tweets_{date}.parquet"
    fake_config.OUTPUT_DIR = str(tmp_path / "output")
    fake_config.START_DATE = "2024-03-01"
    fake_config.END_DATE = "2025-03-20"
    fake_config.TARGET_DAYS = [1, 10, 20]
    fake_config.OPENROUTER_API_KEY = "k"
    fake_config.OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    fake_config.OPENROUTER_MODEL = "m"
    fake_config.MAX_WORKERS = 8
    fake_config.MAX_RETRIES = 3
    fake_config.REQUEST_TIMEOUT = 45
    fake_config.BACKOFF_BASE_SECONDS = 2.5
    monkeypatch.setitem(sys.modules, "config", fake_config)
    return fake_config


@pytest.fixture
def captured_analyze(monkeypatch, tmp_path):
    import run_analysis

    _install_fake_config(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(run_analysis, "analyze", lambda **kw: calls.append(kw))
    return calls


@pytest.mark.parametrize(
    "target_days, expected",
    [
        ("1,10,20", [1, 10, 20]),
        (10, [10]),
        ((1, 10, 20), [1, 10, 20]),
        ([1, 20], [1, 20]),
        (None, [1, 10, 20]),
    ],
)
def test_main_accepts_target_days_in_every_shape(captured_analyze, target_days, expected):
    # (E) Fire turns `--target_days 1,10,20` into a tuple, `--target_days 10`
    # into an int; main() must normalize all of them to List[int].
    from run_analysis import main

    main("weibo", target_days=target_days)
    assert captured_analyze[-1]["target_days"] == expected


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["weibo", "--target_days", "1,10,20"], [1, 10, 20]),
        (["weibo", "--target_days", "10"], [10]),
        (["weibo", "--target_days", "[1,20]"], [1, 20]),
        (["weibo", "--target_days", '"1,10"'], [1, 10]),
    ],
)
def test_main_target_days_through_real_fire_parsing(captured_analyze, argv, expected):
    import fire

    from run_analysis import main

    fire.Fire(main, command=argv)
    assert captured_analyze[-1]["target_days"] == expected


def test_main_passes_request_timeout_and_backoff_from_config(captured_analyze):
    # (C) main() wires config.REQUEST_TIMEOUT / BACKOFF_BASE_SECONDS through
    from run_analysis import main

    main("twitter")
    assert captured_analyze[-1]["timeout"] == 45
    assert captured_analyze[-1]["backoff_base_seconds"] == 2.5
