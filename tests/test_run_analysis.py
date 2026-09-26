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
