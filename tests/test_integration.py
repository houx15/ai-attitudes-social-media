# tests/test_integration.py
import pandas as pd
import pytest

from plot_figures import main as plot_main
from prepare_data import clean, export
from run_analysis import analyze


class ScriptedClient:
    """Deterministic opinion by post text, so the aggregate math is checkable."""

    def __init__(self, opinion_by_text):
        self._opinion_by_text = opinion_by_text

    def analyze_one(self, text):
        return {
            "opinion": self._opinion_by_text[text],
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
        }


def test_full_pipeline_weibo_and_twitter_to_plots(tmp_path):
    # --- Weibo input: two posts on 2024-03-01 ---
    weibo_dir = tmp_path / "weibo_input"
    weibo_dir.mkdir()
    pd.DataFrame({
        "weibo_id": ["w1", "w2"],
        "user_id": ["u1", "u2"],
        "weibo_content": ["weibo positive", "weibo negative"],
        "zan": [0, 0],
    }).to_parquet(weibo_dir / "2024-03-01.parquet", index=False)

    # --- Twitter input: two posts on 2024-03-01 ---
    twitter_dir = tmp_path / "twitter_input"
    twitter_dir.mkdir()
    pd.DataFrame({
        "id": ["t1", "t2"],
        "text": ["tweet positive", "tweet negative"],
        "likeCount": [0, 0],
        "author.id": ["a1", "a2"],
        "createdAt": [
            "Fri Mar 01 12:00:00 +0000 2024",
            "Fri Mar 01 13:00:00 +0000 2024",
        ],
    }).to_parquet(twitter_dir / "tweets_2024-03-01.parquet", index=False)

    output_dir = tmp_path / "output"

    # Stage 1
    weibo_results_path = output_dir / "analysis_results" / "weibo_opinion_results"
    analyze(
        platform="weibo",
        input_dir=str(weibo_dir),
        filename_pattern="{date}.parquet",
        results_dir=str(weibo_results_path),
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m",
        start_date="2024-03-01", end_date="2024-03-05", target_days=[1, 10, 20],
        client=ScriptedClient({"weibo positive": 2, "weibo negative": -2}),
    )

    twitter_results_path = output_dir / "analysis_results" / "twitter_opinion_results"
    analyze(
        platform="twitter",
        input_dir=str(twitter_dir),
        filename_pattern="tweets_{date}.parquet",
        results_dir=str(twitter_results_path),
        api_key="k", base_url="https://openrouter.ai/api/v1", model="m",
        start_date="2024-03-01", end_date="2024-03-05", target_days=[1, 10, 20],
        client=ScriptedClient({"tweet positive": 1, "tweet negative": -1}),
    )

    # Stage 2
    weibo_daily_path = output_dir / "weibo_daily_opinion.parquet"
    clean(
        platform="weibo", input_dir=str(weibo_dir), filename_pattern="{date}.parquet",
        start_date="2024-03-01", end_date="2024-03-05", target_days=[1, 10, 20],
        opinion_results_dir=str(weibo_results_path), output_path=str(weibo_daily_path),
    )

    twitter_daily_path = output_dir / "twitter_daily_opinion.parquet"
    clean(
        platform="twitter", input_dir=str(twitter_dir), filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01", end_date="2024-03-05", target_days=[1, 10, 20],
        opinion_results_dir=str(twitter_results_path), output_path=str(twitter_daily_path),
    )

    figure_data_path = output_dir / "figure_data.parquet"
    figure_data = export(str(weibo_daily_path), str(twitter_daily_path), str(figure_data_path))

    row = figure_data[figure_data["date"] == "2024-03-01"].iloc[0]
    assert row["weibo_avg_opinion"] == pytest.approx(0.0)  # mean(2, -2)
    assert row["twitter_avg_opinion"] == pytest.approx(0.0)  # mean(1, -1)

    # Stage 3
    figures_dir = output_dir / "figures"
    plot_main(figure_data_path=str(figure_data_path), output_dir=str(figures_dir))

    for metric_name in ["avg_opinion", "weighted_opinion", "user_avg_opinion"]:
        assert len(list(figures_dir.glob(f"{metric_name}_comparison_smoothed3d_*.pdf"))) == 1
        assert len(list(figures_dir.glob(f"{metric_name}_comparison_smoothed3d_*.csv"))) == 1
