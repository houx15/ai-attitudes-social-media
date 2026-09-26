import pandas as pd
import pytest

from prepare_data import clean


def test_clean_computes_three_daily_metrics(tmp_path):
    day1 = pd.DataFrame({
        "weibo_id": ["w1", "w2", "w3"],
        "user_id": ["u1", "u1", "u2"],
        "weibo_content": ["a", "b", "c"],
        "zan": [0, 4, 9],
    })
    day1.to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    opinions = pd.DataFrame({
        "id": ["w1", "w2", "w3"],
        "opinion": [2, -2, 1],
        "prompt_tokens": [1, 1, 1],
        "completion_tokens": [1, 1, 1],
        "cached_tokens": [0, 0, 0],
    })
    opinion_results_path = tmp_path / "weibo_opinion_results.csv"
    opinions.to_csv(opinion_results_path, index=False)

    output_path = tmp_path / "weibo_daily_opinion.parquet"

    result = clean(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(opinion_results_path),
        output_path=str(output_path),
    )

    assert list(result.columns) == ["date", "avg_opinion", "weighted_opinion", "user_avg_opinion"]
    row = result[result["date"] == "2024-03-01"].iloc[0]

    # avg_opinion: raw mean of [2, -2, 1] = 0.333...
    assert row["avg_opinion"] == pytest.approx((2 - 2 + 1) / 3)

    # weighted_opinion: weights are zan+1 = [1, 5, 10]
    # weighted mean = (2*1 + -2*5 + 1*10) / (1+5+10) = 2/16
    assert row["weighted_opinion"] == pytest.approx((2 * 1 + -2 * 5 + 1 * 10) / 16)

    # user_avg_opinion: u1 mean = (2 + -2)/2 = 0, u2 mean = 1; mean of [0, 1] = 0.5
    assert row["user_avg_opinion"] == pytest.approx(0.5)

    assert output_path.exists()
    saved = pd.read_parquet(output_path)
    assert len(saved) == 1


def test_clean_drops_cannot_tell_and_missing_opinions(tmp_path):
    day1 = pd.DataFrame({
        "weibo_id": ["w1", "w2", "w3"],
        "user_id": ["u1", "u1", "u2"],
        "weibo_content": ["a", "b", "c"],
        "zan": [0, 0, 0],
    })
    day1.to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    opinions = pd.DataFrame({
        "id": ["w1", "w2", "w3"],
        "opinion": [2, "cannot tell", ""],
        "prompt_tokens": [1, 1, 1],
        "completion_tokens": [1, 1, 1],
        "cached_tokens": [0, 0, 0],
    })
    opinion_results_path = tmp_path / "weibo_opinion_results.csv"
    opinions.to_csv(opinion_results_path, index=False)
    output_path = tmp_path / "weibo_daily_opinion.parquet"

    result = clean(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(opinion_results_path),
        output_path=str(output_path),
    )

    row = result[result["date"] == "2024-03-01"].iloc[0]
    # only w1 (opinion=2) survives the cannot-tell/missing filter
    assert row["avg_opinion"] == pytest.approx(2)
