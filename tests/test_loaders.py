import pandas as pd

from loaders import iter_target_dates, weibo_loader, twitter_loader, STANDARD_COLUMNS


def test_iter_target_dates_single_month():
    dates = iter_target_dates("2024-03-01", "2024-03-31", [1, 10, 20])
    assert dates == ["2024-03-01", "2024-03-10", "2024-03-20"]


def test_iter_target_dates_spans_months():
    dates = iter_target_dates("2024-03-15", "2024-04-15", [1, 10, 20])
    assert dates == ["2024-03-20", "2024-04-01", "2024-04-10"]


def test_iter_target_dates_no_matches():
    dates = iter_target_dates("2024-03-02", "2024-03-09", [1, 10, 20])
    assert dates == []


def test_iter_target_dates_single_day_range():
    dates = iter_target_dates("2024-03-10", "2024-03-10", [1, 10, 20])
    assert dates == ["2024-03-10"]


def test_weibo_loader_reads_matching_dates_only(tmp_path):
    day1 = pd.DataFrame({
        "weibo_id": ["w1", "w2"],
        "user_id": ["u1", "u2"],
        "weibo_content": ["AI is great", "AI is scary"],
        "zan": [5, 0],
    })
    day1.to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    day_not_sampled = pd.DataFrame({
        "weibo_id": ["w3"],
        "user_id": ["u3"],
        "weibo_content": ["not sampled"],
        "zan": [1],
    })
    day_not_sampled.to_parquet(tmp_path / "2024-03-02.parquet", index=False)

    result = weibo_loader(
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )

    assert list(result.columns) == STANDARD_COLUMNS
    assert set(result["id"]) == {"w1", "w2"}
    assert result[result["id"] == "w1"]["text"].iloc[0] == "AI is great"
    assert result[result["id"] == "w1"]["weight_raw"].iloc[0] == 5
    assert result[result["id"] == "w1"]["user_id"].iloc[0] == "u1"
    assert result[result["id"] == "w1"]["date"].iloc[0] == "2024-03-01"


def test_weibo_loader_skips_missing_files(tmp_path):
    result = weibo_loader(
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )
    assert list(result.columns) == STANDARD_COLUMNS
    assert len(result) == 0


def test_twitter_loader_reads_matching_dates_and_parses_created_at(tmp_path):
    day1 = pd.DataFrame({
        "id": ["t1", "t2"],
        "text": ["AI is great", "AI is scary"],
        "likeCount": [10, 0],
        "author.id": ["a1", "a2"],
        "createdAt": [
            "Fri Mar 01 12:00:00 +0000 2024",
            "Fri Mar 01 23:59:00 +0000 2024",
        ],
    })
    day1.to_parquet(tmp_path / "tweets_2024-03-01.parquet", index=False)

    result = twitter_loader(
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )

    assert list(result.columns) == STANDARD_COLUMNS
    assert set(result["id"]) == {"t1", "t2"}
    row = result[result["id"] == "t1"].iloc[0]
    assert row["text"] == "AI is great"
    assert row["weight_raw"] == 10
    assert row["user_id"] == "a1"
    assert row["date"] == "2024-03-01"


def test_twitter_loader_skips_missing_files(tmp_path):
    result = twitter_loader(
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )
    assert list(result.columns) == STANDARD_COLUMNS
    assert len(result) == 0
