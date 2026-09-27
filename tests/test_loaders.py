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


# --- Final-review fix wave: dedup on id (A) and empty/null text (J) ---


def test_weibo_loader_drops_duplicate_ids(tmp_path):
    pd.DataFrame({
        "weibo_id": ["w1", "w1", "w2"],
        "user_id": ["u1", "u1", "u2"],
        "weibo_content": ["AI is great", "AI is great", "AI is scary"],
        "zan": [0, 0, 0],
    }).to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    result = weibo_loader(
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )

    assert len(result) == 2
    assert sorted(result["id"]) == ["w1", "w2"]


def test_twitter_loader_drops_duplicate_ids(tmp_path):
    pd.DataFrame({
        "id": ["t1", "t1", "t2"],
        "text": ["AI is great", "AI is great", "AI is scary"],
        "likeCount": [0, 0, 0],
        "author.id": ["a1", "a1", "a2"],
        "createdAt": ["Fri Mar 01 12:00:00 +0000 2024"] * 3,
    }).to_parquet(tmp_path / "tweets_2024-03-01.parquet", index=False)

    result = twitter_loader(
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )

    assert len(result) == 2
    assert sorted(result["id"]) == ["t1", "t2"]


def test_weibo_loader_keeps_null_and_empty_text_like_legacy(tmp_path):
    # Legacy ai_sentiment_analyzer.py sent every Weibo post, using
    # str(weibo_content or "") as the text.
    pd.DataFrame({
        "weibo_id": ["w1", "w2", "w3", "w4"],
        "user_id": ["u1", "u2", "u3", "u4"],
        "weibo_content": ["AI is great", None, "", "   "],
        "zan": [0, 0, 0, 0],
    }).to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    result = weibo_loader(
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )

    assert list(result["id"]) == ["w1", "w2", "w3", "w4"]
    assert list(result["text"]) == ["AI is great", "", "", "   "]


def test_twitter_loader_drops_null_and_empty_text_like_legacy(tmp_path):
    # Legacy batch_sentiment_analysis.py skipped a tweet only when `text or ""`
    # was falsy, so whitespace-only text was still sent.
    pd.DataFrame({
        "id": ["t1", "t2", "t3", "t4"],
        "text": ["AI is great", None, "", "   "],
        "likeCount": [0, 0, 0, 0],
        "author.id": ["a1", "a2", "a3", "a4"],
        "createdAt": ["Fri Mar 01 12:00:00 +0000 2024"] * 4,
    }).to_parquet(tmp_path / "tweets_2024-03-01.parquet", index=False)

    result = twitter_loader(
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
    )

    assert list(result["id"]) == ["t1", "t4"]


def test_parse_target_days_accepts_str_int_and_sequence():
    from loaders import parse_target_days

    assert parse_target_days("1,10,20") == [1, 10, 20]
    assert parse_target_days("1, 10 ,20") == [1, 10, 20]
    assert parse_target_days(10) == [10]
    assert parse_target_days((1, 10, 20)) == [1, 10, 20]
    assert parse_target_days([1, 10, 20]) == [1, 10, 20]


def test_iter_target_dates_applies_substitutions():
    dates = iter_target_dates(
        "2024-03-01", "2024-03-31", [1, 10, 20], substitutions={"2024-03-01": "2024-02-29"}
    )
    # The substitute stands in for its nominal date even though it falls before
    # start_date.
    assert dates == ["2024-02-29", "2024-03-10", "2024-03-20"]


def test_iter_target_dates_ignores_substitutions_outside_range():
    dates = iter_target_dates(
        "2024-03-05", "2024-03-31", [1, 10, 20], substitutions={"2024-03-01": "2024-02-29"}
    )
    assert dates == ["2024-03-10", "2024-03-20"]


def test_weibo_loader_reads_substituted_day_under_its_actual_date(tmp_path):
    pd.DataFrame({
        "weibo_id": ["w1"], "user_id": ["u1"], "weibo_content": ["AI"], "zan": [0],
    }).to_parquet(tmp_path / "2024-02-29.parquet", index=False)

    result = weibo_loader(
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        substitutions={"2024-03-01": "2024-02-29"},
    )

    assert list(result["id"]) == ["w1"]
    assert list(result["date"]) == ["2024-02-29"]


def test_twitter_loader_reads_substituted_day(tmp_path):
    pd.DataFrame({
        "id": ["t1"], "text": ["AI"], "likeCount": [0], "author.id": ["a1"],
        "createdAt": ["Wed Oct 02 12:00:00 +0000 2024"],
    }).to_parquet(tmp_path / "tweets_2024-10-02.parquet", index=False)

    result = twitter_loader(
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        start_date="2024-10-01",
        end_date="2024-10-05",
        target_days=[1, 10, 20],
        substitutions={"2024-10-01": "2024-10-02"},
    )

    assert list(result["date"]) == ["2024-10-02"]


def test_loader_reports_target_dates_with_no_input_file(tmp_path, capsys):
    pd.DataFrame({
        "weibo_id": ["w1"], "user_id": ["u1"], "weibo_content": ["AI"], "zan": [0],
    }).to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    weibo_loader(
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-20",
        target_days=[1, 10, 20],
    )

    out = capsys.readouterr().out
    assert "2024-03-10" in out and "2024-03-20" in out
    assert "2024-03-01" not in out


def _write_weibo_day(tmp_path, date, ids):
    pd.DataFrame({
        "weibo_id": ids, "user_id": [1] * len(ids),
        "weibo_content": [f"post {i}" for i in ids], "zan": [0] * len(ids),
    }).to_parquet(tmp_path / f"{date}.parquet", index=False)


def test_iter_platform_days_yields_one_day_at_a_time_and_dedups_across_days(tmp_path):
    from loaders import day_files, iter_platform_days

    _write_weibo_day(tmp_path, "2024-03-01", ["w1", "w2", "w2"])
    _write_weibo_day(tmp_path, "2024-03-10", ["w2", "w3"])
    files = day_files("weibo", str(tmp_path), "{date}.parquet", "2024-03-01", "2024-03-10", [1, 10, 20])

    frames = list(iter_platform_days("weibo", files))

    assert [list(f["id"]) for f in frames] == [["w1", "w2"], ["w3"]]
    assert [f.attrs["raw_rows"] for f in frames] == [3, 2]


def test_iter_platform_days_can_skip_the_text_column(tmp_path):
    from loaders import day_files, iter_platform_days

    # No weibo_content column at all: reading it would fail.
    pd.DataFrame({"weibo_id": ["w1"], "user_id": [1], "zan": [0]}).to_parquet(
        tmp_path / "2024-03-01.parquet", index=False
    )
    files = day_files("weibo", str(tmp_path), "{date}.parquet", "2024-03-01", "2024-03-05", [1, 10, 20])

    (frame,) = iter_platform_days("weibo", files, with_text=False)

    assert "text" not in frame.columns
    assert list(frame["id"]) == ["w1"]


def test_count_input_rows_reads_only_file_metadata(tmp_path):
    from loaders import count_input_rows, day_files

    _write_weibo_day(tmp_path, "2024-03-01", ["w1", "w2", "w2"])
    _write_weibo_day(tmp_path, "2024-03-10", ["w3"])
    files = day_files("weibo", str(tmp_path), "{date}.parquet", "2024-03-01", "2024-03-10", [1, 10, 20])

    assert count_input_rows(files) == 4
