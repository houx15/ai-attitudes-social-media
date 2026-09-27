import pandas as pd
import pytest

from prepare_data import clean, export


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


def test_export_merges_both_platforms_unsmoothed(tmp_path):
    weibo_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-10"],
        "avg_opinion": [0.5, 1.0],
        "weighted_opinion": [0.4, 0.9],
        "user_avg_opinion": [0.3, 0.8],
    })
    weibo_path = tmp_path / "weibo_daily_opinion.parquet"
    weibo_df.to_parquet(weibo_path, index=False)

    twitter_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-20"],
        "avg_opinion": [-0.5, -1.0],
        "weighted_opinion": [-0.4, -0.9],
        "user_avg_opinion": [-0.3, -0.8],
    })
    twitter_path = tmp_path / "twitter_daily_opinion.parquet"
    twitter_df.to_parquet(twitter_path, index=False)

    output_path = tmp_path / "figure_data.parquet"

    result = export(str(weibo_path), str(twitter_path), str(output_path))

    expected_columns = [
        "date",
        "weibo_avg_opinion",
        "twitter_avg_opinion",
        "weibo_weighted_opinion",
        "twitter_weighted_opinion",
        "weibo_user_avg_opinion",
        "twitter_user_avg_opinion",
    ]
    assert list(result.columns) == expected_columns
    assert set(result["date"]) == {"2024-03-01", "2024-03-10", "2024-03-20"}

    row = result[result["date"] == "2024-03-01"].iloc[0]
    assert row["weibo_avg_opinion"] == pytest.approx(0.5)
    assert row["twitter_avg_opinion"] == pytest.approx(-0.5)

    # 2024-03-10 has no twitter data -> NaN, not dropped, not zero
    row_weibo_only = result[result["date"] == "2024-03-10"].iloc[0]
    assert row_weibo_only["weibo_avg_opinion"] == pytest.approx(1.0)
    assert pd.isna(row_weibo_only["twitter_avg_opinion"])

    assert output_path.exists()
    assert output_path.with_suffix(".csv").exists()


# --- Final-review fix wave ---


def _clean_weibo(tmp_path, opinion_results_path):
    return clean(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(opinion_results_path),
        output_path=str(tmp_path / "weibo_daily_opinion.parquet"),
    )


def test_clean_duplicate_ids_do_not_inflate_metrics(tmp_path):
    # Reproduction from the final review: [w1(op=2), w1(op=2), w2(op=-2)]
    # previously produced avg_opinion=1.2 instead of the true 0.0.
    pd.DataFrame({
        "weibo_id": ["w1", "w1", "w2"],
        "user_id": ["u1", "u1", "u2"],
        "weibo_content": ["a", "a", "b"],
        "zan": [0, 0, 0],
    }).to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    opinion_results_path = tmp_path / "weibo_opinion_results.csv"
    pd.DataFrame({
        "id": ["w1", "w1", "w2"],
        "opinion": [2, 2, -2],
        "prompt_tokens": [1, 1, 1],
        "completion_tokens": [1, 1, 1],
        "cached_tokens": [0, 0, 0],
    }).to_csv(opinion_results_path, index=False)

    result = _clean_weibo(tmp_path, opinion_results_path)

    row = result[result["date"] == "2024-03-01"].iloc[0]
    assert row["avg_opinion"] == pytest.approx(0.0)
    assert row["weighted_opinion"] == pytest.approx(0.0)
    assert row["user_avg_opinion"] == pytest.approx(0.0)


def test_clean_keeps_last_result_for_retried_id(tmp_path):
    # The same id written twice (e.g. an earlier result superseded by a later
    # re-run): the last write wins, and the id is counted exactly once.
    pd.DataFrame({
        "weibo_id": ["w1", "w2"],
        "user_id": ["u1", "u2"],
        "weibo_content": ["a", "b"],
        "zan": [0, 0],
    }).to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    opinion_results_path = tmp_path / "weibo_opinion_results.csv"
    opinion_results_path.write_text(
        "id,opinion,prompt_tokens,completion_tokens,cached_tokens\n"
        "w1,1,1,1,0\n"
        "w2,-2,1,1,0\n"
        "w1,2,1,1,0\n"
    )

    result = _clean_weibo(tmp_path, opinion_results_path)

    row = result[result["date"] == "2024-03-01"].iloc[0]
    # keep-last -> mean(2, -2) = 0; no dedup would give mean(1, -2, 2) = 1/3
    assert row["avg_opinion"] == pytest.approx(0.0)


def test_clean_prints_coverage_summary(tmp_path, capsys):
    pd.DataFrame({
        "weibo_id": ["w1", "w2", "w3", "w4"],
        "user_id": ["u1", "u2", "u3", "u4"],
        "weibo_content": ["a", "b", "c", "d"],
        "zan": [0, 0, 0, 0],
    }).to_parquet(tmp_path / "2024-03-01.parquet", index=False)

    # w4 has no result at all; w3 is cannot-tell -> 4 loaded, 3 matched, 2 valid
    opinion_results_path = tmp_path / "weibo_opinion_results.csv"
    pd.DataFrame({
        "id": ["w1", "w2", "w3"],
        "opinion": [2, -1, "cannot tell"],
        "prompt_tokens": [1, 1, 1],
        "completion_tokens": [1, 1, 1],
        "cached_tokens": [0, 0, 0],
    }).to_csv(opinion_results_path, index=False)

    _clean_weibo(tmp_path, opinion_results_path)

    out = capsys.readouterr().out
    assert "4 metadata rows loaded" in out
    assert "3 matched an opinion result" in out
    assert "2 with a valid numeric opinion" in out


def _install_fake_config(monkeypatch, tmp_path):
    import sys
    import types

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
    monkeypatch.setitem(sys.modules, "config", fake_config)
    return fake_config


def test_clean_cli_accepts_date_range_overrides(tmp_path, monkeypatch):
    import prepare_data

    _install_fake_config(monkeypatch, tmp_path)
    captured = []
    monkeypatch.setattr(prepare_data, "clean", lambda **kw: captured.append(kw))

    prepare_data._clean_cli(
        "weibo", start_date="2024-04-01", end_date="2024-04-30", target_days=(1, 10)
    )
    assert captured[-1]["start_date"] == "2024-04-01"
    assert captured[-1]["end_date"] == "2024-04-30"
    assert captured[-1]["target_days"] == [1, 10]

    prepare_data._clean_cli("weibo", target_days="1,20")
    assert captured[-1]["target_days"] == [1, 20]

    # no overrides -> falls back to config
    prepare_data._clean_cli("twitter")
    assert captured[-1]["start_date"] == "2024-03-01"
    assert captured[-1]["end_date"] == "2025-03-20"
    assert captured[-1]["target_days"] == [1, 10, 20]


def test_clean_cli_overrides_through_real_fire_parsing(tmp_path, monkeypatch):
    import fire

    import prepare_data

    _install_fake_config(monkeypatch, tmp_path)
    captured = []
    monkeypatch.setattr(prepare_data, "clean", lambda **kw: captured.append(kw))

    fire.Fire(
        {"clean": prepare_data._clean_cli},
        command=["clean", "--platform", "weibo", "--start_date", "2024-04-01",
                 "--end_date", "2024-04-30", "--target_days", "1,10,20"],
    )
    assert captured[-1]["start_date"] == "2024-04-01"
    assert captured[-1]["end_date"] == "2024-04-30"
    assert captured[-1]["target_days"] == [1, 10, 20]


def test_clean_cli_passes_each_platforms_date_substitutions_from_config(tmp_path, monkeypatch):
    import prepare_data

    fake_config = _install_fake_config(monkeypatch, tmp_path)
    fake_config.DATE_SUBSTITUTIONS = {"weibo": {"2024-03-01": "2024-02-29"}, "twitter": {}}
    captured = []
    monkeypatch.setattr(prepare_data, "clean", lambda **kw: captured.append(kw))

    prepare_data._clean_cli("weibo")
    prepare_data._clean_cli("twitter")

    assert captured[0]["date_substitutions"] == {"2024-03-01": "2024-02-29"}
    assert captured[1]["date_substitutions"] == {}


def test_clean_keeps_substituted_day_under_its_actual_date(tmp_path):
    pd.DataFrame({
        "weibo_id": ["w1", "w2"], "user_id": ["u1", "u2"],
        "weibo_content": ["a", "b"], "zan": [0, 0],
    }).to_parquet(tmp_path / "2024-02-29.parquet", index=False)
    results_path = tmp_path / "weibo_opinion_results.csv"
    pd.DataFrame({
        "id": ["w1", "w2"], "opinion": [2, 0],
        "prompt_tokens": [1, 1], "completion_tokens": [1, 1], "cached_tokens": [0, 0],
    }).to_csv(results_path, index=False)

    result = clean(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(results_path),
        output_path=str(tmp_path / "weibo_daily_opinion.parquet"),
        date_substitutions={"2024-03-01": "2024-02-29"},
    )

    assert result["date"].tolist() == ["2024-02-29"]
    assert result["avg_opinion"].tolist() == pytest.approx([1.0])


def test_clean_leaves_posts_with_missing_user_id_out_of_the_user_level_mean(tmp_path):
    # Legacy behaviour: groupby on the raw user_id drops missing users, so their
    # posts count toward avg/weighted opinion but not toward user_avg_opinion.
    pd.DataFrame({
        "weibo_id": ["w1", "w2"], "user_id": [1001, None],
        "weibo_content": ["a", "b"], "zan": [0, 0],
    }).to_parquet(tmp_path / "2024-03-01.parquet", index=False)
    results_path = tmp_path / "weibo_opinion_results.csv"
    pd.DataFrame({
        "id": ["w1", "w2"], "opinion": [2, -2],
        "prompt_tokens": [1, 1], "completion_tokens": [1, 1], "cached_tokens": [0, 0],
    }).to_csv(results_path, index=False)

    result = clean(
        platform="weibo",
        input_dir=str(tmp_path),
        filename_pattern="{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(results_path),
        output_path=str(tmp_path / "weibo_daily_opinion.parquet"),
    )

    row = result.iloc[0]
    assert row["avg_opinion"] == pytest.approx(0.0)
    assert row["weighted_opinion"] == pytest.approx(0.0)
    assert row["user_avg_opinion"] == pytest.approx(2.0)


def _write_twitter_day(tmp_path):
    pd.DataFrame({
        "id": ["t1", "t2", "t3"],
        "text": ["a", "b", "c"],
        "likeCount": [0, 0, 0],
        "author.id": ["us1", "uk1", "us2"],
        "createdAt": ["Fri Mar 01 12:00:00 +0000 2024"] * 3,
    }).to_parquet(tmp_path / "tweets_2024-03-01.parquet", index=False)
    results_path = tmp_path / "twitter_opinion_results.csv"
    pd.DataFrame({
        "id": ["t1", "t2", "t3"], "opinion": [2, -2, 0],
        "prompt_tokens": [1] * 3, "completion_tokens": [1] * 3, "cached_tokens": [0] * 3,
    }).to_csv(results_path, index=False)
    return results_path


def test_clean_keeps_only_listed_users_when_given_a_user_id_filter(tmp_path):
    results_path = _write_twitter_day(tmp_path)
    user_filter = tmp_path / "us_userids.json"
    user_filter.write_text('["us1", "us2"]')

    result = clean(
        platform="twitter",
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(results_path),
        output_path=str(tmp_path / "twitter_daily_opinion.parquet"),
        user_id_filter_path=str(user_filter),
    )

    # uk1's -2 is excluded; mean of us1 (2) and us2 (0)
    assert result.iloc[0]["avg_opinion"] == pytest.approx(1.0)


def test_clean_without_a_user_id_filter_keeps_everyone(tmp_path):
    results_path = _write_twitter_day(tmp_path)

    result = clean(
        platform="twitter",
        input_dir=str(tmp_path),
        filename_pattern="tweets_{date}.parquet",
        start_date="2024-03-01",
        end_date="2024-03-05",
        target_days=[1, 10, 20],
        opinion_results_path=str(results_path),
        output_path=str(tmp_path / "twitter_daily_opinion.parquet"),
    )

    assert result.iloc[0]["avg_opinion"] == pytest.approx(0.0)


def test_clean_cli_location_us_uses_configured_us_userids(tmp_path, monkeypatch):
    import fire

    import prepare_data

    fake_config = _install_fake_config(monkeypatch, tmp_path)
    fake_config.TWITTER_US_USERIDS_PATH = str(tmp_path / "us_userids.json")
    captured = []
    monkeypatch.setattr(prepare_data, "clean", lambda **kw: captured.append(kw))

    fire.Fire({"clean": prepare_data._clean_cli}, command=["clean", "--platform", "twitter", "--location", "us"])
    fire.Fire({"clean": prepare_data._clean_cli}, command=["clean", "--platform", "twitter"])

    assert captured[0]["user_id_filter_path"] == str(tmp_path / "us_userids.json")
    assert captured[1]["user_id_filter_path"] is None


@pytest.mark.parametrize("platform, location", [("weibo", "us"), ("twitter", "uk")])
def test_clean_cli_rejects_unsupported_location(tmp_path, monkeypatch, platform, location):
    import prepare_data

    fake_config = _install_fake_config(monkeypatch, tmp_path)
    fake_config.TWITTER_US_USERIDS_PATH = str(tmp_path / "us_userids.json")
    monkeypatch.setattr(prepare_data, "clean", lambda **kw: None)

    with pytest.raises(ValueError):
        prepare_data._clean_cli(platform, location=location)


def test_missing_like_counts_are_weighted_per_platform_like_legacy(tmp_path):
    # Legacy Weibo: pd.to_numeric(zan).fillna(0) + 1, so a missing like count
    # weighs 1. Legacy Twitter: likeCount + 1 with no fillna, so the post drops
    # out of the weighted mean only.
    weibo_dir, twitter_dir = tmp_path / "weibo", tmp_path / "twitter"
    weibo_dir.mkdir()
    twitter_dir.mkdir()
    pd.DataFrame({
        "weibo_id": ["w1", "w2"], "user_id": [1, 2], "weibo_content": ["a", "b"],
        "zan": [None, 0],
    }).to_parquet(weibo_dir / "2024-03-01.parquet", index=False)
    pd.DataFrame({
        "id": ["t1", "t2"], "text": ["a", "b"], "likeCount": [None, 0],
        "author.id": ["a1", "a2"], "createdAt": ["Fri Mar 01 12:00:00 +0000 2024"] * 2,
    }).to_parquet(twitter_dir / "tweets_2024-03-01.parquet", index=False)
    for platform, ids in (("weibo", ["w1", "w2"]), ("twitter", ["t1", "t2"])):
        pd.DataFrame({
            "id": ids, "opinion": [2, 0],
            "prompt_tokens": [1, 1], "completion_tokens": [1, 1], "cached_tokens": [0, 0],
        }).to_csv(tmp_path / f"{platform}_results.csv", index=False)

    def run(platform, input_dir, pattern):
        return clean(
            platform=platform, input_dir=str(input_dir), filename_pattern=pattern,
            start_date="2024-03-01", end_date="2024-03-05", target_days=[1, 10, 20],
            opinion_results_path=str(tmp_path / f"{platform}_results.csv"),
            output_path=str(tmp_path / f"{platform}_daily.parquet"),
        ).iloc[0]

    weibo = run("weibo", weibo_dir, "{date}.parquet")
    twitter = run("twitter", twitter_dir, "tweets_{date}.parquet")

    assert weibo["avg_opinion"] == pytest.approx(1.0)
    assert weibo["weighted_opinion"] == pytest.approx(1.0)
    assert twitter["avg_opinion"] == pytest.approx(1.0)
    assert twitter["weighted_opinion"] == pytest.approx(0.0)
