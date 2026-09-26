"""Stage 2 CLI: clean opinion results, compute daily aggregates, export figure data.

Usage:
    python prepare_data.py clean --platform weibo
    python prepare_data.py clean --platform twitter
    python prepare_data.py export
"""

from pathlib import Path
from typing import List

import fire
import pandas as pd

from loaders import twitter_loader, weibo_loader

LOADERS = {"weibo": weibo_loader, "twitter": twitter_loader}


def clean(
    platform: str,
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
    opinion_results_path: str,
    output_path: str,
) -> pd.DataFrame:
    if platform not in LOADERS:
        raise ValueError(f"Unknown platform: {platform!r}, expected one of {list(LOADERS)}")

    loader = LOADERS[platform]
    meta_df = loader(input_dir, filename_pattern, start_date, end_date, target_days)

    opinions_df = pd.read_csv(opinion_results_path, dtype={"id": str})
    merged = meta_df.merge(opinions_df, on="id", how="inner")

    merged["opinion"] = pd.to_numeric(merged["opinion"], errors="coerce")
    merged = merged.dropna(subset=["opinion"])
    merged["weight"] = pd.to_numeric(merged["weight_raw"], errors="coerce").fillna(0) + 1

    daily_avg = merged.groupby("date")["opinion"].mean().reset_index()
    daily_avg.columns = ["date", "avg_opinion"]

    merged["opinion_weight"] = merged["opinion"] * merged["weight"]
    daily_weighted = (
        merged.groupby("date")
        .agg({"opinion_weight": "sum", "weight": "sum"})
        .reset_index()
    )
    daily_weighted["weighted_opinion"] = (
        daily_weighted["opinion_weight"] / daily_weighted["weight"]
    )
    daily_weighted = daily_weighted[["date", "weighted_opinion"]]

    user_daily_avg = (
        merged.groupby(["date", "user_id"])["opinion"].mean().reset_index()
    )
    user_daily_avg.columns = ["date", "user_id", "user_daily_avg_opinion"]
    daily_user_avg = (
        user_daily_avg.groupby("date")["user_daily_avg_opinion"].mean().reset_index()
    )
    daily_user_avg.columns = ["date", "user_avg_opinion"]

    result = daily_avg.merge(daily_weighted, on="date", how="outer")
    result = result.merge(daily_user_avg, on="date", how="outer")
    result = result.sort_values("date").reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved daily opinion for {platform} to {output_path} ({len(result)} dates)")
    return result


def _clean_cli(platform: str):
    import config

    input_dir = config.WEIBO_INPUT_DIR if platform == "weibo" else config.TWITTER_INPUT_DIR
    filename_pattern = (
        config.WEIBO_FILENAME_PATTERN
        if platform == "weibo"
        else config.TWITTER_FILENAME_PATTERN
    )
    opinion_results_path = f"{config.OUTPUT_DIR}/analysis_results/{platform}_opinion_results.csv"
    output_path = f"{config.OUTPUT_DIR}/{platform}_daily_opinion.parquet"

    clean(
        platform=platform,
        input_dir=input_dir,
        filename_pattern=filename_pattern,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        target_days=config.TARGET_DAYS,
        opinion_results_path=opinion_results_path,
        output_path=output_path,
    )


def export(weibo_path: str, twitter_path: str, output_path: str) -> pd.DataFrame:
    weibo_df = pd.read_parquet(weibo_path).rename(
        columns={
            "avg_opinion": "weibo_avg_opinion",
            "weighted_opinion": "weibo_weighted_opinion",
            "user_avg_opinion": "weibo_user_avg_opinion",
        }
    )
    twitter_df = pd.read_parquet(twitter_path).rename(
        columns={
            "avg_opinion": "twitter_avg_opinion",
            "weighted_opinion": "twitter_weighted_opinion",
            "user_avg_opinion": "twitter_user_avg_opinion",
        }
    )

    merged = weibo_df.merge(twitter_df, on="date", how="outer")
    merged = merged[
        [
            "date",
            "weibo_avg_opinion",
            "twitter_avg_opinion",
            "weibo_weighted_opinion",
            "twitter_weighted_opinion",
            "weibo_user_avg_opinion",
            "twitter_user_avg_opinion",
        ]
    ]
    merged = merged.sort_values("date").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    merged.to_csv(output_path.with_suffix(".csv"), index=False)
    print(f"Saved figure data to {output_path} ({len(merged)} dates)")
    return merged


def _export_cli():
    import config

    export(
        weibo_path=f"{config.OUTPUT_DIR}/weibo_daily_opinion.parquet",
        twitter_path=f"{config.OUTPUT_DIR}/twitter_daily_opinion.parquet",
        output_path=f"{config.OUTPUT_DIR}/figure_data.parquet",
    )


def main():
    fire.Fire({"clean": _clean_cli, "export": _export_cli})


if __name__ == "__main__":
    main()
