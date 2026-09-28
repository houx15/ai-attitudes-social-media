"""Stage 2 CLI: clean opinion results, compute daily aggregates, export figure data.

Usage:
    python prepare_data.py clean --platform weibo
    python prepare_data.py clean --platform twitter
    python prepare_data.py clean --platform weibo --start_date 2024-03-01 --end_date 2024-03-31 --target_days 1,10,20
    python prepare_data.py export
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import fire
import pandas as pd

from loaders import day_files, iter_platform_days, parse_target_days
from openrouter_client import normalize_opinion, read_results, results_parts


def _load_opinions(opinion_results_dir: str) -> pd.DataFrame:
    opinions = read_results(opinion_results_dir, columns=["id", "opinion", "provider"])
    opinions = opinions.astype({"id": str})
    # A retried post appears again (a later part, or another task's part). Each
    # post counts once: its latest valid label, else its latest attempt.
    valid = opinions["opinion"].map(normalize_opinion).notna()
    opinions = opinions.assign(_valid=valid).sort_values("_valid", kind="stable")
    return opinions.drop_duplicates(subset=["id"], keep="last").drop(columns="_valid")


ATTITUDES = ["2", "1", "0", "-1", "-2"]
CANNOT_TELL_KEY = "cannot tell"
UNLABELED_KEY = "unlabeled"


def _label_key(value) -> str:
    """"2".."-2", "cannot tell", or "unlabeled" (failed, invalid, or never sent)."""
    label = normalize_opinion(value)
    return UNLABELED_KEY if label is None else str(label)


def _label_summary(labels: Counter) -> dict:
    with_attitude = sum(labels[a] for a in ATTITUDES)
    return {
        "posts": sum(labels.values()),
        "with_attitude": with_attitude,
        "cannot_tell": labels[CANNOT_TELL_KEY],
        "unlabeled": labels[UNLABELED_KEY],
        "attitude_counts": {a: labels[a] for a in ATTITUDES},
        "attitude_percent": {
            a: (round(100 * labels[a] / with_attitude, 2) if with_attitude else None)
            for a in ATTITUDES
        },
    }


def _print_sample_stats(stats: dict) -> None:
    sample, analytic, user_filter = stats["sample"], stats["analytic"], stats["user_filter"]
    print(
        f"Sample statistics for {stats['platform']} "
        f"({stats['start_date']}..{stats['end_date']}, {len(stats['dates'])} dates):"
    )
    print(
        f"  Input files: {sample['raw_rows']:,} rows; after removing "
        f"{sample['duplicates_removed']:,} duplicate ids: {sample['posts']:,} posts "
        f"from {sample['users']:,} users"
    )
    if user_filter is not None:
        print(
            f"  User filter: {user_filter['listed_users']:,} users listed; "
            f"{analytic['users']:,} of them posted, {analytic['posts']:,} posts"
        )
    print(
        f"  Analytic subset: {analytic['posts']:,} posts: "
        f"{analytic['with_attitude']:,} with an attitude, "
        f"{analytic['cannot_tell']:,} cannot tell, {analytic['unlabeled']:,} unlabeled"
    )
    print(
        "  Attitude distribution (% of posts with an attitude): "
        + ", ".join(
            f"{a} = {p:.2f}%" if p is not None else f"{a} = n/a"
            for a, p in analytic["attitude_percent"].items()
        )
    )


def clean(
    platform: str,
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
    opinion_results_dir: str,
    output_path: str,
    date_substitutions: Optional[Dict[str, str]] = None,
    user_id_filter_path: Optional[str] = None,
) -> pd.DataFrame:
    """Daily avg / weighted / user-level opinion, reading one day file at a time.

    Each file contributes sums and counts (per date, and per date and user); the
    three metrics are computed from their totals, which is identical to computing
    them over all posts at once.

    Also writes `{platform}_sample_stats.json` next to `output_path`: post and
    user counts, and the label breakdown of the analytic subset (after the user
    filter), for the paper's data description.
    """
    files = day_files(
        platform, input_dir, filename_pattern, start_date, end_date, target_days, date_substitutions
    )
    opinions = _load_opinions(opinion_results_dir)
    label_by_id = opinions.set_index("id")["opinion"].map(_label_key)
    kept_user_ids = None
    if user_id_filter_path is not None:
        with open(user_id_filter_path, "r") as f:
            kept_user_ids = json.load(f)

    counts = {"loaded": 0, "matched": 0, "user_filtered": 0, "valid": 0}
    providers = Counter()
    kept_users = set()
    raw_rows, sample_users, analytic_users = 0, set(), set()
    sample_labels, analytic_labels = Counter(), Counter()
    posts_by_date = Counter()
    date_parts, user_parts = [], []
    for frame in iter_platform_days(platform, files, with_text=False):
        counts["loaded"] += len(frame)
        raw_rows += frame.attrs.get("raw_rows", len(frame))
        labels = frame["id"].map(label_by_id).fillna(UNLABELED_KEY)
        sample_labels.update(labels)
        sample_users.update(frame["user_id"].dropna())
        analytic = frame["user_id"].isin(kept_user_ids) if kept_user_ids is not None else None
        analytic_frame = frame if analytic is None else frame[analytic]
        analytic_labels.update(labels if analytic is None else labels[analytic])
        analytic_users.update(analytic_frame["user_id"].dropna())
        posts_by_date.update(analytic_frame["date"])

        merged = frame.merge(opinions, on="id", how="inner")
        counts["matched"] += len(merged)

        if kept_user_ids is not None:
            merged = merged[merged["user_id"].isin(kept_user_ids)]
            counts["user_filtered"] += len(merged)
            kept_users.update(merged["user_id"].dropna())

        merged = merged.assign(opinion=pd.to_numeric(merged["opinion"], errors="coerce"))
        merged = merged.dropna(subset=["opinion"])
        counts["valid"] += len(merged)
        if "provider" in merged.columns:
            providers.update(merged["provider"].fillna("unknown"))
        else:
            providers["unknown"] += len(merged)

        weight_raw = pd.to_numeric(merged["weight_raw"], errors="coerce")
        # Legacy behaviour differs per platform: Weibo filled a missing like count
        # with 0; Twitter left it missing, which drops the post from the weighted
        # mean only (pandas sums skip NaN in both numerator and denominator).
        if platform == "weibo":
            weight_raw = weight_raw.fillna(0)
        merged = merged.assign(weight=weight_raw + 1)
        merged = merged.assign(opinion_weight=merged["opinion"] * merged["weight"])

        date_parts.append(
            merged.groupby("date").agg(
                opinion_sum=("opinion", "sum"),
                opinion_count=("opinion", "size"),
                opinion_weight_sum=("opinion_weight", "sum"),
                weight_sum=("weight", "sum"),
            )
        )
        # groupby drops posts with a missing user_id, as in the legacy code.
        user_parts.append(
            merged.groupby(["date", "user_id"])["opinion"].agg(["sum", "count"])
        )

    if kept_user_ids is not None:
        print(
            f"User filter {user_id_filter_path}: kept {counts['user_filtered']} of "
            f"{counts['matched']} posts from {len(kept_users)} users"
        )
    print(
        f"Coverage for {platform} ({start_date}..{end_date}, days {list(target_days)}): "
        f"{counts['loaded']} metadata rows loaded, "
        f"{counts['matched']} matched an opinion result, "
        f"{counts['valid']} with a valid numeric opinion"
    )
    print("Labels by provider: " + ", ".join(f"{name} {n}" for name, n in providers.most_common()))

    stats = {
        "platform": platform,
        "start_date": start_date,
        "end_date": end_date,
        "target_days": list(target_days),
        "dates": [date for date, _ in files],
        "sample": {
            "raw_rows": raw_rows,
            "duplicates_removed": raw_rows - counts["loaded"],
            "posts": counts["loaded"],
            "users": len(sample_users),
            "labels": _label_summary(sample_labels),
        },
        "user_filter": (
            None
            if kept_user_ids is None
            else {"path": user_id_filter_path, "listed_users": len(set(kept_user_ids))}
        ),
        "analytic": {"users": len(analytic_users), **_label_summary(analytic_labels)},
        "analytic_posts_by_date": dict(sorted(posts_by_date.items())),
    }
    _print_sample_stats(stats)
    stats_path = Path(output_path).with_name(f"{platform}_sample_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Saved sample statistics to {stats_path}")

    if date_parts:
        by_date = pd.concat(date_parts).groupby(level="date").sum()
        by_user = pd.concat(user_parts).groupby(level=["date", "user_id"]).sum()
        user_means = (by_user["sum"] / by_user["count"]).groupby(level="date").mean()
        result = pd.DataFrame(
            {
                "avg_opinion": by_date["opinion_sum"] / by_date["opinion_count"],
                "weighted_opinion": by_date["opinion_weight_sum"] / by_date["weight_sum"],
                "user_avg_opinion": user_means,
            }
        ).rename_axis("date").reset_index()
    else:
        result = pd.DataFrame(columns=["date", "avg_opinion", "weighted_opinion", "user_avg_opinion"])
    result = result.sort_values("date").reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved daily opinion for {platform} to {output_path} ({len(result)} dates)")
    return result


def _clean_cli(
    platform: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_days: Optional[Union[str, int, Sequence[int]]] = None,
):
    import config

    # Twitter is always restricted to US users, applied at aggregation time as
    # in the legacy `batch_sentiment_analysis.py calculate --location us`.
    user_id_filter_path = config.TWITTER_US_USERIDS_PATH if platform == "twitter" else None

    input_dir = config.WEIBO_INPUT_DIR if platform == "weibo" else config.TWITTER_INPUT_DIR
    filename_pattern = (
        config.WEIBO_FILENAME_PATTERN
        if platform == "weibo"
        else config.TWITTER_FILENAME_PATTERN
    )
    # Every run and task of this platform (see run_analysis.py --task_id).
    opinion_results_dir = f"{config.OUTPUT_DIR}/analysis_results/{platform}_opinion_results"
    if not results_parts(opinion_results_dir):
        raise FileNotFoundError(f"No Stage 1 results in {opinion_results_dir}/")
    output_path = f"{config.OUTPUT_DIR}/{platform}_daily_opinion.parquet"

    clean(
        platform=platform,
        input_dir=input_dir,
        filename_pattern=filename_pattern,
        start_date=start_date or config.START_DATE,
        end_date=end_date or config.END_DATE,
        target_days=(
            parse_target_days(target_days) if target_days is not None else config.TARGET_DAYS
        ),
        opinion_results_dir=opinion_results_dir,
        output_path=output_path,
        date_substitutions=getattr(config, "DATE_SUBSTITUTIONS", {}).get(platform, {}),
        user_id_filter_path=user_id_filter_path,
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
