"""Platform-specific loaders that read already-extracted per-day parquet files
from the sibling youth-analysis (Weibo) and twitterapi-io (Twitter) repos, and
normalize them into a standard {id, text, user_id, weight_raw, date} frame.
"""

import os
from datetime import datetime, timedelta
from typing import List, Sequence, Union

import pandas as pd

STANDARD_COLUMNS = ["id", "text", "user_id", "weight_raw", "date"]


def parse_target_days(target_days: Union[str, int, Sequence[int]]) -> List[int]:
    """Normalize a --target_days CLI value to List[int].

    Python Fire parses `--target_days 1,10,20` as the tuple (1, 10, 20) and
    `--target_days 10` as the int 10; a quoted value arrives as a string.
    """
    if isinstance(target_days, str):
        return [int(d) for d in target_days.split(",") if d.strip()]
    if isinstance(target_days, int):
        return [target_days]
    return [int(d) for d in target_days]


def _finalize(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    # Empty/null text can't be meaningfully analyzed; don't send it to the model.
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")]
    # The same post can appear more than once in the raw input; each post must
    # count exactly once in the daily metrics.
    df = df.drop_duplicates(subset=["id"])
    return df.reset_index(drop=True)


def iter_target_dates(start_date: str, end_date: str, target_days: List[int]) -> List[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        if current.day in target_days:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def weibo_loader(
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
) -> pd.DataFrame:
    dates = iter_target_dates(start_date, end_date, target_days)
    frames = []
    for date_str in dates:
        file_path = os.path.join(input_dir, filename_pattern.format(date=date_str))
        if not os.path.exists(file_path):
            continue
        df = pd.read_parquet(
            file_path, columns=["weibo_id", "user_id", "weibo_content", "zan"]
        )
        df = df.rename(
            columns={"weibo_id": "id", "weibo_content": "text", "zan": "weight_raw"}
        )
        df["id"] = df["id"].astype(str)
        df["user_id"] = df["user_id"].astype(str)
        # Weibo date = the source file's date, already correctly bucketed
        # upstream by youth-analysis/ai_content_extractor.py. Do not re-derive
        # from time_stamp (ambiguous timezone, dead code in the old pipeline).
        df["date"] = date_str
        frames.append(df[STANDARD_COLUMNS])
    return _finalize(frames)


def twitter_loader(
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
) -> pd.DataFrame:
    dates = iter_target_dates(start_date, end_date, target_days)
    frames = []
    for date_str in dates:
        file_path = os.path.join(input_dir, filename_pattern.format(date=date_str))
        if not os.path.exists(file_path):
            continue
        df = pd.read_parquet(
            file_path, columns=["id", "text", "likeCount", "author.id", "createdAt"]
        )
        df["id"] = df["id"].astype(str)
        # Twitter date = parsed from the post's own createdAt (UTC), same as
        # the current batch_sentiment_analysis.py behavior.
        df["date"] = df["createdAt"].apply(
            lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S +0000 %Y").strftime(
                "%Y-%m-%d"
            )
        )
        df = df.rename(columns={"likeCount": "weight_raw", "author.id": "user_id"})
        df["user_id"] = df["user_id"].astype(str)
        frames.append(df[STANDARD_COLUMNS])
    return _finalize(frames)
