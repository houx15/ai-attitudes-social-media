"""Platform-specific loaders that read already-extracted per-day parquet files
from the sibling youth-analysis (Weibo) and twitterapi-io (Twitter) repos, and
normalize them into a standard {id, text, user_id, weight_raw, date} frame.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq

STANDARD_COLUMNS = ["id", "text", "user_id", "weight_raw", "date"]
PLATFORMS = ("weibo", "twitter")


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


def iter_target_dates(
    start_date: str,
    end_date: str,
    target_days: List[int],
    substitutions: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Sampled dates in range. A nominal date listed in `substitutions` is replaced
    by its stand-in (a nearby day read because the nominal day's data is missing)."""
    substitutions = substitutions or {}
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        if current.day in target_days:
            nominal = current.strftime("%Y-%m-%d")
            dates.append(substitutions.get(nominal, nominal))
        current += timedelta(days=1)
    return dates


def day_files(
    platform: str,
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
    substitutions: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """(date, path) for each sampled date that has an input file; warns about the rest."""
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform: {platform!r}, expected one of {list(PLATFORMS)}")
    found, missing = [], []
    for date_str in iter_target_dates(start_date, end_date, target_days, substitutions):
        file_path = os.path.join(input_dir, filename_pattern.format(date=date_str))
        if os.path.exists(file_path):
            found.append((date_str, file_path))
        else:
            missing.append(date_str)
    if missing:
        print(
            f"Warning: no {platform} input file for {len(missing)} target date(s): "
            + ", ".join(missing)
        )
    return found


def count_input_rows(files: List[Tuple[str, str]]) -> int:
    """Total rows across the day files, read from parquet metadata only."""
    return sum(pq.ParquetFile(path).metadata.num_rows for _, path in files)


def _read_weibo_day(date_str: str, file_path: str, with_text: bool) -> pd.DataFrame:
    columns = ["weibo_id", "user_id", "zan"] + (["weibo_content"] if with_text else [])
    df = pd.read_parquet(file_path, columns=columns)
    df = df.rename(columns={"weibo_id": "id", "weibo_content": "text", "zan": "weight_raw"})
    df["id"] = df["id"].astype(str)
    # Weibo date = the source file's date, already correctly bucketed
    # upstream by youth-analysis/ai_content_extractor.py. Do not re-derive
    # from time_stamp (ambiguous timezone, dead code in the old pipeline).
    df["date"] = date_str
    if with_text:
        # Legacy ai_sentiment_analyzer.py sent every post as str(weibo_content or "").
        df["text"] = df["text"].fillna("")
    return df


def _read_twitter_day(date_str: str, file_path: str, with_text: bool) -> pd.DataFrame:
    columns = ["id", "likeCount", "author.id", "createdAt"] + (["text"] if with_text else [])
    df = pd.read_parquet(file_path, columns=columns)
    df["id"] = df["id"].astype(str)
    # Twitter date = parsed from the post's own createdAt (UTC), same as
    # the legacy batch_sentiment_analysis.py.
    df["date"] = df["createdAt"].apply(
        lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S +0000 %Y").strftime("%Y-%m-%d")
    )
    df = df.rename(columns={"likeCount": "weight_raw", "author.id": "user_id"})
    if with_text:
        # Legacy batch_sentiment_analysis.py skipped a tweet only when its text
        # was null or "". (Stage 2 never reads text; those tweets have no label.)
        df = df[df["text"].notna() & (df["text"] != "")]
    return df


def iter_platform_days(
    platform: str, files: List[Tuple[str, str]], with_text: bool = True
) -> Iterator[pd.DataFrame]:
    """Yield one normalized frame per day file, reading a single file at a time.

    A post seen on an earlier day is dropped from later days, so each post counts
    once (legacy: first occurrence wins). `frame.attrs["raw_rows"]` is the file's
    row count before any row was dropped, for progress reporting.
    """
    read_day = _read_weibo_day if platform == "weibo" else _read_twitter_day
    columns = STANDARD_COLUMNS if with_text else [c for c in STANDARD_COLUMNS if c != "text"]
    seen_ids = set()
    for date_str, file_path in files:
        df = read_day(date_str, file_path, with_text)
        raw_rows = pq.ParquetFile(file_path).metadata.num_rows
        df = df.drop_duplicates(subset=["id"])
        df = df[~df["id"].isin(seen_ids)]
        seen_ids.update(df["id"])
        df = df[columns].reset_index(drop=True)
        df.attrs["raw_rows"] = raw_rows
        yield df


def _load_all(platform: str, files: List[Tuple[str, str]]) -> pd.DataFrame:
    frames = list(iter_platform_days(platform, files))
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def weibo_loader(
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
    substitutions: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """All sampled Weibo posts in one frame (small data only; the stages stream)."""
    files = day_files("weibo", input_dir, filename_pattern, start_date, end_date, target_days, substitutions)
    return _load_all("weibo", files)


def twitter_loader(
    input_dir: str,
    filename_pattern: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
    substitutions: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """All sampled tweets in one frame (small data only; the stages stream)."""
    files = day_files("twitter", input_dir, filename_pattern, start_date, end_date, target_days, substitutions)
    return _load_all("twitter", files)
