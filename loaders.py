"""Platform-specific loaders that read already-extracted per-day parquet files
from the sibling youth-analysis (Weibo) and twitterapi-io (Twitter) repos, and
normalize them into a standard {id, text, user_id, weight_raw, date} frame.
"""

import os
from datetime import datetime, timedelta
from typing import List

import pandas as pd

STANDARD_COLUMNS = ["id", "text", "user_id", "weight_raw", "date"]


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
