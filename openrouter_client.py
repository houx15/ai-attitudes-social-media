"""Concurrent OpenRouter chat-completions client used by both platforms
(replaces the old OpenAI Batch API + hosted-prompt mechanism)."""

import csv
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from tqdm import tqdm

from prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger(__name__)

VALID_NUMERIC_OPINIONS = {-2, -1, 0, 1, 2}
CANNOT_TELL = "cannot tell"
RESULTS_HEADER = ["id", "opinion", "prompt_tokens", "completion_tokens", "cached_tokens"]
# Same for both platforms, like the prompt: the model answers directly, no thinking step.
REASONING = {"effort": "none"}


def normalize_opinion(value: Any) -> Optional[Union[int, str]]:
    """Return the canonical opinion (-2..2 as int, or "cannot tell"), or None if
    the value is not a valid label. Numeric-looking values (2, 2.0, "2") are
    normalized to int; anything else (lists, dicts, bools, out-of-range numbers,
    arbitrary strings) is rejected."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == CANNOT_TELL:
            return CANNOT_TELL
        try:
            value = float(stripped)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        if value != value or not float(value).is_integer():  # NaN or non-integer
            return None
        as_int = int(value)
        return as_int if as_int in VALID_NUMERIC_OPINIONS else None
    return None


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 3,
        timeout: int = 60,
        client=None,
        backoff_base_seconds: float = 0.01,
    ):
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        # Retry sleep is (attempt + 1) * backoff_base_seconds. The small default
        # keeps tests fast; production callers (run_analysis.main) pass a
        # realistic value from config.
        self.backoff_base_seconds = backoff_base_seconds

    def analyze_one(self, text: str) -> Dict:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_message(text)},
                    ],
                    extra_body={"reasoning": REASONING},
                )
                response_text = response.choices[0].message.content.strip()
                usage = response.usage
                token_stats = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "cached_tokens": getattr(usage, "cached_tokens", 0) if usage else 0,
                }

                opinion = None
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    try:
                        parsed = json.loads(response_text[json_start:json_end])
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        opinion = normalize_opinion(parsed.get("opinion"))

                return {"opinion": opinion, **token_stats}
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep((attempt + 1) * self.backoff_base_seconds)

        logger.error(f"analyze_one failed after {self.max_retries} retries: {last_error}")
        return {
            "opinion": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }


def load_processed_ids(results_path: str) -> set:
    path = Path(results_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "id" not in df.columns or "opinion" not in df.columns:
        return set()
    # Only rows with a valid opinion count as done; failed rows (empty or
    # invalid opinion) are retried on the next run instead of being dropped.
    valid = df["opinion"].map(normalize_opinion).notna()
    ids = df.loc[valid, "id"]
    return set(ids[ids != ""].unique())


def analyze_many(
    client, df: pd.DataFrame, results_path: str, max_workers: int = 8, desc: str = "Analyzing"
) -> Dict:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = load_processed_ids(results_path)
    df = df.copy()
    df["id"] = df["id"].astype(str)
    todo = df[~df["id"].isin(processed_ids)]

    summary = {
        "total": len(df),
        "skipped": len(df) - len(todo),
        "completed": 0,
        "failed": 0,
    }
    print(
        f"{desc}: {summary['total']} posts: {summary['skipped']} already done, "
        f"{len(todo)} to analyze",
        flush=True,
    )
    if len(todo) == 0:
        return summary

    write_lock = threading.Lock()
    write_header = not path.exists() or path.stat().st_size == 0

    def process_row(row_id, row_text):
        try:
            result = client.analyze_one(row_text)
        except Exception as e:
            logger.error(f"Exception calling client.analyze_one for row {row_id}: {e}")
            result = {
                "opinion": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
            }
        opinion_value = "" if result["opinion"] is None else result["opinion"]
        nonlocal write_header
        with write_lock:
            with open(path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(RESULTS_HEADER)
                    write_header = False
                writer.writerow(
                    [
                        row_id,
                        opinion_value,
                        result["prompt_tokens"],
                        result["completion_tokens"],
                        result["cached_tokens"],
                    ]
                )
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_row, row["id"], row["text"])
            for _, row in todo.iterrows()
        ]
        # Refresh every 30s when output goes to a log file, so it isn't flooded.
        with tqdm(
            total=len(futures),
            desc=desc,
            unit="post",
            mininterval=0.5 if sys.stderr.isatty() else 30,
        ) as bar:
            for future in as_completed(futures):
                result = future.result()
                if result["opinion"] is None:
                    summary["failed"] += 1
                else:
                    summary["completed"] += 1
                bar.update(1)
                bar.set_postfix(completed=summary["completed"], failed=summary["failed"], refresh=False)

    return summary
