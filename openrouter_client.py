"""Concurrent OpenRouter chat-completions client used by both platforms
(replaces the old OpenAI Batch API + hosted-prompt mechanism)."""

import csv
import json
import logging
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import pandas as pd
from tqdm import tqdm

from prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger(__name__)

VALID_NUMERIC_OPINIONS = {-2, -1, 0, 1, 2}
CANNOT_TELL = "cannot tell"
RESULTS_HEADER = ["id", "opinion", "prompt_tokens", "completion_tokens", "cached_tokens", "provider"]
# Fixed for every request on both platforms, like the prompt:
# - no thinking step, and deterministic decoding;
# - one upstream provider (fp8), never silently switching: if it is unavailable
#   the request fails and is retried on the next run.
REASONING = {"effort": "none"}
TEMPERATURE = 0
PROVIDER = {"only": ["deepinfra"], "allow_fallbacks": False}


def request_options() -> Dict[str, Any]:
    return {
        "temperature": TEMPERATURE,
        "extra_body": {"reasoning": REASONING, "provider": PROVIDER},
    }


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


def _cached_tokens(usage) -> int:
    # OpenAI-compatible APIs nest it under prompt_tokens_details; some put it top-level.
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) or getattr(usage, "cached_tokens", None)
    return int(cached or 0)


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
                    **request_options(),
                )
                response_text = response.choices[0].message.content.strip()
                usage = response.usage
                token_stats = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "cached_tokens": _cached_tokens(usage),
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

                provider = getattr(response, "provider", None) or ""
                return {"opinion": opinion, **token_stats, "provider": provider}
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
            "provider": "",
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


TOKEN_COLUMNS = ["prompt_tokens", "cached_tokens", "completion_tokens"]


def format_tokens(n: int) -> str:
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.1f}k"
    return f"{n / 1_000_000:.2f}M"


def _print_token_summary(desc: str, label: str, totals: Dict[str, int]) -> None:
    print(
        f"{desc} tokens {label}: {totals['prompt_tokens']:,} prompt "
        f"({totals['cached_tokens']:,} cached) + {totals['completion_tokens']:,} output",
        flush=True,
    )


def analyze_many(
    client, df: pd.DataFrame, results_path: str, max_workers: int = 8, desc: str = "Analyzing"
) -> Dict:
    """Analyze the posts in one frame (see analyze_stream)."""
    return analyze_stream(client, [df], results_path, max_workers=max_workers, desc=desc, total_rows=len(df))


def analyze_stream(
    client,
    frames: Iterable[pd.DataFrame],
    results_path: str,
    max_workers: int = 8,
    desc: str = "Analyzing",
    total_rows: Optional[int] = None,
    done_paths: Optional[Iterable[str]] = None,
) -> Dict:
    """Label posts frame by frame (one day at a time), appending to the results CSV.

    Only max_workers * 4 requests are ever queued, so memory stays flat however
    many posts there are, and the next frame is read only when needed. Posts
    whose id already has a valid label in the results CSV are skipped.
    """
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Posts with a valid label in any of these files are skipped (default: this
    # run's own results file).
    done_paths = list(done_paths) if done_paths is not None else [results_path]
    processed_ids = set()
    for done_path in done_paths:
        processed_ids |= load_processed_ids(done_path)
    print(
        f"{desc}: {len(processed_ids):,} posts already labelled in "
        + ", ".join(Path(d).name for d in done_paths),
        flush=True,
    )

    summary = {"total": 0, "skipped": 0, "completed": 0, "failed": 0}
    run_tokens = {column: 0 for column in TOKEN_COLUMNS}
    write_lock = threading.Lock()
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with open(path, "r", encoding="utf-8", newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != RESULTS_HEADER:
            raise ValueError(
                f"{path} was written in an older format without the provider column. "
                "Move it out of analysis_results/ so its posts are re-labelled by the "
                "pinned provider, then rerun."
            )

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
                "provider": "",
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
                        result.get("provider") or "",
                    ]
                )
        return result

    def record(future, bar):
        result = future.result()
        if result["opinion"] is None:
            summary["failed"] += 1
        else:
            summary["completed"] += 1
        for column in TOKEN_COLUMNS:
            run_tokens[column] += int(result.get(column) or 0)
        bar.update(1)
        bar.set_postfix(
            {
                "completed": summary["completed"],
                "failed": summary["failed"],
                "in": format_tokens(run_tokens["prompt_tokens"]),
                "out": format_tokens(run_tokens["completion_tokens"]),
            },
            refresh=False,
        )

    max_in_flight = max_workers * 4
    in_flight = set()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    # Refresh every 30s when output goes to a log file, so it isn't flooded.
    bar = tqdm(
        total=total_rows,
        desc=desc,
        unit="post",
        mininterval=0.5 if sys.stderr.isatty() else 30,
    )
    try:
        for frame in frames:
            ids = frame["id"].astype(str)
            todo = ~ids.isin(processed_ids)
            n_todo = int(todo.sum())
            summary["total"] += len(frame)
            summary["skipped"] += len(frame) - n_todo
            # Rows dropped while reading (duplicates, empty text) and rows already
            # labelled count toward the bar right away.
            bar.update(frame.attrs.get("raw_rows", len(frame)) - n_todo)
            for row_id, text in zip(ids[todo], frame.loc[todo, "text"]):
                while len(in_flight) >= max_in_flight:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        record(future, bar)
                in_flight.add(executor.submit(process_row, row_id, text))
        done, in_flight = wait(in_flight)
        for future in done:
            record(future, bar)
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        bar.close()
        print(f"\n{desc}: interrupted; rerun the same command to resume.", flush=True)
        raise
    executor.shutdown()
    bar.close()

    _print_token_summary(desc, "this run", run_tokens)
    if path.exists():
        all_runs = pd.read_csv(path, usecols=TOKEN_COLUMNS).fillna(0).astype(int).sum()
        _print_token_summary(desc, "all runs", {column: int(all_runs[column]) for column in TOKEN_COLUMNS})
    return summary
