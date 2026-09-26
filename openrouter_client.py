"""Concurrent OpenRouter chat-completions client used by both platforms
(replaces the old OpenAI Batch API + hosted-prompt mechanism)."""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from prompts import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 3,
        timeout: int = 60,
        client=None,
    ):
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.max_retries = max_retries

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
                        opinion = json.loads(response_text[json_start:json_end]).get(
                            "opinion"
                        )
                    except json.JSONDecodeError:
                        opinion = None

                return {"opinion": opinion, **token_stats}
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep((attempt + 1) * 0.01)

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
    df = pd.read_csv(path, dtype=str)
    if "id" not in df.columns:
        return set()
    return set(df["id"].dropna().unique())


def analyze_many(
    client, df: pd.DataFrame, results_path: str, max_workers: int = 8
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
    if len(todo) == 0:
        return summary

    write_lock = threading.Lock()
    write_header = not path.exists() or path.stat().st_size == 0

    def process_row(row_id, row_text):
        result = client.analyze_one(row_text)
        opinion_value = "" if result["opinion"] is None else result["opinion"]
        nonlocal write_header
        with write_lock:
            with open(path, "a", encoding="utf-8", newline="") as f:
                if write_header:
                    f.write("id,opinion,prompt_tokens,completion_tokens,cached_tokens\n")
                    write_header = False
                f.write(
                    f'{row_id},{opinion_value},{result["prompt_tokens"]},'
                    f'{result["completion_tokens"]},{result["cached_tokens"]}\n'
                )
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_row, row["id"], row["text"])
            for _, row in todo.iterrows()
        ]
        for future in as_completed(futures):
            result = future.result()
            if result["opinion"] is None:
                summary["failed"] += 1
            else:
                summary["completed"] += 1

    return summary
