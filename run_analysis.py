"""Stage 1 CLI: AI-opinion analysis via OpenRouter, for one platform at a time.

Usage:
    python run_analysis.py weibo
    python run_analysis.py twitter
    python run_analysis.py weibo --start_date 2024-03-01 --end_date 2024-03-31
    python run_analysis.py weibo --target_days 1,10,20
    python run_analysis.py twitter --task_id 2 --num_tasks 4   # one of 4 parallel tasks
"""

from pathlib import Path

from typing import Dict, List, Optional, Sequence, Union

import fire

from loaders import assign_task_files, count_input_rows, day_files, iter_platform_days, parse_target_days
from openrouter_client import OpenRouterClient, analyze_stream


def analyze(
    platform: str,
    input_dir: str,
    filename_pattern: str,
    output_path: str,
    api_key: str,
    base_url: str,
    model: str,
    start_date: str,
    end_date: str,
    target_days: List[int],
    workers: int = 8,
    max_retries: int = 3,
    client=None,
    timeout: int = 60,
    backoff_base_seconds: float = 0.01,
    date_substitutions: Optional[Dict[str, str]] = None,
    task_id: int = 1,
    num_tasks: int = 1,
) -> dict:
    files = day_files(
        platform, input_dir, filename_pattern, start_date, end_date, target_days, date_substitutions
    )
    files = assign_task_files(files, task_id, num_tasks)
    if not files:
        print(f"No input files found for platform={platform} in range {start_date}..{end_date}")
        return {"total": 0, "skipped": 0, "completed": 0, "failed": 0}

    if client is None:
        client = OpenRouterClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_retries=max_retries,
            timeout=timeout,
            backoff_base_seconds=backoff_base_seconds,
        )

    # A post labelled by any earlier run or task of this platform is not re-sent.
    output_path = Path(output_path)
    done_paths = sorted(set(output_path.parent.glob(f"{platform}_opinion_results*.csv")) | {output_path})
    summary = analyze_stream(
        client,
        iter_platform_days(platform, files),
        str(output_path),
        max_workers=workers,
        desc=platform if num_tasks == 1 else f"{platform} {task_id}/{num_tasks}",
        total_rows=count_input_rows(files),
        done_paths=[str(p) for p in done_paths],
    )
    print(f"Stage 1 analysis summary for {platform}: {summary}")
    return summary


def main(
    platform: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_days: Optional[Union[str, int, Sequence[int]]] = None,
    workers: Optional[int] = None,
    task_id: int = 1,
    num_tasks: int = 1,
):
    import config

    input_dir = config.WEIBO_INPUT_DIR if platform == "weibo" else config.TWITTER_INPUT_DIR
    filename_pattern = (
        config.WEIBO_FILENAME_PATTERN
        if platform == "weibo"
        else config.TWITTER_FILENAME_PATTERN
    )
    suffix = "" if num_tasks == 1 else f"_task{task_id}of{num_tasks}"
    output_path = f"{config.OUTPUT_DIR}/analysis_results/{platform}_opinion_results{suffix}.csv"
    days = parse_target_days(target_days) if target_days is not None else config.TARGET_DAYS

    analyze(
        platform=platform,
        input_dir=input_dir,
        filename_pattern=filename_pattern,
        output_path=output_path,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        model=config.OPENROUTER_MODEL,
        start_date=start_date or config.START_DATE,
        end_date=end_date or config.END_DATE,
        target_days=days,
        workers=workers or config.MAX_WORKERS,
        max_retries=config.MAX_RETRIES,
        timeout=config.REQUEST_TIMEOUT,
        backoff_base_seconds=getattr(config, "BACKOFF_BASE_SECONDS", 2.0),
        date_substitutions=getattr(config, "DATE_SUBSTITUTIONS", {}).get(platform, {}),
        task_id=task_id,
        num_tasks=num_tasks,
    )


if __name__ == "__main__":
    fire.Fire(main)
