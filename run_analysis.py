"""Stage 1 CLI: AI-opinion analysis via OpenRouter, for one platform at a time.

Usage:
    python run_analysis.py weibo
    python run_analysis.py twitter
    python run_analysis.py weibo --start_date 2024-03-01 --end_date 2024-03-31
    python run_analysis.py weibo --target_days 1,10,20
"""

from typing import List, Optional, Sequence, Union

import fire

from loaders import parse_target_days, twitter_loader, weibo_loader
from openrouter_client import OpenRouterClient, analyze_many

LOADERS = {"weibo": weibo_loader, "twitter": twitter_loader}


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
) -> dict:
    if platform not in LOADERS:
        raise ValueError(f"Unknown platform: {platform!r}, expected one of {list(LOADERS)}")

    loader = LOADERS[platform]
    df = loader(input_dir, filename_pattern, start_date, end_date, target_days)

    if len(df) == 0:
        print(f"No input rows found for platform={platform} in range {start_date}..{end_date}")
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

    summary = analyze_many(client, df, output_path, max_workers=workers)
    print(f"Stage 1 analysis summary for {platform}: {summary}")
    return summary


def main(
    platform: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_days: Optional[Union[str, int, Sequence[int]]] = None,
    workers: Optional[int] = None,
):
    import config

    input_dir = config.WEIBO_INPUT_DIR if platform == "weibo" else config.TWITTER_INPUT_DIR
    filename_pattern = (
        config.WEIBO_FILENAME_PATTERN
        if platform == "weibo"
        else config.TWITTER_FILENAME_PATTERN
    )
    output_path = f"{config.OUTPUT_DIR}/analysis_results/{platform}_opinion_results.csv"
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
    )


if __name__ == "__main__":
    fire.Fire(main)
