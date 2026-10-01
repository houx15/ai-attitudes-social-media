"""
Copy this file to config.py and fill in real values.
config.py is gitignored — never commit real API keys.
"""

# OpenRouter
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash"  # verify exact slug on openrouter.ai/models

# Shared sampling window (day-of-month sampling, applied identically to both platforms)
START_DATE = "2024-03-01"
END_DATE = "2025-03-20"
TARGET_DAYS = [1, 10, 20]
# Sampled days whose data is missing, replaced by a nearby day that was crawled
# instead. Posts keep their actual date (e.g. Weibo's point sits on 2024-02-29).
DATE_SUBSTITUTIONS = {
    "weibo": {"2024-03-01": "2024-02-29", "2024-10-01": "2024-10-02"},
    "twitter": {},
}

# Weibo input: already keyword-filtered per-day files produced by
# youth-analysis/ai_content_extractor.py. This pipeline only reads them.
WEIBO_INPUT_DIR = "YOUR_PATH_HERE"  # e.g. /path/to/youth-analysis/ai_attitudes/ai_weibo_text
WEIBO_FILENAME_PATTERN = "{date}.parquet"

# Twitter input: already crawled per-day files produced by
# twitterapi-io's crawler + convert_to_parquet.py. This pipeline only reads them.
TWITTER_INPUT_DIR = "YOUR_PATH_HERE"  # e.g. /path/to/twitterapi-io/parquet_data
TWITTER_FILENAME_PATTERN = "tweets_{date}.parquet"
# us_userids.json from twitterapi-io/user_location_filter.py. Required:
# `prepare_data.py clean --platform twitter` always keeps only these US users.
TWITTER_US_USERIDS_PATH = "YOUR_PATH_HERE"  # e.g. /path/to/user_location_filter/us_userids.json

# Stage 3 only: the earlier GPT-5-mini Twitter user-level means (uncorrected, unsmoothed;
# columns date, weibo, twitter), added as `twitter-gpt` to the main result's CSV.
TWITTER_GPT_USER_AVG_PATH = None  # e.g. "../0518/user_avg_opinion_comparison_uncorrected_raw_2026-05-18.csv"

# Output roots
OUTPUT_DIR = "output"

# Concurrency / retries for OpenRouter calls
MAX_WORKERS = 32  # requests in flight per process; see README / probe_throughput.py
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # seconds per request; p90 latency is ~2.5s, so this only cuts stragglers
BACKOFF_BASE_SECONDS = 2.0  # retry sleep = (attempt + 1) * BACKOFF_BASE_SECONDS
