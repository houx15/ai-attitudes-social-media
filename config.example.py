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

# Weibo input: already keyword-filtered per-day files produced by
# youth-analysis/ai_content_extractor.py. This pipeline only reads them.
WEIBO_INPUT_DIR = "YOUR_PATH_HERE"  # e.g. /path/to/youth-analysis/ai_attitudes/ai_weibo_text
WEIBO_FILENAME_PATTERN = "{date}.parquet"

# Twitter input: already crawled per-day files produced by
# twitterapi-io's crawler + convert_to_parquet.py. This pipeline only reads them.
TWITTER_INPUT_DIR = "YOUR_PATH_HERE"  # e.g. /path/to/twitterapi-io/parquet_data
TWITTER_FILENAME_PATTERN = "tweets_{date}.parquet"

# Output roots
OUTPUT_DIR = "output"

# Concurrency / retries for OpenRouter calls
MAX_WORKERS = 8
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
