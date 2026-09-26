# Unified AI-opinion social media pipeline — design

## Background

The PNAS paper compares public AI attitudes in China (Weibo) and the US (Twitter/X).
Historically the two platforms were analyzed with different models for data-privacy
reasons: Twitter via OpenAI GPT (through OpenAI's Batch API with a hosted prompt,
`twitterapi-io/batch_sentiment_analysis.py`), Weibo via Kimi/Moonshot
(`youth-analysis/ai_content_extractor.py` + `ai_sentiment_analyzer.py`). That
model/prompt asymmetry required a cross-lingual bias correction
(`twitterapi-io/plot.py`'s `CORRECTION_FACTOR = -0.0722`, derived in
`cross_lingual_validation.py`) before the two series were comparable.

This introduces a single new pipeline, in its own repo, that re-runs the opinion
analysis for both platforms through **the same model (`deepseek/deepseek-v4.1-flash`
via OpenRouter) and the same prompt**, eliminating the need for that correction.

## Non-goals

- No crawling or keyword-filtering logic. Both platforms' upstream extraction stays
  exactly as-is (`youth-analysis/ai_content_extractor.py` output, `twitterapi-io`'s
  crawler + `convert_to_parquet.py` output). This pipeline only *reads* their already
  filtered per-day parquet files, via a configurable input path.
- `youth-analysis/` and `twitterapi-io/` are not modified. Both stay as historical
  record and remain usable for re-extraction if ever needed.
- No OpenAI Batch API. Both platforms now use direct concurrent chat-completions calls.

## Repo layout

New sibling repo: `ai_attitudes/ai-attitudes-social-media/` (own git repo).

```
ai-attitudes-social-media/
  config.example.py     # tracked: OpenRouter creds placeholder + per-platform input paths
  config.py             # gitignored: real config (API key, real paths)
  prompts.py            # canonical system prompt + "Post text: {text}" user template
  openrouter_client.py  # concurrent (thread-pool) chat-completions calls, retry/backoff
  loaders.py            # weibo_loader() / twitter_loader() -> standard per-day frame
  run_analysis.py       # STAGE 1 CLI: OpenRouter opinion analysis, resumable
  prepare_data.py       # STAGE 2 CLI: clean + 3 daily aggregates + figure_data export
  plot_figures.py       # STAGE 3 CLI: sliding-window smoothing at plot time, draws PDFs
  requirements.txt
  .gitignore
  README.md
```

Stages are separate command-line invocations by design, so Stage 1 can run on a
different VPS (with OpenRouter network access) than Stage 2/3.

## Canonical prompt (`prompts.py`)

Single prompt, identical for both platforms and both languages — this is the fix for
the bias the old two-model setup had.

**System prompt:**
```
You will read a piece of text posted by a user on social media. The text may be written
in Chinese, English, or any other language — analyze it regardless of the language it is
written in. Please analyze the text's opinion toward AI technology and assign it an
opinion label. Your output must be a JSON object containing only one field, "opinion",
whose value must be one of: -2, -1, 0, 1, 2, or "cannot tell".

Label definitions:
2 = The text expresses a strongly positive attitude toward AI technology, clearly
asserting that the benefits of AI outweigh the harms and that AI brings significant
positive impacts to society (e.g., improving convenience, enhancing health and medical
services, creating economic opportunities, increasing learning or work efficiency,
improving safety, or supporting research and innovation), or expresses reliance on or
trust in AI.
1 = The text is overall positive toward AI, but the attitude is mild or includes
reservations (e.g., expresses support but without strong enthusiasm; believes AI is
"generally beneficial" while also mentioning risks or limitations; expresses
expectation, interest, or positive impressions, but not strong praise).
0 = The text is neutral or difficult to judge (e.g., mentions both pros and cons but
without a clear leaning).
-1 = The text is overall negative toward AI, but the attitude is mild or includes
reservations (e.g., expresses concern or opposition but does not fully reject AI;
believes AI is "risky or harmful" but acknowledges certain benefits; expresses caution,
unease, or negative views, but without strong condemnation).
-2 = The text expresses a strongly negative attitude toward AI technology, clearly
asserting that the harms outweigh the benefits and that AI brings significant negative
impacts to society (e.g., leading to unemployment, contributing to economic bubbles,
increasing privacy risks, reinforcing bias against marginalized groups, generating
misinformation or rumors, or posing safety threats), or expresses resistance to or
distrust in AI.
"cannot tell" = The text does not express any attitude toward AI (e.g., content is
unrelated to AI, or does not reflect a clear viewpoint).

Please return only a JSON object, for example:
{"opinion": 1}
```

**User message (identical template, both platforms):** `Post text: {text}`

No retweet/quote flag and no timestamp in the prompt — Weibo's `is_retweet` signal
was considered (it can flip meaning, e.g. a retweet-with-disapproving-comment) but
Twitter's raw data has no equivalent captured field
(`convert_to_parquet.py` extracts no `isRetweet`/`isQuote`/quoted-text column), and
extending the Twitter extraction was ruled out of scope. Decision: omit the field on
both sides rather than have platforms diverge.

## Data sources (read-only inputs)

| Platform | Input dir (configurable) | Filename pattern | Columns used |
|---|---|---|---|
| Weibo | `youth-analysis`'s `ai_attitudes/ai_weibo_text/` | `{date}.parquet` | `weibo_id`, `user_id`, `weibo_content`, `zan`, (date from filename) |
| Twitter | `twitterapi-io`'s `parquet_data/` | `tweets_{date}.parquet` | `id`, `text`, `likeCount`, `author.id`, `createdAt` |

`loaders.py` exposes one function per platform returning a standard frame:
`{id, text, user_id, weight_raw, date}` — `weight_raw` is `zan`/`likeCount` (weight
used later is `weight_raw + 1`, matching existing "+1 to avoid zero" convention).

**Date assignment (kept as each platform's current, already-correct logic — do not
unify to a single method):**
- Weibo: `date` = the source file's date (already correctly bucketed upstream by
  `ai_content_extractor.py`'s daily partitioning). Do **not** re-derive from
  `time_stamp` — its timezone convention is ambiguous and the existing pipeline
  never actually applies it for date bucketing (the +8 conversion in
  `ai_sentiment_analyzer.py` is dead/commented-out code).
- Twitter: `date` = parsed from the `createdAt` field (`"%a %b %d %H:%M:%S +0000 %Y"`,
  UTC), same as `batch_sentiment_analysis.py` does today.

**Day sampling:** both platforms currently analyze only day 1/10/20 of each month
(Twitter via `TARGET_DAYS` config; Weibo via a hardcoded per-group date-triple list
that happens to also always be 1/10/20). The new pipeline generalizes this into one
shared `START_DATE` / `END_DATE` / `TARGET_DAYS` config applied identically to file
discovery for both platforms.

## Config (`config.example.py`, tracked; `config.py`, gitignored)

```python
# OpenRouter
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY_HERE"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash"  # verify exact slug on openrouter.ai/models

# Shared sampling window
START_DATE = "2024-03-01"
END_DATE = "2025-03-20"
TARGET_DAYS = [1, 10, 20]

# Weibo input (already keyword-filtered per-day files from youth-analysis)
WEIBO_INPUT_DIR = "YOUR_PATH_HERE"          # e.g. .../youth-analysis/ai_attitudes/ai_weibo_text
WEIBO_FILENAME_PATTERN = "{date}.parquet"

# Twitter input (already crawled per-day files from twitterapi-io)
TWITTER_INPUT_DIR = "YOUR_PATH_HERE"        # e.g. .../twitterapi-io/parquet_data
TWITTER_FILENAME_PATTERN = "tweets_{date}.parquet"

# Output roots
OUTPUT_DIR = "output"

# Concurrency / retries for OpenRouter calls
MAX_WORKERS = 8
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
```

## Stage 1 — `run_analysis.py`

```
python run_analysis.py --platform weibo   [--start_date ... --end_date ... --target_days 1,10,20 --workers 8]
python run_analysis.py --platform twitter [--start_date ... --end_date ... --target_days 1,10,20 --workers 8]
```

- Discovers per-day input files in range matching `TARGET_DAYS`, via the matching
  platform loader.
- Skips ids already present in `output/analysis_results/{platform}_opinion_results.csv`
  (resumable, matching the existing processed-id-cache pattern in both legacy
  scripts).
- Fires `MAX_WORKERS` concurrent OpenRouter chat-completions calls (thread pool),
  each with retry/exponential backoff on failure (mirrors
  `ai_sentiment_analyzer.analyze_single`'s retry logic).
- Appends each result as it completes to
  `output/analysis_results/{platform}_opinion_results.csv`
  (`id, opinion, prompt_tokens, completion_tokens, cached_tokens`) — crash-safe.
- Prints a running opinion-distribution + token-usage summary at the end.

## Stage 2 — `prepare_data.py`

```
python prepare_data.py clean --platform weibo
python prepare_data.py clean --platform twitter
python prepare_data.py export
```

`clean --platform X`:
- Reloads the platform's per-day metadata via the same loader used in Stage 1
  (`id, user_id, weight_raw, date`), merges with
  `{platform}_opinion_results.csv` on `id`.
- Drops rows with missing/`"cannot tell"` opinion; coerces `opinion` to numeric.
- Computes three daily aggregates, all three kept:
  - `avg_opinion` — raw mean opinion per date.
  - `weighted_opinion` — `(weight_raw + 1)`-weighted mean opinion per date.
  - `user_avg_opinion` — mean of per-user daily-mean opinions per date. **Primary
    result metric.**
- Saves `output/{platform}_daily_opinion.parquet` (`date, avg_opinion,
  weighted_opinion, user_avg_opinion`).

`export`:
- Loads both platforms' daily-opinion parquets, outer-joins on `date`.
- Saves `output/figure_data.parquet` / `.csv` with columns
  `date, weibo_avg_opinion, twitter_avg_opinion, weibo_weighted_opinion,
  twitter_weighted_opinion, weibo_user_avg_opinion, twitter_user_avg_opinion`.
- **No smoothing, no correction factor** — this is the raw, comparable daily series
  per your instruction ("export the figure data (not sliding windowed)").

## Stage 3 — `plot_figures.py`

```
python plot_figures.py --window_size 3 [--no_smoothing]
```

- Loads `output/figure_data.parquet`.
- For each of the three metrics, applies sliding-window smoothing (default 3-day,
  centered, matching current `plot.py`) **only at plot time** — the canonical figure
  data on disk stays unsmoothed.
- Draws the same visual style as today's `plot.py`: Weibo (`#ff7333`) vs. Twitter
  (`#20AEE6`) lines, "AI benefits"/"AI concerns" annotations, `y=0` neutral dashed
  line for the weighted metric, month-formatted x-axis.
- Saves one PDF per metric to `output/figures/`, plus a companion CSV of the
  plotted (smoothed) points for reproducibility — same convention as today.
- `plot_four_line.py`'s original-vs-translated cross-lingual-correction diagnostic is
  **dropped** — it existed specifically to reconcile the two-model bias, which no
  longer exists once both platforms share model + prompt.

## Open items carried forward (not blocking, but worth re-confirming during review)

- Exact OpenRouter model slug for "deepseek-v4.1-flash" — placeholder value in
  `config.example.py`; verify against `openrouter.ai/models` before first real run.
- `MAX_WORKERS` / retry tuning will need adjustment once real rate limits are known.
