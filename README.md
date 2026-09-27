# ai-attitudes-social-media

Unified AI-opinion analysis pipeline for the Weibo (China) / Twitter (US) AI-attitudes
paper. Both platforms are analyzed with the same OpenRouter model and the same prompt
(see `prompts.py`), with reasoning turned off (`reasoning: {effort: none}`). This
removes the need for the cross-lingual bias correction the old two-model (GPT + Kimi)
setup required.

This repo does **not** crawl or keyword-filter data. It reads already-extracted,
per-day parquet files produced by the sibling `youth-analysis` (Weibo) and
`twitterapi-io` (Twitter) repos — configure their paths in `config.py`.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/). `uv sync` installs
Python 3.12 (from `.python-version`) and the exact versions in `uv.lock` into `.venv/`.

```bash
uv sync
cp config.example.py config.py
# edit config.py: OPENROUTER_API_KEY, OPENROUTER_MODEL, WEIBO_INPUT_DIR, TWITTER_INPUT_DIR,
# TWITTER_US_USERIDS_PATH, OUTPUT_DIR
```

Run the test suite with `uv run pytest`. Every command below runs inside the environment
via `uv run`.

On a server in mainland China, if downloads are slow, point uv at mirrors, e.g.
`UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` for packages and
`UV_PYTHON_INSTALL_MIRROR=<mirror>` for the Python download (or use a local Python with
`uv sync --python /path/to/python3.12`).

## Smoke test before touching real data

`smoke_test.py` generates ~100 mock Weibo/Twitter posts across 4 sampled days, then runs
the real CLI commands for all three stages against them. It writes a throwaway config
into `smoke_output/` (gitignored), so your own `config.py` is never modified.

```bash
uv run python smoke_test.py          # offline: Stage 1 hits a local fake OpenRouter server (free)
uv run python smoke_test.py --live   # Stage 1 hits the real OpenRouter API using your config.py key/model
```

Offline mode also plants a transient HTTP 500, a non-JSON reply, duplicate and empty
posts, and an off-day file, and checks the daily metrics against independently computed
ground truth. `--live` confirms your key and model slug work and prints how often the
real model's labels agree with the planted intent, plus token usage, and fails if the
average output is over 50 tokens per post (a sign reasoning is still on). It exits non-zero
if any check fails; inspect the figures in `smoke_output/output/figures/`.

## Usage — Twitter server, Weibo server, then your own machine

Twitter and Weibo data live on different servers. Each server labels and aggregates its
own platform; your machine merges the two small daily files and plots.

**Twitter server** (`config.py`: key, `TWITTER_INPUT_DIR`, `TWITTER_US_USERIDS_PATH`, `OUTPUT_DIR`)

```bash
uv run python run_analysis.py twitter                  # Stage 1: label tweets via OpenRouter (resumable)
uv run python prepare_data.py clean --platform twitter  # Stage 2: -> OUTPUT_DIR/twitter_daily_opinion.parquet
```

**Weibo server** (`config.py`: key, `WEIBO_INPUT_DIR`, `OUTPUT_DIR`)

```bash
uv run python run_analysis.py weibo
uv run python prepare_data.py clean --platform weibo    # -> OUTPUT_DIR/weibo_daily_opinion.parquet
```

**Your machine** (`config.py`: only `OUTPUT_DIR` is read). Download both
`*_daily_opinion.parquet` files into `OUTPUT_DIR`, then:

```bash
uv run python prepare_data.py export   # -> figure_data.parquet / .csv (both platforms, unsmoothed)
uv run python plot_figures.py          # -> figures/*_comparison_smoothed3d_<date>.pdf (+ plotted points .csv)
uv run python plot_figures.py --window_size 5
uv run python plot_figures.py --use_smoothing False   # -> *_raw_<date>.pdf (equivalently: --nouse_smoothing)
```

Don't point `OUTPUT_DIR` at the old `twitterapi-io/sentiment_results`: the daily files
share the legacy names and would overwrite the published results.

### Stage 1 (`run_analysis.py`)

- Needs network access to `openrouter.ai`; stages 2–3 run offline.
- Resumable: rerunning skips posts that already have a valid label and retries failed
  ones. Results append to `OUTPUT_DIR/analysis_results/{platform}_opinion_results.csv`.
- Shows a progress bar (done/total, speed, ETA, completed/failed, tokens in/out) and
  prints token totals for the run and for all runs so far. Under `nohup ... > log` the
  bar refreshes every 30s.

### Stage 2 (`prepare_data.py clean`)

- Re-reads post metadata (user id, like counts, dates) from the platform's raw input
  directory, so run it on the same server as Stage 1.
- Computes three daily metrics: raw mean (`avg_opinion`), like-weighted mean
  (`weighted_opinion`, weight = likes + 1), and user-level mean (`user_avg_opinion`,
  mean of each user's daily mean, the main result).
- Prints a coverage line (metadata rows loaded / matched a label / valid numeric
  opinion) so a partial or mismatched run is visible.
- Twitter is always restricted to US users listed in `TWITTER_US_USERIDS_PATH` (the old
  `--location us`). Stage 1 still labels all tweets, as before.

### Dates

Sampled days are the 1st/10th/20th of each month between `START_DATE` and `END_DATE`.
Where a platform's data is missing on a nominal day, `DATE_SUBSTITUTIONS` names the
nearby day that was crawled instead (Weibo: 2024-02-29 for 2024-03-01, 2024-10-02 for
2024-10-01). Those posts keep their actual date, and each platform is smoothed over its
own dates, as before. Stages 1 and 2 accept `--start_date`, `--end_date` and
`--target_days` overrides (e.g. `--target_days 1,10,20`); pass the **same** values to
both stages.

### Consistency with the legacy code

Everything except the prompt and the LLM caller follows the legacy scripts
(`youth-analysis/ai_sentiment_analyzer.py`, `twitterapi-io/batch_sentiment_analysis.py`,
`twitterapi-io/plot.py`): same input files, dates, dedup, empty-text rules, weights,
three metrics, and figure style and file names.
