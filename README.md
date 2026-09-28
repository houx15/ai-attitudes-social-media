# ai-attitudes-social-media

Unified AI-opinion analysis pipeline for the Weibo (China) / Twitter (US) AI-attitudes
paper. Both platforms are analyzed with the same OpenRouter model and the same prompt
(see `prompts.py`), with reasoning off, temperature 0, and every request pinned to one
upstream provider (DeepInfra, fp8, no fallback), all fixed in `openrouter_client.py`. This
removes the need for the cross-lingual bias correction the old two-model (GPT + Kimi)
setup required. Each result row records the provider that answered, and `clean` prints
the provider breakdown of the labels it used.

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
conda activate opinion                          # the scripts use the active env's python
./run_twitter.sh                                # Stage 1: 4 parallel background tasks (resumable)
tail -f logs/twitter_task*of4.log               # watch; when all tasks have finished:
python prepare_data.py clean --platform twitter # Stage 2: -> OUTPUT_DIR/twitter_daily_opinion.parquet
```

**Weibo server** (`config.py`: key, `WEIBO_INPUT_DIR`, `OUTPUT_DIR`)

```bash
conda activate opinion
./run_weibo.sh
tail -f logs/weibo_task*of4.log
python prepare_data.py clean --platform weibo   # -> OUTPUT_DIR/weibo_daily_opinion.parquet
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
- Reads one day file at a time and keeps at most `MAX_WORKERS × 4` requests queued, so
  memory stays flat for millions of posts (Ctrl-C stops promptly; rerun to resume).
- **Speed = total requests in flight.** Each process sends `MAX_WORKERS` requests at
  once. Measured on 2026-09-26 with the pipeline's settings (DeepInfra pinned,
  `probe_throughput.py`): ~23 posts/s at 32 in flight and ~52/s at 128, no errors.
  Rerun `uv run python probe_throughput.py` from each server before a big run.
- **Parallel tasks.** `./run_twitter.sh [N]` / `./run_weibo.sh [N]` (default N = 4) run
  with the active environment's `python` (e.g. after `conda activate opinion`; with uv,
  `uv run ./run_weibo.sh`; or set `PYTHON=/path/to/python`). They start
  N background tasks with `nohup`, one log each in `logs/`, and refuse to start if that
  platform is already running. Stop with `pkill -f 'run_analysis.py twitter'` (labels so
  far are saved); rerun the script to resume. Under the hood each task is
  `run_analysis.py PLATFORM --task_id i --num_tasks N` (1-based), which SLURM can also
  run: `sbatch --array=1-4 ... --task_id $SLURM_ARRAY_TASK_ID --num_tasks 4`. Days are
  split so each task gets a similar number of posts. Use the same N for all tasks of one
  run.
- **Results** are parquet parts in `OUTPUT_DIR/analysis_results/{platform}_opinion_results/`
  (`task2of4-part000001.parquet`, ...) with columns `id, opinion, prompt_tokens,
  completion_tokens, cached_tokens, provider`. Each task writes a new part every 1,000
  labels and when it stops, so a crash loses at most one part's worth, which is re-sent
  on resume. Every task skips posts with a valid label in **any** part (any earlier run
  or task) and retries failed ones. Older CSV results are not read: move them aside so
  those posts are re-labelled by the pinned provider.
- Shows a progress bar (done/total, speed, ETA, completed/failed, tokens in/out) and
  prints token totals for the run and for all runs so far. Under `nohup ... > log` the
  bar refreshes every 30s.

### Stage 2 (`prepare_data.py clean`)

- Re-reads post metadata (user id, like counts, dates, never the text) from the
  platform's raw input directory one day at a time, so run it on the same server as
  Stage 1.
- Computes three daily metrics: raw mean (`avg_opinion`), like-weighted mean
  (`weighted_opinion`, weight = likes + 1), and user-level mean (`user_avg_opinion`,
  mean of each user's daily mean, the main result).
- Prints a coverage line (metadata rows loaded / matched a label / valid numeric
  opinion) so a partial or mismatched run is visible.
- Twitter is always restricted to US users listed in `TWITTER_US_USERIDS_PATH` (the old
  `--location us`). Stage 1 still labels all tweets, as before.
- Prints and saves the numbers for the paper's data description to
  `OUTPUT_DIR/{platform}_sample_stats.json`:
  - input rows, duplicate ids removed, posts, and unique users;
  - for Twitter, the size of the US user list and how many of those users actually posted;
  - for the analytic subset (US users for Twitter, all posts for Weibo): posts with an
    attitude, "cannot tell", and unlabeled (failed or never sent);
  - the -2..2 distribution as a % of posts with an attitude, and posts per date.

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
