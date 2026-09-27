# ai-attitudes-social-media

Unified AI-opinion analysis pipeline for the Weibo (China) / Twitter (US) AI-attitudes
paper. Both platforms are analyzed with the same OpenRouter model and the same prompt
(see `prompts.py`), removing the need for the cross-lingual bias correction the old
two-model (GPT + Kimi) setup required.

This repo does **not** crawl or keyword-filter data. It reads already-extracted,
per-day parquet files produced by the sibling `youth-analysis` (Weibo) and
`twitterapi-io` (Twitter) repos — configure their paths in `config.py`.

## Setup

```bash
pip install -r requirements.txt
cp config.example.py config.py
# edit config.py: OPENROUTER_API_KEY, WEIBO_INPUT_DIR, TWITTER_INPUT_DIR
```

Run the test suite from the repo root with `pytest` (or `python -m pytest`).

## Smoke test before touching real data

`smoke_test.py` generates ~100 mock Weibo/Twitter posts across 4 sampled days, then runs
the real CLI commands for all three stages against them. It writes a throwaway config
into `smoke_output/` (gitignored), so your own `config.py` is never modified.

```bash
python smoke_test.py          # offline: Stage 1 hits a local fake OpenRouter server (free)
python smoke_test.py --live   # Stage 1 hits the real OpenRouter API using your config.py key/model
```

Offline mode also plants a transient HTTP 500, a non-JSON reply, duplicate and empty
posts, and an off-day file, and checks the daily metrics against independently computed
ground truth. `--live` confirms your key and model slug work and prints how often the
real model's labels agree with the planted intent, plus token usage. It exits non-zero
if any check fails; inspect the figures in `smoke_output/output/figures/`.

## Usage — three independent stages (can run on different machines)

```bash
# Stage 1: AI opinion analysis via OpenRouter (needs network access to OpenRouter)
python run_analysis.py weibo
python run_analysis.py twitter

# Stage 2: clean + compute daily aggregates (raw mean / like-weighted / user-mean) + export figure data
python prepare_data.py clean --platform weibo
python prepare_data.py clean --platform twitter
python prepare_data.py export

# Stage 3: plot (sliding-window smoothing happens here only, figure_data itself stays unsmoothed)
python plot_figures.py
python plot_figures.py --window_size 5
python plot_figures.py --use_smoothing False   # disable smoothing (equivalently: --nouse_smoothing)
```

Stages 1 and 2 both accept `--start_date`, `--end_date` and `--target_days` overrides
(e.g. `--target_days 1,10,20`); if you override them for Stage 1, pass the **same**
values to Stage 2 so both stages cover the same sample. `clean` prints a coverage line
(metadata rows loaded / matched an opinion result / valid numeric opinion) so a
mismatched or partial run is visible.

Twitter is always restricted to US users: `clean --platform twitter` keeps only authors
listed in `TWITTER_US_USERIDS_PATH` (the old `--location us`). Stage 1 still analyzes all
tweets, as before.

Sampled days are the 1st/10th/20th of each month. Where a platform's data is missing on
a nominal day, `DATE_SUBSTITUTIONS` in `config.py` names the nearby day that was crawled
instead (Weibo: 2024-02-29 for 2024-03-01, 2024-10-02 for 2024-10-01). Those posts keep
their actual date, and each platform is smoothed over its own dates, as before.

Everything except the prompt and the LLM caller follows the legacy scripts
(`youth-analysis/ai_sentiment_analyzer.py`, `twitterapi-io/batch_sentiment_analysis.py`,
`twitterapi-io/plot.py`): same input files, dates, dedup, empty-text rules, weights, three
metrics, and figure style and file names.

Stage 2 (`clean`) re-reads post metadata (user id, like counts, dates) through the same
loaders as Stage 1, so the machine running Stage 2 needs read access to **both
platforms' raw input directories** (`WEIBO_INPUT_DIR`, `TWITTER_INPUT_DIR`), not just
Stage 1's `analysis_results/*.csv` outputs.

Outputs land under `output/`: `analysis_results/{platform}_opinion_results.csv`,
`{platform}_daily_opinion.parquet`, `figure_data.parquet`/`.csv`, `figures/*.pdf`.
