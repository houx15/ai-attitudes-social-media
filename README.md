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

Stage 2 (`clean`) re-reads post metadata (user id, like counts, dates) through the same
loaders as Stage 1, so the machine running Stage 2 needs read access to **both
platforms' raw input directories** (`WEIBO_INPUT_DIR`, `TWITTER_INPUT_DIR`), not just
Stage 1's `analysis_results/*.csv` outputs.

Outputs land under `output/`: `analysis_results/{platform}_opinion_results.csv`,
`{platform}_daily_opinion.parquet`, `figure_data.parquet`/`.csv`, `figures/*.pdf`.
