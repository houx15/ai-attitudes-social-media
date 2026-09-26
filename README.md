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
```

Outputs land under `output/`: `analysis_results/{platform}_opinion_results.csv`,
`{platform}_daily_opinion.parquet`, `figure_data.parquet`/`.csv`, `figures/*.pdf`.
