# plot_figures.py
"""Stage 3 CLI: plot the Weibo-vs-Twitter comparison figures.

figure_data on disk stays unsmoothed; sliding-window smoothing happens only
here, at plot time, and is never written back.

Usage:
    python plot_figures.py
    python plot_figures.py --window_size 5
    python plot_figures.py --use_smoothing False
    python plot_figures.py --twitter_gpt_path ../0518/user_avg_opinion_comparison_uncorrected_raw_2026-05-18.csv

Each PDF gets a CSV of the unsmoothed values behind it. The main result
(user-level mean) also carries the earlier GPT-5-mini Twitter series
(uncorrected, unsmoothed) as `twitter-gpt` when its CSV is given, and a
fourth figure draws it as a dashed third line for comparison.
"""

import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

WEIBO_COLOR = "tab:orange"
TWITTER_COLOR = "tab:blue"
LINE_WIDTH = 5
FIGSIZE = (6.4, 4.8)
TICK_STEP = 0.5
SCALE_END = 2  # the attitude scale runs -2 (concerned) .. 2 (excited)

METRICS = [
    ("avg_opinion", "Post-level attitude toward AI"),
    ("weighted_opinion", "Like-weighted attitude toward AI"),
    ("user_avg_opinion", "Social media attitude toward AI"),
]


def place_legend(ax) -> None:
    """Frameless legend at the top left, as in the mentor's replot; if it would
    cover a line there, let matplotlib pick the emptiest spot instead."""
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.04, 0.94), frameon=False, handlelength=3)
    ax.figure.canvas.draw()
    box = legend.get_window_extent()
    for line in ax.get_lines():
        if not line.get_label().startswith("_") and len(line.get_xdata()):
            points = line.get_transform().transform(line.get_xydata())
            if box.count_contains(points):
                ax.legend(loc="best", frameon=False, handlelength=3)
                return


def draw_attitude_axis(ax, data_min: float, data_max: float) -> None:
    """Y-axis zoomed on the data, ticks every 0.5, with the scale ends (-2
    "Concerned", 2 "Excited") shown past a break mark, and 0 marked "Neutral"."""
    # A line may poke slightly past the outer tick (e.g. -0.01 under 0.0).
    slack = 0.1 * TICK_STEP
    low = min(math.floor((data_min + slack) / TICK_STEP) * TICK_STEP, 0.0)
    high = max(math.ceil((data_max - slack) / TICK_STEP) * TICK_STEP, 0.0)
    gap = 0.1 * (high - low or 1.0)  # room between the data range and a scale end

    ticks = [low + i * TICK_STEP for i in range(round((high - low) / TICK_STEP) + 1)]
    labels = [f"{t:.1f}" for t in ticks]
    breaks = []
    bottom, top = low, high
    if low > -SCALE_END:
        bottom = low - gap
        ticks.insert(0, bottom)
        labels.insert(0, f"{-SCALE_END:.1f}")
        breaks.append(low - gap / 2)
    if high < SCALE_END:
        top = high + gap
        ticks.append(top)
        labels.append(f"{SCALE_END:.1f}")
        breaks.append(high + gap / 2)
    pad = 0.03 * (top - bottom)
    ax.set_ylim(bottom - pad, top + pad)
    ax.set_yticks(ticks, labels)

    # The left spine is redrawn in pieces so it can show a // at each break.
    ax.spines["left"].set_visible(False)
    trans = ax.get_yaxis_transform()  # x in axes units, y in data units
    half = 0.012 * (top - bottom + 2 * pad)
    edges = [bottom - pad] + [y for b in sorted(breaks) for y in (b - half, b + half)] + [top]
    for y0, y1 in zip(edges[::2], edges[1::2]):
        ax.plot([0, 0], [y0, y1], color="black", linewidth=0.8, transform=trans, clip_on=False)
    for b in breaks:
        for y in (b - half, b + half):
            ax.plot([-0.012, 0.012], [y - half / 2, y + half / 2], color="black", linewidth=0.8,
                    transform=trans, clip_on=False)

    for y, text in ((top if high < SCALE_END else SCALE_END, "Excited"), (0.0, "Neutral"),
                    (bottom if low > -SCALE_END else -SCALE_END, "Concerned")):
        if bottom <= y <= top:
            ax.text(0.012, y, text, color="grey", transform=trans, va="center", ha="left")


def apply_sliding_window(series: pd.Series, window_size: int = 3) -> pd.Series:
    """Centered moving average over a date-sorted series (legacy plot.py)."""
    return series.rolling(window=window_size, center=True, min_periods=1).mean()


def plot_metric(
    ax,
    figure_df: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    window_size: int = 3,
    use_smoothing: bool = True,
    twitter_gpt: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Draw Weibo vs Twitter; with `twitter_gpt` (unsmoothed, indexed by
    yyyy-mm-dd), also draw the earlier GPT-5-mini Twitter series as a dashed line."""
    df = figure_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    weibo_col = f"weibo_{metric_name}"
    twitter_col = f"twitter_{metric_name}"

    # Each platform keeps only its own dates: substituted days leave the other
    # platform empty on that row, and a window over merged rows would invent values.
    weibo_values = df[["date", weibo_col]].dropna().set_index("date")[weibo_col]
    twitter_values = df[["date", twitter_col]].dropna().set_index("date")[twitter_col]
    weibo_unsmoothed, twitter_unsmoothed = weibo_values, twitter_values

    # Sliding window happens here, at draw time only; figure_data stays unsmoothed.
    if use_smoothing:
        weibo_values = apply_sliding_window(weibo_values, window_size)
        twitter_values = apply_sliding_window(twitter_values, window_size)

    gpt_values = gpt_unsmoothed = None
    if twitter_gpt is not None:
        gpt_unsmoothed = twitter_gpt.dropna()
        gpt_unsmoothed.index = pd.to_datetime(gpt_unsmoothed.index)
        gpt_unsmoothed = gpt_unsmoothed.sort_index()
        gpt_values = apply_sliding_window(gpt_unsmoothed, window_size) if use_smoothing else gpt_unsmoothed

    # Style of the mentor's replot: thick solid lines, GPT dotted, frameless legend.
    model_tag = ", DeepSeek" if gpt_values is not None else ""
    ax.plot(weibo_values.index, weibo_values.values, color=WEIBO_COLOR, linewidth=LINE_WIDTH,
            label=f"China (Weibo{model_tag})")
    ax.plot(twitter_values.index, twitter_values.values, color=TWITTER_COLOR, linewidth=LINE_WIDTH,
            label=f"United States (Twitter{model_tag})")
    if gpt_values is not None:
        ax.plot(gpt_values.index, gpt_values.values, color=TWITTER_COLOR, linewidth=LINE_WIDTH,
                linestyle=":", label="United States (Twitter, GPT)")

    all_values = pd.concat([v for v in (weibo_values, twitter_values, gpt_values) if v is not None])
    if len(all_values) > 0:
        draw_attitude_axis(ax, all_values.min(), all_values.max())

    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 7, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.spines[["top", "right"]].set_visible(False)
    place_legend(ax)

    points = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d").values,
            "weibo-deepseek": weibo_unsmoothed.reindex(df["date"]).values,
            "twitter-deepseek": twitter_unsmoothed.reindex(df["date"]).values,
        }
    )
    if gpt_unsmoothed is not None:
        points["twitter-gpt"] = gpt_unsmoothed.reindex(df["date"]).values
    return points


def main(
    figure_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    window_size: int = 3,
    use_smoothing: bool = True,
    twitter_gpt_path: Optional[str] = None,
):
    if figure_data_path is None or output_dir is None:
        import config

        figure_data_path = figure_data_path or f"{config.OUTPUT_DIR}/figure_data.parquet"
        output_dir = output_dir or f"{config.OUTPUT_DIR}/figures"
        twitter_gpt_path = twitter_gpt_path or getattr(config, "TWITTER_GPT_USER_AVG_PATH", None)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    figure_df = pd.read_parquet(figure_data_path)

    twitter_gpt = None
    if twitter_gpt_path:
        # Earlier GPT-5-mini run (uncorrected, unsmoothed): columns date, weibo, twitter.
        gpt_df = pd.read_csv(twitter_gpt_path, dtype={"date": str})
        twitter_gpt = gpt_df.set_index("date")["twitter"]
    else:
        print("No twitter_gpt_path given: the user_avg_opinion CSV will have no twitter-gpt column")

    # Date first (yyyy-mm-dd-name), minus the retired cross-lingual correction tag.
    smoothing_tag = f"_smoothed{window_size}d" if use_smoothing else "_raw"
    today_str = datetime.now().strftime("%Y-%m-%d")

    figures = [(metric_name, ylabel, metric_name, None) for metric_name, ylabel in METRICS]
    if twitter_gpt is not None:
        # Comparison figure: the main result plus the earlier GPT-5-mini Twitter line (dashed).
        figures.append(("user_avg_opinion", dict(METRICS)["user_avg_opinion"], "user_avg_opinion_with_gpt", twitter_gpt))

    for metric_name, ylabel, name, gpt_line in figures:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        points = plot_metric(
            ax, figure_df, metric_name, ylabel, window_size=window_size, use_smoothing=use_smoothing,
            twitter_gpt=gpt_line,
        )

        output_path = Path(output_dir) / f"{today_str}-{name}_comparison{smoothing_tag}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        if metric_name == "user_avg_opinion" and twitter_gpt is not None and gpt_line is None:
            points["twitter-gpt"] = twitter_gpt.reindex(points["date"]).values
        points.to_csv(output_path.with_suffix(".csv"), index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
