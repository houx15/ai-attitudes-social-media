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

from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.size"] = 12

WEIBO_COLOR = "#ff7333"
TWITTER_COLOR = "#20AEE6"

METRICS = [
    ("avg_opinion", "Average Opinion"),
    ("weighted_opinion", "LikeCount Weighted Opinion"),
    ("user_avg_opinion", "User-level Average Opinion"),
]


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

    model_tag = " (DeepSeek)" if gpt_values is not None else ""
    ax.plot(weibo_values.index, weibo_values.values, color=WEIBO_COLOR, linewidth=5, alpha=0.7, label=f"Weibo, China{model_tag}")
    ax.plot(twitter_values.index, twitter_values.values, color=TWITTER_COLOR, linewidth=5, alpha=0.7, label=f"Twitter, USA{model_tag}")
    if gpt_values is not None:
        ax.plot(gpt_values.index, gpt_values.values, color=TWITTER_COLOR, linewidth=3, linestyle="--", label="Twitter, USA (GPT-5-mini)")

    if metric_name == "weighted_opinion":
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=2, zorder=0)

    # "AI benefits" / "AI concerns" annotations near the top/bottom of the
    # y-range, at the left edge — same positioning as the legacy plot.py.
    all_values = pd.concat([v for v in (weibo_values, twitter_values, gpt_values) if v is not None])
    if len(all_values) > 0:
        x_min = df["date"].min()
        x_max = df["date"].max()
        x_min = x_min - (x_max - x_min) * 0.03

        y_max = all_values.max()
        y_min = all_values.min()
        y_range = y_max - y_min
        y_padding = max(0.1 * y_range, 0.1)  # at least 10% padding or 0.1 unit
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        y_top = y_max + 0.05 * y_range
        y_bottom = y_min - 0.05 * y_range
        ax.text(
            x_min, y_top, "AI benefits", fontsize=10, color="black",
            verticalalignment="top", horizontalalignment="left",
        )
        ax.text(
            x_min, y_bottom, "AI concerns", fontsize=10, color="black",
            verticalalignment="bottom", horizontalalignment="left",
        )
        if metric_name == "weighted_opinion":
            ax.text(
                x_min, 0.05, "neutral", fontsize=10, color="grey",
                verticalalignment="center", horizontalalignment="left",
            )

    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

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
        fig, ax = plt.subplots(figsize=(10, 6))
        points = plot_metric(
            ax, figure_df, metric_name, ylabel, window_size=window_size, use_smoothing=use_smoothing,
            twitter_gpt=gpt_line,
        )
        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.15)

        output_path = Path(output_dir) / f"{today_str}-{name}_comparison{smoothing_tag}.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        if metric_name == "user_avg_opinion" and twitter_gpt is not None and gpt_line is None:
            points["twitter-gpt"] = twitter_gpt.reindex(points["date"]).values
        points.to_csv(output_path.with_suffix(".csv"), index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
