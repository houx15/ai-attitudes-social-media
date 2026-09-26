# plot_figures.py
"""Stage 3 CLI: plot the Weibo-vs-Twitter comparison figures.

figure_data on disk stays unsmoothed; sliding-window smoothing happens only
here, at plot time, and is never written back.

Usage:
    python plot_figures.py
    python plot_figures.py --window_size 5
    python plot_figures.py --use_smoothing False
"""

from pathlib import Path
from typing import Optional

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

WEIBO_COLOR = "#ff7333"
TWITTER_COLOR = "#20AEE6"

METRICS = [
    ("avg_opinion", "Average Opinion"),
    ("weighted_opinion", "LikeCount Weighted Opinion"),
    ("user_avg_opinion", "User-level Average Opinion"),
]


def apply_sliding_window(df: pd.DataFrame, metric: str, window_size: int = 3) -> pd.Series:
    df = df.sort_values("date")
    return df[metric].rolling(window=window_size, center=True, min_periods=1).mean()


def plot_metric(
    ax,
    figure_df: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    window_size: int = 3,
    use_smoothing: bool = True,
) -> pd.DataFrame:
    df = figure_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    weibo_col = f"weibo_{metric_name}"
    twitter_col = f"twitter_{metric_name}"

    if use_smoothing:
        weibo_values = apply_sliding_window(
            df.rename(columns={weibo_col: metric_name}), metric_name, window_size
        )
        twitter_values = apply_sliding_window(
            df.rename(columns={twitter_col: metric_name}), metric_name, window_size
        )
    else:
        weibo_values = df[weibo_col]
        twitter_values = df[twitter_col]

    ax.plot(df["date"], weibo_values, color=WEIBO_COLOR, linewidth=5, alpha=0.7, label="Weibo, China")
    ax.plot(df["date"], twitter_values, color=TWITTER_COLOR, linewidth=5, alpha=0.7, label="Twitter, USA")

    if metric_name == "weighted_opinion":
        ax.axhline(y=0, color="grey", linestyle="--", linewidth=2, zorder=0)

    # "AI benefits" / "AI concerns" annotations near the top/bottom of the
    # y-range, at the left edge — same positioning as the legacy plot.py.
    all_values = pd.concat([weibo_values, twitter_values]).dropna()
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

    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    return pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d").values,
            "weibo": weibo_values.values,
            "twitter": twitter_values.values,
        }
    )


def main(
    figure_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    window_size: int = 3,
    use_smoothing: bool = True,
):
    if figure_data_path is None or output_dir is None:
        import config

        figure_data_path = figure_data_path or f"{config.OUTPUT_DIR}/figure_data.parquet"
        output_dir = output_dir or f"{config.OUTPUT_DIR}/figures"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    figure_df = pd.read_parquet(figure_data_path)

    for metric_name, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        points = plot_metric(
            ax, figure_df, metric_name, ylabel, window_size=window_size, use_smoothing=use_smoothing
        )
        plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.15)

        output_path = Path(output_dir) / f"{metric_name}_comparison.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

        points.to_csv(output_path.with_suffix(".csv"), index=False)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
