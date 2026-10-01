# tests/test_plot_figures.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from datetime import date

from plot_figures import apply_sliding_window, main, plot_metric


def test_apply_sliding_window_centered_mean():
    series = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2024-03-01", periods=5))
    smoothed = apply_sliding_window(series, window_size=3)
    assert smoothed.tolist() == pytest.approx([1.5, 2.0, 3.0, 4.0, 4.5])


def test_plot_metric_without_smoothing_returns_raw_two_lines():
    figure_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-10"],
        "weibo_avg_opinion": [0.5, 1.0],
        "twitter_avg_opinion": [-0.5, -1.0],
    })
    fig, ax = plt.subplots()
    points = plot_metric(
        ax, figure_df, "avg_opinion", "Average Opinion", use_smoothing=False
    )
    plt.close(fig)

    assert list(points.columns) == ["date", "weibo-deepseek", "twitter-deepseek"]
    assert points["weibo-deepseek"].tolist() == pytest.approx([0.5, 1.0])
    assert points["twitter-deepseek"].tolist() == pytest.approx([-0.5, -1.0])


def test_main_writes_one_pdf_and_csv_per_metric(tmp_path):
    nan = float("nan")
    figure_df = pd.DataFrame({
        "date": pd.date_range("2024-03-01", periods=5).strftime("%Y-%m-%d"),
        "weibo_avg_opinion": [0.1, 0.2, 0.3, 0.4, 0.5],
        "twitter_avg_opinion": [-0.1, -0.2, -0.3, -0.4, -0.5],
        "weibo_weighted_opinion": [0.1, 0.2, 0.3, 0.4, 0.5],
        "twitter_weighted_opinion": [-0.1, -0.2, -0.3, -0.4, -0.5],
        "weibo_user_avg_opinion": [0.1, 0.2, 0.3, 0.4, 0.5],
        "twitter_user_avg_opinion": [-0.1, -0.2, -0.3, -0.4, -0.5],
    })
    figure_data_path = tmp_path / "figure_data.parquet"
    figure_df.to_parquet(figure_data_path, index=False)
    output_dir = tmp_path / "figures"

    # Earlier GPT-5-mini run: only its Twitter column is used, and only for user_avg_opinion.
    gpt_path = tmp_path / "gpt.csv"
    pd.DataFrame({
        "date": ["2024-03-01", "2024-03-03"],
        "weibo": [9.0, 9.0],
        "twitter": [0.7, 0.8],
    }).to_csv(gpt_path, index=False)

    main(figure_data_path=str(figure_data_path), output_dir=str(output_dir), twitter_gpt_path=str(gpt_path))

    for metric_name in ["avg_opinion", "weighted_opinion", "user_avg_opinion"]:
        stem = f"{date.today():%Y-%m-%d}-{metric_name}_comparison_smoothed3d"
        pdf_path = output_dir / f"{stem}.pdf"
        csv_path = output_dir / f"{stem}.csv"
        assert pdf_path.exists()
        assert csv_path.exists()
        saved = pd.read_csv(csv_path)
        # unsmoothed values only, even though the figure is smoothed
        assert saved["weibo-deepseek"].tolist() == pytest.approx(figure_df[f"weibo_{metric_name}"].tolist())
        if metric_name == "user_avg_opinion":
            assert list(saved.columns) == ["date", "weibo-deepseek", "twitter-deepseek", "twitter-gpt"]
            assert saved["twitter-gpt"].tolist() == pytest.approx([0.7, nan, 0.8, nan, nan], nan_ok=True)
        else:
            assert list(saved.columns) == ["date", "weibo-deepseek", "twitter-deepseek"]

    # Fourth figure: same main result plus the GPT line, same unsmoothed CSV.
    stem = f"{date.today():%Y-%m-%d}-user_avg_opinion_with_gpt_comparison_smoothed3d"
    assert (output_dir / f"{stem}.pdf").exists()
    saved = pd.read_csv(output_dir / f"{stem}.csv")
    assert list(saved.columns) == ["date", "weibo-deepseek", "twitter-deepseek", "twitter-gpt"]
    assert saved["twitter-gpt"].tolist() == pytest.approx([0.7, nan, 0.8, nan, nan], nan_ok=True)


def test_main_without_gpt_path_writes_only_the_three_figures(tmp_path):
    figure_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-10"],
        **{f"{p}_{m}": [0.1, 0.2] for p in ("weibo", "twitter")
           for m in ("avg_opinion", "weighted_opinion", "user_avg_opinion")},
    })
    figure_data_path = tmp_path / "figure_data.parquet"
    figure_df.to_parquet(figure_data_path, index=False)

    main(figure_data_path=str(figure_data_path), output_dir=str(tmp_path / "figures"))

    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 3


def test_plot_metric_draws_gpt_twitter_as_a_dotted_third_line():
    nan = float("nan")
    figure_df = pd.DataFrame({
        "date": ["2024-02-29", "2024-03-01", "2024-03-10", "2024-03-20"],
        "weibo_user_avg_opinion": [1.0, nan, 3.0, 5.0],
        "twitter_user_avg_opinion": [nan, 0.0, 2.0, 4.0],
    })
    gpt = pd.Series([1.0, 2.0, 6.0], index=["2024-03-01", "2024-03-10", "2024-03-20"])
    fig, ax = plt.subplots()
    points = plot_metric(ax, figure_df, "user_avg_opinion", "label", window_size=3, twitter_gpt=gpt)
    labels = [line.get_label() for line in ax.lines]
    styles = [line.get_linestyle() for line in ax.lines]
    gpt_y = list(ax.lines[2].get_ydata())
    plt.close(fig)

    assert labels[:3] == [
        "China (Weibo, DeepSeek)", "United States (Twitter, DeepSeek)", "United States (Twitter, GPT)"
    ]
    assert styles[:3] == ["-", "-", ":"]
    # smoothed over its own dates, like the other lines
    assert gpt_y == pytest.approx([1.5, 3.0, 4.0])
    # the CSV keeps it unsmoothed
    assert points["twitter-gpt"].tolist() == pytest.approx([nan, 1.0, 2.0, 6.0], nan_ok=True)


def _y_ticks(ax):
    return [(round(t, 3), label.get_text()) for t, label in zip(ax.get_yticks(), ax.get_yticklabels())]


def test_attitude_axis_zooms_on_data_and_shows_scale_ends_past_breaks():
    # Mentor's style: ticks every 0.5 around the data, -2 "Concerned" / 2 "Excited"
    # past a break, 0 marked "Neutral". A -0.01 dip does not add a -0.5 tick.
    figure_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-10", "2024-03-20"],
        "weibo_user_avg_opinion": [0.5, 0.6, 0.87],
        "twitter_user_avg_opinion": [0.2, -0.01, 0.3],
    })
    fig, ax = plt.subplots()
    plot_metric(ax, figure_df, "user_avg_opinion", "label", use_smoothing=False)
    ticks = _y_ticks(ax)
    texts = {t.get_text(): t.get_position()[1] for t in ax.texts}
    plt.close(fig)

    assert [label for _, label in ticks] == ["-2.0", "0.0", "0.5", "1.0", "2.0"]
    assert [t for t, _ in ticks][1:4] == [0.0, 0.5, 1.0]
    assert texts["Neutral"] == 0.0
    assert texts["Excited"] == ticks[-1][0] and texts["Concerned"] == ticks[0][0]
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()


def test_attitude_axis_covers_negative_values():
    figure_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-10"],
        "weibo_weighted_opinion": [0.9, 1.2],
        "twitter_weighted_opinion": [-0.6, 0.3],
    })
    fig, ax = plt.subplots()
    plot_metric(ax, figure_df, "weighted_opinion", "label", use_smoothing=False)
    labels = [label for _, label in _y_ticks(ax)]
    plt.close(fig)

    assert labels == ["-2.0", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0"]


def test_plot_metric_smooths_each_platform_over_its_own_dates_only():
    # A substituted day (Weibo on 02-29, Twitter on 03-01) leaves one platform
    # empty on each of those rows; smoothing must not invent values there.
    nan = float("nan")
    figure_df = pd.DataFrame({
        "date": ["2024-02-29", "2024-03-01", "2024-03-10", "2024-03-20"],
        "weibo_avg_opinion": [1.0, nan, 3.0, 5.0],
        "twitter_avg_opinion": [nan, 0.0, 2.0, 4.0],
    })
    fig, ax = plt.subplots()
    points = plot_metric(ax, figure_df, "avg_opinion", "Average Opinion", window_size=3)
    plt.close(fig)

    assert points["date"].tolist() == ["2024-02-29", "2024-03-01", "2024-03-10", "2024-03-20"]
    assert points["weibo-deepseek"].tolist() == pytest.approx([1.0, nan, 3.0, 5.0], nan_ok=True)
    assert points["twitter-deepseek"].tolist() == pytest.approx([nan, 0.0, 2.0, 4.0], nan_ok=True)
    weibo_line, twitter_line = ax.lines[0], ax.lines[1]
    assert list(weibo_line.get_ydata()) == pytest.approx([2.0, 3.0, 4.0])
    assert list(twitter_line.get_ydata()) == pytest.approx([1.0, 2.0, 3.0])


def test_plot_metric_unsmoothed_draws_each_platform_without_gaps():
    nan = float("nan")
    figure_df = pd.DataFrame({
        "date": ["2024-02-29", "2024-03-01", "2024-03-10"],
        "weibo_avg_opinion": [1.0, nan, 3.0],
        "twitter_avg_opinion": [nan, 0.0, 2.0],
    })
    fig, ax = plt.subplots()
    points = plot_metric(ax, figure_df, "avg_opinion", "Average Opinion", use_smoothing=False)
    plt.close(fig)

    assert points["weibo-deepseek"].tolist() == pytest.approx([1.0, nan, 3.0], nan_ok=True)
    assert list(ax.lines[0].get_ydata()) == pytest.approx([1.0, 3.0])
    assert list(ax.lines[1].get_ydata()) == pytest.approx([0.0, 2.0])


def test_plot_uses_month_year_ticks_and_a_frameless_legend():
    figure_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-07-01", "2024-11-01"],
        "weibo_avg_opinion": [0.1, 0.2, 0.3],
        "twitter_avg_opinion": [-0.1, -0.2, -0.3],
    })
    fig, ax = plt.subplots()
    plot_metric(ax, figure_df, "avg_opinion", "label")
    fig.canvas.draw()
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    rotations = {t.get_rotation() for t in ax.get_xticklabels()}
    legend = ax.get_legend()
    plt.close(fig)

    assert tick_labels == ["Mar 2024", "Jul 2024", "Nov 2024"]
    assert rotations == {0.0}
    assert not legend.get_frame_on()
    assert [t.get_text() for t in legend.get_texts()] == ["China (Weibo)", "United States (Twitter)"]


def test_main_names_unsmoothed_figures_raw(tmp_path):
    figure_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-10"],
        **{f"{p}_{m}": [0.1, 0.2] for p in ("weibo", "twitter")
           for m in ("avg_opinion", "weighted_opinion", "user_avg_opinion")},
    })
    figure_data_path = tmp_path / "figure_data.parquet"
    figure_df.to_parquet(figure_data_path, index=False)

    main(figure_data_path=str(figure_data_path), output_dir=str(tmp_path / "figures"), use_smoothing=False)

    assert (tmp_path / "figures" / f"{date.today():%Y-%m-%d}-avg_opinion_comparison_raw.pdf").exists()
