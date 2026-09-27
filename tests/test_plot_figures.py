# tests/test_plot_figures.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from datetime import date

from plot_figures import apply_sliding_window, main, plot_metric


def test_apply_sliding_window_centered_mean():
    df = pd.DataFrame({"date": pd.date_range("2024-03-01", periods=5), "metric": [1, 2, 3, 4, 5]})
    smoothed = apply_sliding_window(df, "metric", window_size=3)
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

    assert list(points.columns) == ["date", "weibo", "twitter"]
    assert points["weibo"].tolist() == pytest.approx([0.5, 1.0])
    assert points["twitter"].tolist() == pytest.approx([-0.5, -1.0])


def test_main_writes_one_pdf_and_csv_per_metric(tmp_path):
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

    main(figure_data_path=str(figure_data_path), output_dir=str(output_dir))

    for metric_name in ["avg_opinion", "weighted_opinion", "user_avg_opinion"]:
        # Legacy plot.py naming, minus the retired correction tag.
        stem = f"{metric_name}_comparison_smoothed3d_{date.today():%Y-%m-%d}"
        pdf_path = output_dir / f"{stem}.pdf"
        csv_path = output_dir / f"{stem}.csv"
        assert pdf_path.exists()
        assert csv_path.exists()
        saved = pd.read_csv(csv_path)
        # exactly two lines: weibo vs twitter, no four-line variant
        assert list(saved.columns) == ["date", "weibo", "twitter"]


@pytest.mark.parametrize("metric_name", ["avg_opinion", "weighted_opinion", "user_avg_opinion"])
def test_plot_metric_draws_benefits_and_concerns_annotations(metric_name):
    # (G) spec: same visual style as legacy plot.py, incl. the "AI benefits" /
    # "AI concerns" annotations near the top/bottom of the y-range, all metrics.
    figure_df = pd.DataFrame({
        "date": pd.date_range("2024-03-01", periods=5).strftime("%Y-%m-%d"),
        f"weibo_{metric_name}": [0.1, 0.2, 0.3, 0.4, 0.5],
        f"twitter_{metric_name}": [-0.1, -0.2, -0.3, -0.4, -0.5],
    })
    fig, ax = plt.subplots()
    plot_metric(ax, figure_df, metric_name, "label", use_smoothing=False)
    texts = {t.get_text(): t for t in ax.texts}
    y_low, y_high = ax.get_ylim()
    plt.close(fig)

    expected = {"AI benefits", "AI concerns"}
    if metric_name == "weighted_opinion":
        # legacy plot.py also labels the y=0 dashed line on this figure only
        expected.add("neutral")
        assert texts["neutral"].get_position()[1] == pytest.approx(0.05)
    assert len(ax.texts) == len(expected)
    assert set(texts) == expected
    benefits_y = texts["AI benefits"].get_position()[1]
    concerns_y = texts["AI concerns"].get_position()[1]
    # benefits near the top, concerns near the bottom, both inside the axes
    assert concerns_y < 0 < benefits_y
    assert y_low < concerns_y and benefits_y < y_high
    assert benefits_y > 0.5 and concerns_y < -0.5


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
    assert points["weibo"].tolist() == pytest.approx([2.0, nan, 3.0, 4.0], nan_ok=True)
    assert points["twitter"].tolist() == pytest.approx([nan, 1.0, 2.0, 3.0], nan_ok=True)
    weibo_line, twitter_line = ax.lines[0], ax.lines[1]
    assert len(weibo_line.get_xdata()) == 3
    assert len(twitter_line.get_xdata()) == 3


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

    assert points["weibo"].tolist() == pytest.approx([1.0, nan, 3.0], nan_ok=True)
    assert list(ax.lines[0].get_ydata()) == pytest.approx([1.0, 3.0])
    assert list(ax.lines[1].get_ydata()) == pytest.approx([0.0, 2.0])


def test_plot_matches_legacy_font_size_and_tick_rotation():
    figure_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-04-01", "2024-05-01"],
        "weibo_avg_opinion": [0.1, 0.2, 0.3],
        "twitter_avg_opinion": [-0.1, -0.2, -0.3],
    })
    fig, ax = plt.subplots()
    plot_metric(ax, figure_df, "avg_opinion", "label")
    rotations = {t.get_rotation() for t in ax.get_xticklabels()}
    plt.close(fig)

    assert plt.rcParams["font.size"] == 12
    assert rotations == {45.0}


def test_main_names_unsmoothed_figures_raw(tmp_path):
    figure_df = pd.DataFrame({
        "date": ["2024-03-01", "2024-03-10"],
        **{f"{p}_{m}": [0.1, 0.2] for p in ("weibo", "twitter")
           for m in ("avg_opinion", "weighted_opinion", "user_avg_opinion")},
    })
    figure_data_path = tmp_path / "figure_data.parquet"
    figure_df.to_parquet(figure_data_path, index=False)

    main(figure_data_path=str(figure_data_path), output_dir=str(tmp_path / "figures"), use_smoothing=False)

    assert (tmp_path / "figures" / f"avg_opinion_comparison_raw_{date.today():%Y-%m-%d}.pdf").exists()
