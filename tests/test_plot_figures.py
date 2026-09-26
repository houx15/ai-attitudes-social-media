# tests/test_plot_figures.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

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
        pdf_path = output_dir / f"{metric_name}_comparison.pdf"
        csv_path = output_dir / f"{metric_name}_comparison.csv"
        assert pdf_path.exists()
        assert csv_path.exists()
        saved = pd.read_csv(csv_path)
        # exactly two lines: weibo vs twitter, no four-line variant
        assert list(saved.columns) == ["date", "weibo", "twitter"]
