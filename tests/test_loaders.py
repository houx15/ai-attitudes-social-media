from loaders import iter_target_dates


def test_iter_target_dates_single_month():
    dates = iter_target_dates("2024-03-01", "2024-03-31", [1, 10, 20])
    assert dates == ["2024-03-01", "2024-03-10", "2024-03-20"]


def test_iter_target_dates_spans_months():
    dates = iter_target_dates("2024-03-15", "2024-04-15", [1, 10, 20])
    assert dates == ["2024-03-20", "2024-04-01", "2024-04-10"]


def test_iter_target_dates_no_matches():
    dates = iter_target_dates("2024-03-02", "2024-03-09", [1, 10, 20])
    assert dates == []


def test_iter_target_dates_single_day_range():
    dates = iter_target_dates("2024-03-10", "2024-03-10", [1, 10, 20])
    assert dates == ["2024-03-10"]
