"""Write Stage 1 results (parquet parts) for tests, from CSV-like text or a DataFrame."""

import io

import pandas as pd

from openrouter_client import RESULTS_COLUMNS, write_results_part


def seed_results(results_dir, rows, prefix="task1of1"):
    df = pd.read_csv(io.StringIO(rows), dtype=str, keep_default_na=False) if isinstance(rows, str) else rows.copy()
    for column in RESULTS_COLUMNS:
        if column not in df.columns:
            df[column] = 0 if column.endswith("tokens") else ""
    records = []
    for row in df[RESULTS_COLUMNS].to_dict("records"):
        opinion = row["opinion"]
        empty = opinion is None or (isinstance(opinion, float) and opinion != opinion) or str(opinion) == ""
        records.append({**row, "id": str(row["id"]), "opinion": None if empty else str(opinion)})
    return write_results_part(str(results_dir), prefix, records)
