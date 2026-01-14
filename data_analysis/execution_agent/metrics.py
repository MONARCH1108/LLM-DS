import pandas as pd


def evaluate_step(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "row_drop_pct": round(
            (len(before) - len(after)) / max(len(before), 1) * 100, 2
        ),
        "columns_before": len(before.columns),
        "columns_after": len(after.columns),
        "new_columns": sorted(set(after.columns) - set(before.columns)),
        "removed_columns": sorted(set(before.columns) - set(after.columns)),
    }
