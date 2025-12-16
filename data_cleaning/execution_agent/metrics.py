import pandas as pd


def evaluate_step(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """
    Compare dataframe before and after a cleaning step.
    """

    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "rows_dropped": len(before) - len(after),
        "row_drop_pct": round(
            (len(before) - len(after)) / max(len(before), 1) * 100, 2
        ),
        "nulls_before": int(before.isna().sum().sum()),
        "nulls_after": int(after.isna().sum().sum()),
        "nulls_delta": int(after.isna().sum().sum() - before.isna().sum().sum()),
        "columns_before": len(before.columns),
        "columns_after": len(after.columns),
    }
