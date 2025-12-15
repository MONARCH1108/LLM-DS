import pandas as pd

# =========================================================
# Helper
# =========================================================

def load_df(dataset_path: str) -> pd.DataFrame:
    if not dataset_path:
        raise ValueError("Dataset path is empty")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")
    return df


# =========================================================
# Level-1 Metrics (Basic / Cheap / Always-on)
# =========================================================

def basic_stats(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {"error": "No numeric columns found"}
    return numeric_df.describe().to_dict()


def missingness(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100
    return {
        "missing_count": missing_count.to_dict(),
        "missing_percent": missing_percent.round(2).to_dict()
    }


def dtype_summary(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def unique_counts(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    return df.nunique(dropna=True).to_dict()


def duplicate_rows(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    return {"duplicate_row_count": int(df.duplicated().sum())}


# =========================================================
# Level-1 Aggregator / Checklist
# =========================================================

def run_level_1_checks(dataset_path: str) -> dict:
    miss = missingness(dataset_path)
    cols_with_missing = [c for c, p in miss["missing_percent"].items() if p > 0]

    dup = duplicate_rows(dataset_path)

    uniq = unique_counts(dataset_path)
    high_cardinality_cols = [
        col for col, count in uniq.items()
        if count > 0.5 * max(uniq.values())
    ]

    stats = basic_stats(dataset_path)
    outlier_cols = [
        col for col, s in stats.items()
        if "std" in s and "mean" in s and s["std"] > abs(s["mean"])
    ]

    df = load_df(dataset_path)

    return {
        "missing_values_present": bool(cols_with_missing),
        "columns_with_missing": cols_with_missing,
        "duplicate_rows_present": dup["duplicate_row_count"] > 0,
        "high_cardinality_columns_present": bool(high_cardinality_cols),
        "high_cardinality_columns": high_cardinality_cols,
        "outliers_detected": bool(outlier_cols),
        "outlier_columns": outlier_cols,
        "empty_dataset": df.empty,
        "row_count": len(df),
        "column_count": len(df.columns)
    }


# =========================================================
# Temporary Test Runner
# =========================================================

if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\data\spotify.csv"

    print("\n=== LEVEL 1: DATA QUALITY CHECKLIST ===")
    report = run_level_1_checks(DATASET_PATH)
    for k, v in report.items():
        print(f"{k}: {v}")
