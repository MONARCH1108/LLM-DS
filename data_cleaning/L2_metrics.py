import pandas as pd
from data_cleaning.L1_metrics import load_df


# =========================================================
# Level-2 Metrics (Advanced / Diagnostic)
# =========================================================

# A. Constant & Near-Constant Columns

def constant_columns(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    return {
        col: int(df[col].nunique(dropna=False))
        for col in df.columns
        if df[col].nunique(dropna=False) <= 1
    }


def near_constant_columns(dataset_path: str, threshold: float = 0.95) -> dict:
    df = load_df(dataset_path)
    result = {}
    for col in df.columns:
        top_freq = df[col].value_counts(dropna=False, normalize=True).iloc[0]
        if top_freq >= threshold:
            result[col] = round(float(top_freq), 3)
    return result


# B. Outliers (IQR)

def outlier_iqr(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    num_df = df.select_dtypes(include="number")
    outliers = {}

    for col in num_df.columns:
        q1 = num_df[col].quantile(0.25)
        q3 = num_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((num_df[col] < lower) | (num_df[col] > upper)).sum())

        outliers[col] = {
            "outlier_count": count,
            "lower_bound": float(lower),
            "upper_bound": float(upper)
        }

    return outliers


# C. Skewness

def skewness(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    return df.select_dtypes(include="number").skew().round(3).to_dict()


# D. Row-Level Missingness

def row_missingness(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    row_missing_pct = (df.isna().sum(axis=1) / len(df.columns)) * 100
    return {
        "rows_above_50pct_missing": int((row_missing_pct > 50).sum()),
        "rows_above_80pct_missing": int((row_missing_pct > 80).sum())
    }


# E. Numeric-Looking Strings

def numeric_string_ratio(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    result = {}

    for col in df.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        ratio = coerced.notna().mean()
        if ratio > 0:
            result[col] = round(float(ratio), 3)

    return result


# F. Text Quality

def text_quality(dataset_path: str) -> dict:
    df = load_df(dataset_path)
    result = {}

    for col in df.select_dtypes(include="object").columns:
        s = df[col].astype(str)
        result[col] = {
            "avg_length": round(float(s.str.len().mean()), 2),
            "empty_or_space_only": int((s.str.strip() == "").sum())
        }

    return result


# G. High Cardinality Categories

def high_cardinality(dataset_path: str, threshold: int = 50) -> dict:
    df = load_df(dataset_path)
    return {
        col: int(df[col].nunique())
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() >= threshold
    }


# =========================================================
# Level-2 → LLM Signal Compressor
# =========================================================

def summarize_level_2_for_llm(level_2_metrics: dict) -> dict:
    """
    Compresses detailed Level-2 metrics into LLM-friendly signals.
    NO raw stats, NO bounds, NO large dicts.
    """

    summary = {}

    # ---------- Outliers ----------
    outlier_info = level_2_metrics.get("outlier_iqr", {})
    outlier_columns = []
    severe_outliers = []

    total_outliers = sum(v["outlier_count"] for v in outlier_info.values()) or 1

    for col, info in outlier_info.items():
        count = info.get("outlier_count", 0)
        if count > 0:
            outlier_columns.append(col)
            if count / total_outliers > 0.3:
                severe_outliers.append(col)

    summary["outliers"] = {
        "present": bool(outlier_columns),
        "columns": outlier_columns,
        "severe_columns": severe_outliers
    }

    # ---------- Skewness ----------
    skew = level_2_metrics.get("skewness", {})
    highly_skewed = [col for col, val in skew.items() if abs(val) > 2]

    summary["skewness"] = {
        "highly_skewed_columns": highly_skewed,
        "transform_recommended": bool(highly_skewed)
    }

    # ---------- High Cardinality ----------
    high_card = level_2_metrics.get("high_cardinality", {})

    summary["categorical_cardinality"] = {
        "high_cardinality_columns": list(high_card.keys()),
        "encoding_required": bool(high_card)
    }

    # ---------- Numeric-looking Strings ----------
    num_str = level_2_metrics.get("numeric_string_ratio", {})
    convertible_cols = [
        col for col, ratio in num_str.items() if ratio >= 0.8
    ]

    summary["numeric_string_columns"] = {
        "columns": convertible_cols,
        "convert_to_numeric": bool(convertible_cols)
    }

    # ---------- Row-level Missingness ----------
    row_miss = level_2_metrics.get("row_missingness", {})

    summary["row_quality"] = {
        "drop_rows_recommended": row_miss.get("rows_above_80pct_missing", 0) > 0
    }

    # ---------- Constant / Near-Constant ----------
    summary["low_variance_features"] = {
        "constant_columns": list(level_2_metrics.get("constant_columns", {}).keys()),
        "near_constant_columns": list(level_2_metrics.get("near_constant_columns", {}).keys()),
        "drop_recommended": bool(
            level_2_metrics.get("constant_columns") or
            level_2_metrics.get("near_constant_columns")
        )
    }

    # ---------- Text Quality ----------
    text_q = level_2_metrics.get("text_quality", {})
    bad_text_cols = [
        col for col, v in text_q.items()
        if v.get("empty_or_space_only", 0) > 0
    ]

    summary["text_quality"] = {
        "issues_present": bool(bad_text_cols),
        "affected_columns": bad_text_cols
    }

    # ---------- Final Hint ----------
    summary["cleaning_complexity"] = (
        "high"
        if (
            summary["outliers"]["present"]
            or summary["skewness"]["transform_recommended"]
            or summary["categorical_cardinality"]["encoding_required"]
        )
        else "low"
    )

    return summary


# =========================================================
# Level-2 Aggregator (RAW diagnostics)
# =========================================================

def run_level_2_metrics(dataset_path: str) -> dict:
    return {
        "constant_columns": constant_columns(dataset_path),
        "near_constant_columns": near_constant_columns(dataset_path),
        "outlier_iqr": outlier_iqr(dataset_path),
        "skewness": skewness(dataset_path),
        "row_missingness": row_missingness(dataset_path),
        "numeric_string_ratio": numeric_string_ratio(dataset_path),
        "text_quality": text_quality(dataset_path),
        "high_cardinality": high_cardinality(dataset_path),
    }

# =========================================================
# Temporary Test Runner
# =========================================================

if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\data\spotify.csv"
    level_2_raw = run_level_2_metrics(DATASET_PATH)
    print("\n=== LEVEL 2 → LLM SUMMARY ===")
    summary = summarize_level_2_for_llm(level_2_raw)
    for k, v in summary.items():
        print(f"{k}: {v}")
