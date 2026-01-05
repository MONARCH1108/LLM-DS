import pandas as pd

# -------- Layer 1: Dataset Metadata --------

def dataset_shape(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    }

def column_names(df: pd.DataFrame) -> list[str]:
    return list(df.columns)

def column_dtypes(df: pd.DataFrame) -> dict:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

# -------- Layer 2: Column Type Grouping --------

def column_type_groups(df: pd.DataFrame) -> dict:
    return {
        "numerical": df.select_dtypes(include="number").columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime": df.select_dtypes(include="datetime").columns.tolist(),
        "boolean": df.select_dtypes(include="bool").columns.tolist(),
    }


# -------- Layer 3: Sample Rows --------

def sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    return df.head(n).to_dict(orient="records")

# -------- Layer 4: Basic Numeric Stats --------

def basic_numeric_stats(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}

    desc = numeric_df.describe().T
    return desc[["count", "mean", "std", "min", "max"]].to_dict()


# -------- Layer 5: Missing Value Summary --------

def missing_value_summary(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    total = len(df)

    return {
        col: {
            "missing_count": int(missing[col]),
            "missing_pct": float(missing[col] / total),
        }
        for col in df.columns
        if missing[col] > 0
    }

# -------- Layer 6: Cardinality --------

def column_cardinality(df: pd.DataFrame) -> dict:
    return {
        col: int(df[col].nunique(dropna=True))
        for col in df.columns
    }


# -------- Layer 7: Possible Identifiers --------

def possible_identifier_columns(df: pd.DataFrame) -> list[str]:
    n_rows = len(df)
    return [
        col for col in df.columns
        if df[col].nunique(dropna=True) == n_rows
    ]


# -------- Layer 8: Low-Information Columns --------

def low_information_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if df[col].nunique(dropna=True) <= 1
    ]

# -------- Layer 9: Duplicate Row Summary --------

def duplicate_row_summary(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())

    return {
        "duplicate_row_count": duplicate_rows,
        "duplicate_row_pct": float(duplicate_rows / total_rows) if total_rows > 0 else 0.0,
    }

# -------- Layer 10: Zero-Value Summary (Numeric Only) --------

def zero_value_summary(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}

    return {
        col: int((numeric_df[col] == 0).sum())
        for col in numeric_df.columns
    }

# -------- Layer 11: Text Length Summary (pure structural) --------

def text_length_summary(df: pd.DataFrame) -> dict:
    text_cols = df.select_dtypes(include=["object", "category"])
    summary = {}

    for col in text_cols.columns:
        lengths = text_cols[col].dropna().astype(str).str.len()
        if not lengths.empty:
            summary[col] = {
                "min_length": int(lengths.min()),
                "max_length": int(lengths.max()),
                "avg_length": float(lengths.mean()),
            }

    return summary

# -------- Layer 12: Numeric Skew Signal --------

def numeric_skew_summary(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return {}

    return {
        col: float(numeric_df[col].skew())
        for col in numeric_df.columns
        if numeric_df[col].nunique(dropna=True) > 2
    }

# -------- Final Profile --------

def dataset_profile(df: pd.DataFrame) -> dict:
    return {
        "shape": dataset_shape(df),
        "columns": column_names(df),
        "dtypes": column_dtypes(df),
        "column_groups": column_type_groups(df),
        "sample_rows": sample_rows(df),
        "numeric_stats": basic_numeric_stats(df),
        "missing_summary": missing_value_summary(df),
        "cardinality": column_cardinality(df),
        "possible_identifiers": possible_identifier_columns(df),
        "low_information_columns": low_information_columns(df),

        "duplicate_rows": duplicate_row_summary(df),
        "zero_value_summary": zero_value_summary(df),
        "text_length_summary": text_length_summary(df),
        "numeric_skew": numeric_skew_summary(df),
    }

# -------------------------------------------------------------------
# Simple local test (NO CLI, NO orchestration)
# -------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from pprint import pprint

    # 🔹 CHANGE THIS PATH WHEN TESTING
    TEST_DATASET_PATH = r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\execution_agent\cleaned_dataset.csv"

    print("\n🔍 Running dataset_profile local test...\n")

    try:
        df = pd.read_csv(TEST_DATASET_PATH)
    except Exception as e:
        print("❌ Failed to load dataset")
        print(e)
        raise SystemExit(1)

    print("✅ Dataset loaded")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    profile = dataset_profile(df)

    print("\n📊 Dataset Profile Output\n" + "-" * 50)
    pprint(profile)

    print("\n✅ Local test completed successfully\n")
