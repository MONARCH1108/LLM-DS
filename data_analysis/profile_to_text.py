# data_analysis/profile_to_text.py

from typing import Dict, Any
from pathlib import Path


DIVIDER = "-" * 60


def fmt(v):
    """
    Stable formatting for display only.
    Never used for logic.
    """
    return round(v, 4) if isinstance(v, float) else v


def profile_to_text(
    profile: Dict[str, Any],
    output_path: str | None = None,
) -> str:
    """
    Convert dataset_profile dictionary into STRICT, deterministic text
    for LLM consumption.

    CONTRACT:
    - FACTUAL DESCRIPTION ONLY
    - NO INTERPRETATION
    - NO CONCLUSIONS
    - NO SUGGESTIONS

    If output_path is NOT provided, the output is saved as:
        dataset_profile.txt
    in the SAME directory as this file.
    """

    lines: list[str] = []

    # ------------------------------------------------------------------
    # FACT-ONLY CONTRACT HEADER
    # ------------------------------------------------------------------
    lines.append("DATASET PROFILE — FACTUAL DESCRIPTION ONLY")
    lines.append("NO INTERPRETATION. NO CONCLUSIONS. NO SUGGESTIONS.")
    lines.append(DIVIDER)
    lines.append("")

    # ------------------------------------------------------------------
    # DATASET OVERVIEW
    # ------------------------------------------------------------------
    shape = profile.get("shape", {})
    lines.append("DATASET OVERVIEW")
    lines.append(DIVIDER)
    lines.append(f"- Rows: {shape.get('rows')}")
    lines.append(f"- Columns: {shape.get('columns')}")
    lines.append("")

    # ------------------------------------------------------------------
    # COLUMN SCHEMA
    # ------------------------------------------------------------------
    lines.append("COLUMN SCHEMA (NAME : DTYPE)")
    lines.append(DIVIDER)
    for col in sorted(profile.get("dtypes", {})):
        lines.append(f"- {col}: {profile['dtypes'][col]}")
    lines.append("")

    # ------------------------------------------------------------------
    # COLUMN GROUPS
    # ------------------------------------------------------------------
    lines.append("COLUMN GROUPS")
    lines.append(DIVIDER)
    for group in sorted(profile.get("column_groups", {})):
        cols = profile["column_groups"][group]
        if cols:
            lines.append(f"- {group}: {', '.join(sorted(cols))}")
        else:
            lines.append(f"- {group}: none")
    lines.append("")

    # ------------------------------------------------------------------
    # SAMPLE ROWS
    # ------------------------------------------------------------------
    lines.append("SAMPLE ROWS (FIRST N)")
    lines.append(DIVIDER)
    for i, row in enumerate(profile.get("sample_rows", []), start=1):
        lines.append(f"- Row {i}: {row}")
    lines.append("")

    # ------------------------------------------------------------------
    # NUMERIC STATISTICS
    # ------------------------------------------------------------------
    numeric_stats = profile.get("numeric_stats", {})
    if numeric_stats:
        lines.append("NUMERIC STATISTICS (SUMMARY)")
        lines.append(DIVIDER)
        for stat_name in sorted(numeric_stats):
            lines.append(f"- {stat_name}:")
            for col in sorted(numeric_stats[stat_name]):
                lines.append(
                    f"  - {col}: {fmt(numeric_stats[stat_name][col])}"
                )
        lines.append("")

    # ------------------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------------------
    lines.append("MISSING VALUE SUMMARY")
    lines.append(DIVIDER)
    missing = profile.get("missing_summary", {})
    if missing:
        for col in sorted(missing):
            info = missing[col]
            lines.append(
                f"- {col}: missing_count={info['missing_count']}, "
                f"missing_pct={fmt(info['missing_pct'])}"
            )
    else:
        lines.append("- No missing values detected")
    lines.append("")

    # ------------------------------------------------------------------
    # CARDINALITY
    # ------------------------------------------------------------------
    lines.append("COLUMN CARDINALITY")
    lines.append(DIVIDER)
    for col in sorted(profile.get("cardinality", {})):
        lines.append(f"- {col}: {profile['cardinality'][col]}")
    lines.append("")

    # ------------------------------------------------------------------
    # POSSIBLE IDENTIFIERS
    # ------------------------------------------------------------------
    lines.append("POSSIBLE IDENTIFIER COLUMNS")
    lines.append(DIVIDER)
    identifiers = profile.get("possible_identifiers", [])
    if identifiers:
        for col in sorted(identifiers):
            lines.append(f"- {col}")
    else:
        lines.append("- None detected")
    lines.append("")

    # ------------------------------------------------------------------
    # LOW INFORMATION COLUMNS
    # ------------------------------------------------------------------
    lines.append("LOW INFORMATION COLUMNS")
    lines.append(DIVIDER)
    low_info = profile.get("low_information_columns", [])
    if low_info:
        for col in sorted(low_info):
            lines.append(f"- {col}")
    else:
        lines.append("- None detected")
    lines.append("")

    # ------------------------------------------------------------------
    # DUPLICATE ROWS
    # ------------------------------------------------------------------
    dup = profile.get("duplicate_rows", {})
    lines.append("DUPLICATE ROW SUMMARY (STRUCTURAL SIGNAL)")
    lines.append(DIVIDER)
    lines.append(f"- Duplicate row count: {dup.get('duplicate_row_count')}")
    lines.append(f"- Duplicate row percentage: {fmt(dup.get('duplicate_row_pct'))}")
    lines.append("")

    # ------------------------------------------------------------------
    # ZERO VALUE SUMMARY
    # ------------------------------------------------------------------
    lines.append("ZERO VALUE SUMMARY (STRUCTURAL SIGNAL)")
    lines.append(DIVIDER)
    zero_vals = profile.get("zero_value_summary", {})
    if zero_vals:
        for col in sorted(zero_vals):
            lines.append(f"- {col}: {zero_vals[col]}")
    else:
        lines.append("- No numeric columns with zero values")
    lines.append("")

    # ------------------------------------------------------------------
    # TEXT LENGTH SUMMARY
    # ------------------------------------------------------------------
    lines.append("TEXT LENGTH SUMMARY (STRUCTURAL SIGNAL)")
    lines.append(DIVIDER)
    text_len = profile.get("text_length_summary", {})
    if text_len:
        for col in sorted(text_len):
            stats = text_len[col]
            lines.append(
                f"- {col}: min_length={stats['min_length']}, "
                f"max_length={stats['max_length']}, "
                f"avg_length={fmt(stats['avg_length'])}"
            )
    else:
        lines.append("- No text columns detected")
    lines.append("")

    # ------------------------------------------------------------------
    # NUMERIC SKEW
    # ------------------------------------------------------------------
    skew = profile.get("numeric_skew", {})
    if skew:
        lines.append("NUMERIC SKEW (STATISTICAL SIGNAL)")
        lines.append(DIVIDER)
        for col in sorted(skew):
            lines.append(f"- {col}: {fmt(skew[col])}")
        lines.append("")

    # ------------------------------------------------------------------
    # FOOTER
    # ------------------------------------------------------------------
    lines.append(DIVIDER)
    lines.append("END OF DATASET PROFILE")

    text_output = "\n".join(lines)

    # ------------------------------------------------------------------
    # WRITE TO FILE
    # ------------------------------------------------------------------
    if output_path is None:
        # Save in SAME directory as this file
        output_path = Path(__file__).parent / "dataset_profile.txt"
    else:
        output_path = Path(output_path)

    output_path.write_text(text_output, encoding="utf-8")

    return text_output


# ----------------------------------------------------------------------
# Local test (developer-only)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from dataset_profile import dataset_profile

    TEST_DATASET_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\execution_agent\cleaned_dataset.csv"
    )

    df = pd.read_csv(TEST_DATASET_PATH)
    profile = dataset_profile(df)

    text = profile_to_text(profile)

    print("\n📄 PROFILE TO TEXT OUTPUT\n" + "-" * 60)
    print(text)
    print("\n✅ Saved to same directory as profile_to_text.py")
