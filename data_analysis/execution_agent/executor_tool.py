import pandas as pd
from langchain_core.tools import tool


SAFE_BUILTINS = {
    "list": list,
    "dict": dict,
    "set": set,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "float": float,
    "int": int,
}


@tool
def execute_analysis_code(code: str, df: pd.DataFrame) -> dict:
    """
    Execute LLM-generated pandas code in a sandbox.
    """

    allowed_globals = {
        "__builtins__": SAFE_BUILTINS,  # ✅ FIX
        "pd": pd,
        "df": df.copy(),
    }

    try:
        exec(code, allowed_globals)

        # -----------------------------
        # df must exist
        # -----------------------------
        if "df" not in allowed_globals:
            return {
                "status": "error",
                "error": "LLM code did not define `df`"
            }

        new_df = allowed_globals["df"]

        # -----------------------------
        # HARD TYPE GUARDRAILS
        # -----------------------------
        if isinstance(new_df, pd.Series):
            return {
                "status": "error",
                "error": (
                    "Execution returned a pandas Series. "
                    "You MUST return a pandas DataFrame. "
                    "Use reset_index() or to_frame()."
                )
            }

        if not isinstance(new_df, pd.DataFrame):
            return {
                "status": "error",
                "error": "`df` is not a pandas DataFrame after execution"
            }

        return {
            "status": "success",
            "df": new_df
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
