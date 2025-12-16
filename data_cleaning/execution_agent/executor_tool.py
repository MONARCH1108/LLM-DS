import pandas as pd
from langchain_core.tools import tool


@tool
def execute_cleaning_code(code: str, df: pd.DataFrame) -> dict:
    """
    Executes LLM-generated pandas cleaning code safely.

    Rules:
    - Code MUST define or reassign `df`
    - `df` MUST remain a pandas DataFrame
    - Valid no-op (skip) steps are allowed
    """

    df_before = df.copy()

    allowed_globals = {
        "__builtins__": {},
        "pd": pd,
        "df": df.copy()
    }

    try:
        exec(code, allowed_globals)

        # ---- df must exist ----
        if "df" not in allowed_globals:
            return {
                "status": "error",
                "error": "LLM code did not define `df`"
            }

        new_df = allowed_globals["df"]

        # ---- df must be a DataFrame ----
        if not isinstance(new_df, pd.DataFrame):
            return {
                "status": "error",
                "error": "`df` is not a pandas DataFrame after execution"
            }

        # ---- VALID NO-OP (SKIP) ----
        if new_df.equals(df_before):
            return {
                "status": "success",
                "df": new_df,
                "noop": True
            }

        # ---- Normal successful execution ----
        return {
            "status": "success",
            "df": new_df
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
