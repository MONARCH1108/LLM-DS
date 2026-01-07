from typing import Dict, Any, Tuple, Optional
import pandas as pd


class AnalysisStepExecutionError(Exception):
    """Hard failure during step execution."""
    pass


# ----------------------------------------------------------------------
# STEP EXECUTOR
# ----------------------------------------------------------------------

class AnalysisStepExecutor:
    """
    Executes ONE analysis step on a pandas DataFrame.

    This class:
    - Executes exactly one step
    - Does NOT interpret intent
    - Does NOT call LLM
    - Does NOT validate schema beyond existence
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise AnalysisStepExecutionError("Input df must be a pandas DataFrame")
        self.df = df

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def execute_step(self, step_instruction: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute a single step.

        step_instruction MUST contain:
        - step_number
        - operation
        - input_columns
        - optional parameters (operation-specific)

        Returns:
        - updated DataFrame
        - execution metadata
        """

        operation = step_instruction.get("OPERATION")

        if not operation:
            raise AnalysisStepExecutionError("Missing OPERATION in step instruction")

        operation = operation.lower()

        if operation == "group":
            return self._execute_group(step_instruction)

        if operation == "rank_and_filter":
            return self._execute_rank_and_filter(step_instruction)

        if operation == "select":
            return self._execute_select(step_instruction)

        raise AnalysisStepExecutionError(f"Unsupported operation: {operation}")

    # ------------------------------------------------------------------
    # OPERATION IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def _execute_group(self, step: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        GROUP operation:
        Produces a grouped object but does NOT aggregate.
        """

        group_by = step.get("GROUP_BY_COLUMNS")

        if not group_by:
            raise AnalysisStepExecutionError("GROUP_BY_COLUMNS required for group operation")

        for col in group_by:
            if col not in self.df.columns:
                raise AnalysisStepExecutionError(f"Column not found: {col}")

        grouped = self.df.groupby(group_by, dropna=False)

        metadata = {
            "operation": "group",
            "group_by_columns": group_by,
            "group_count": len(grouped),
        }

        return grouped, metadata

    def _execute_rank_and_filter(self, step: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        RANK_AND_FILTER operation:
        Selects top-N rows per partition based on ordering.
        """

        partition_cols = step.get("PARTITION_BY_COLUMNS")
        order_col = step.get("ORDER_BY_COLUMN")
        order_dir = step.get("ORDER_DIRECTION", "DESCENDING")
        rank_limit = step.get("RANK_LIMIT", 1)

        if not partition_cols or not order_col:
            raise AnalysisStepExecutionError("PARTITION_BY_COLUMNS and ORDER_BY_COLUMN required")

        ascending = order_dir.upper() != "DESCENDING"

        for col in partition_cols + [order_col]:
            if col not in self.df.columns:
                raise AnalysisStepExecutionError(f"Column not found: {col}")

        ranked = (
            self.df
            .sort_values(order_col, ascending=ascending)
            .groupby(partition_cols, dropna=False)
            .head(rank_limit)
            .reset_index(drop=True)
        )

        metadata = {
            "operation": "rank_and_filter",
            "partition_by": partition_cols,
            "order_by": order_col,
            "rank_limit": rank_limit,
            "result_rows": len(ranked),
        }

        return ranked, metadata

    def _execute_select(self, step: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        SELECT operation:
        Column projection only.
        """

        columns = step.get("OUTPUT_COLUMNS") or step.get("INPUT_COLUMNS")

        if not columns:
            raise AnalysisStepExecutionError("No columns specified for select operation")

        for col in columns:
            if col not in self.df.columns:
                raise AnalysisStepExecutionError(f"Column not found: {col}")

        selected = self.df[columns].copy()

        metadata = {
            "operation": "select",
            "selected_columns": columns,
            "result_rows": len(selected),
        }

        return selected, metadata
