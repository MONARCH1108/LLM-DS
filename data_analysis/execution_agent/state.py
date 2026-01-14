from typing import List, Dict, Any
import pandas as pd


class ExecutionState:
    """
    Holds execution state and history.
    """

    def __init__(self, df: pd.DataFrame, plan: List[str]):
        self.df = df
        self.plan = plan
        self.step_index = 0
        self.attempt = 1
        self.history: List[Dict[str, Any]] = []

    def current_step(self) -> str:
        return self.plan[self.step_index]

    def has_more_steps(self) -> bool:
        return self.step_index < len(self.plan)

    def advance_step(self):
        self.step_index += 1
        self.attempt = 1

    def record(self, record: Dict[str, Any]):
        self.history.append(record)
