from typing import List, Dict, Any
import pandas as pd


class ExecutionState:
    """
    Holds the full execution state of the cleaning agent.
    """
    def __init__(self, df: pd.DataFrame, plan: List[Dict[str, Any]]):
        self.df = df                     
        self.plan = plan                
        self.step_index = 0             
        self.attempt = 1              
        self.history = []                

    def current_step(self) -> Dict[str, Any]:
        return self.plan[self.step_index]

    def advance_step(self):
        self.step_index += 1
        self.attempt = 1

    def record(self, record: Dict[str, Any]):
        self.history.append(record)

    def has_more_steps(self) -> bool:
        return self.step_index < len(self.plan)
