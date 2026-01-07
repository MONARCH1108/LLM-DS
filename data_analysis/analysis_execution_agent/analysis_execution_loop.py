from pathlib import Path
from typing import Dict, Any, List
import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from analysis_step_executor import AnalysisStepExecutor, AnalysisStepExecutionError

load_dotenv()


class AnalysisExecutionLoopError(Exception):
    """Fatal error in analysis execution loop."""
    pass


# ----------------------------------------------------------------------
# EXECUTION TRACE PRINTER (ADDED)
# ----------------------------------------------------------------------

def pretty_print_instruction(step_number: int, instruction: Dict[str, Any]) -> None:
    """
    Human-readable execution trace.
    NO execution. NO logic.
    """
    print("\n" + "=" * 70)
    print(f"🧠 EXECUTION INSTRUCTION — STEP {step_number}")
    print("=" * 70)

    for key, value in instruction.items():
        print(f"{key}: {value}")

    print("=" * 70)


# ----------------------------------------------------------------------
# GROQ PROMPT (LOCKED)
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a Step-Level Analysis Runtime Interpreter.

Your task:
- Interpret EXACTLY ONE analysis step
- Convert it into an EXECUTION INSTRUCTION

STRICT RULES:
- Do NOT plan ahead
- Do NOT modify intent
- Do NOT invent columns
- Do NOT choose statistics
- Do NOT execute code
- Do NOT return explanations

You MUST output a Python-dict-like structure
using ONLY the keys required by the executor.

Allowed operations:
- group
- rank_and_filter
- select

Output format (STRICT, NO MARKDOWN):

{
  "OPERATION": "...",
  "...": "..."
}
"""

USER_PROMPT_TEMPLATE = """
DATASET COLUMNS:
{dataset_columns}

CURRENT DATAFRAME SHAPE:
{df_shape}

EXECUTION CONTEXT:
{execution_context}

NEXT ANALYSIS STEP (RAW TEXT):
{step_text}

Produce the execution instruction ONLY.
"""


# ----------------------------------------------------------------------
# EXECUTION LOOP
# ----------------------------------------------------------------------

class AnalysisExecutionLoop:
    """
    End-to-end orchestration of analysis execution.
    """

    def __init__(
        self,
        cleaned_dataset_path: str,
        normalized_plan_path: str,
        output_dir: str | None = None,
    ):
        self.cleaned_dataset_path = Path(cleaned_dataset_path)
        self.normalized_plan_path = Path(normalized_plan_path)

        self.output_dir = (
            Path(output_dir)
            if output_dir
            else Path(__file__).parent / "outputs"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df: pd.DataFrame | None = None
        self.execution_context: Dict[str, Any] = {}

        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ------------------------------------------------------------------
    # LOADERS
    # ------------------------------------------------------------------

    def load_dataset(self) -> None:
        if not self.cleaned_dataset_path.exists():
            raise AnalysisExecutionLoopError("Cleaned dataset not found")

        self.df = pd.read_csv(self.cleaned_dataset_path)

    def load_plan_steps(self) -> List[str]:
        if not self.normalized_plan_path.exists():
            raise AnalysisExecutionLoopError("Normalized plan not found")

        text = self.normalized_plan_path.read_text(encoding="utf-8")

        steps = [
            block.strip()
            for block in text.split("STEP_")
            if block.strip().startswith(("1", "2", "3", "4", "5"))
        ]

        return steps

    # ------------------------------------------------------------------
    # STEP INTERPRETATION (LLM)
    # ------------------------------------------------------------------

    def interpret_step(self, step_text: str) -> Dict[str, Any]:
        if self.df is None:
            raise AnalysisExecutionLoopError("Dataset not loaded")

        prompt = USER_PROMPT_TEMPLATE.format(
            dataset_columns=list(self.df.columns),
            df_shape=self.df.shape,
            execution_context=self.execution_context,
            step_text=step_text,
        )

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()

        try:
            instruction = eval(raw, {}, {})
        except Exception:
            raise AnalysisExecutionLoopError(
                f"Failed to interpret step into executable instruction:\n{raw}"
            )

        return instruction

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.load_dataset()
        steps = self.load_plan_steps()

        executor = AnalysisStepExecutor(self.df)

        for idx, step_text in enumerate(steps, start=1):
            print(f"\n▶ Executing Step {idx}")

            instruction = self.interpret_step(step_text)

            # 🔍 PRINT EXECUTION TRACE (ADDED)
            pretty_print_instruction(idx, instruction)

            try:
                result, metadata = executor.execute_step(instruction)
            except AnalysisStepExecutionError as e:
                raise AnalysisExecutionLoopError(
                    f"Execution failed at step {idx}: {e}"
                )

            if isinstance(result, pd.DataFrame):
                self.df = result
                executor.df = self.df

            self.execution_context[f"step_{idx}"] = {
                "instruction": instruction,
                "metadata": metadata,
            }

        final_path = self.output_dir / "final_result.csv"
        self.df.to_csv(final_path, index=False)

        print("\n✅ Analysis execution completed")
        print(f"📁 Final result saved to: {final_path}")


# ----------------------------------------------------------------------
# LOCAL TEST
# ----------------------------------------------------------------------

if __name__ == "__main__":
    loop = AnalysisExecutionLoop(
        cleaned_dataset_path=(
            r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\execution_agent\cleaned_dataset.csv"
        ),
        normalized_plan_path=(
            r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_analysis\analysis_execution_agent\analysis_plan_normalized_raw.txt"
        ),
    )

    loop.run()
