import os
import pandas as pd

"""
from state import ExecutionState
from executor_tool import execute_cleaning_code
from metrics import evaluate_step
from code_writer import generate_code_for_step
"""

from data_cleaning.execution_agent.state import ExecutionState
from data_cleaning.execution_agent.executor_tool import execute_cleaning_code
from data_cleaning.execution_agent.metrics import evaluate_step
from data_cleaning.execution_agent.code_writer import generate_code_for_step


# =====================
# DEBUG CONFIG
# =====================
DEBUG_LLM_CODE = True   
MAX_RETRIES = 5


def load_plan_text(plan_path: str) -> list[str]:
    """
    Splits a plain-text cleaning plan into individual steps.
    Each step starts with 'Step X:'.
    """
    with open(plan_path, "r", encoding="utf-8") as f:
        text = f.read()

    steps = []
    buffer = []

    for line in text.splitlines():
        if line.strip().startswith("Step "):
            if buffer:
                steps.append("\n".join(buffer).strip())
                buffer = []
        buffer.append(line)

    if buffer:
        steps.append("\n".join(buffer).strip())

    return steps


def run_execution_agent(dataset_path: str, plan_path: str) -> pd.DataFrame:
    """
    Main execution loop using a plain-text cleaning plan.
    """

    # ---------- Load dataset ----------
    df = pd.read_csv(dataset_path)

    # ---------- Load plan ----------
    plan_steps = load_plan_text(plan_path)

    # ---------- Initialize state ----------
    state = ExecutionState(df=df, plan=plan_steps)

    print("\n=== EXECUTION AGENT STARTED ===")

    # ---------- Execute plan ----------
    while state.has_more_steps():
        step_text = state.current_step()
        step_number = state.step_index + 1

        print(f"\n▶ Executing Step {step_number}")
        print(step_text)

        success = False
        last_error = None

        while state.attempt <= MAX_RETRIES and not success:
            print(f"  Attempt {state.attempt}")

            before_df = state.df.copy()

            # --- LLM generates code (WITH FEEDBACK) ---
            code = generate_code_for_step(
                step_text=step_text,
                df_sample=state.df,
                feedback=last_error
            )

            # =====================
            # DEBUG: SHOW LLM CODE
            # =====================
            if DEBUG_LLM_CODE:
                print("\n🧠 LLM GENERATED CODE:")
                print("-" * 50)
                print(code)
                print("-" * 50)

            # --- Execute generated code ---
            result = execute_cleaning_code.run({
                "df": state.df,
                "code": code
            })

            # --- Execution error ---
            if result["status"] == "error":
                last_error = result["error"]

                state.record({
                    "step": step_number,
                    "attempt": state.attempt,
                    "error": last_error,
                    "status": "execution_error",
                    "generated_code": code
                })

                state.attempt += 1
                continue

            after_df = result["df"]

            # --- Evaluate impact ---
            metrics = evaluate_step(before_df, after_df)

            # --- Acceptable change ---
            if metrics["row_drop_pct"] <= 10:
                state.df = after_df
                success = True

                state.record({
                    "step": step_number,
                    "attempt": state.attempt,
                    "metrics": metrics,
                    "status": "accepted",
                    "generated_code": code
                })

                print("  ✅ Step accepted")

            # --- Reject due to high data loss ---
            else:
                last_error = (
                    f"Row drop percentage too high: "
                    f"{metrics['row_drop_pct']}% (limit: 10%)"
                )

                state.record({
                    "step": step_number,
                    "attempt": state.attempt,
                    "metrics": metrics,
                    "status": "rejected",
                    "generated_code": code
                })

                state.attempt += 1
                print("  ❌ Too much data loss, retrying")

        if not success:
            raise RuntimeError(f"Step {step_number} failed after {MAX_RETRIES} retries")

        state.advance_step()

    print("\n🎉 CLEANING COMPLETE")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_path = os.path.join(current_dir, "cleaned_dataset.csv")
    state.df.to_csv(cleaned_path, index=False)
    print(f"💾 Cleaned dataset saved to: {cleaned_path}")
    return state.df


# ---------------- TEST ONLY ----------------
if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\data\spotify.csv"
    PLAN_PATH = r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\cleaning_plan.txt"

    final_df = run_execution_agent(DATASET_PATH, PLAN_PATH)
    print("\nFinal shape:", final_df.shape)
