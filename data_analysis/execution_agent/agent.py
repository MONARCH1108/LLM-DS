from pathlib import Path
import pandas as pd

from code_writer import generate_code_for_step
from executor_tool import execute_analysis_code
from metrics import evaluate_step
from state import ExecutionState


MAX_RETRIES = 5
DEBUG_LLM_CODE = True


class AnalysisExecutionAgentError(Exception):
    """Fatal error during analysis execution."""


def load_analysis_plan(plan_path: str) -> list[str]:
    """
    Extract executable Step blocks from analysis_plan.txt.
    """
    text = Path(plan_path).read_text(encoding="utf-8")

    if "PLAN STEPS" not in text:
        raise AnalysisExecutionAgentError("PLAN STEPS section not found")

    plan_steps_text = text.split("PLAN STEPS", 1)[1]

    steps = []
    buffer = []
    in_step = False

    for line in plan_steps_text.splitlines():
        stripped = line.strip()

        if stripped.startswith("Step ") and stripped.endswith(":"):
            if buffer:
                steps.append("\n".join(buffer).strip())
                buffer = []
            in_step = True
            buffer.append(line)
            continue

        if in_step and stripped:
            buffer.append(line)

    if buffer:
        steps.append("\n".join(buffer).strip())

    if not steps:
        raise AnalysisExecutionAgentError("No executable steps found")

    return steps


def run_analysis_execution_agent(
    cleaned_dataset_path: str,
    analysis_plan_path: str,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    LLM-based execution of an analysis plan.
    """

    df = pd.read_csv(cleaned_dataset_path)
    plan_steps = load_analysis_plan(analysis_plan_path)

    state = ExecutionState(df=df, plan=plan_steps)

    print("\n=== ANALYSIS EXECUTION AGENT STARTED ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Total steps: {len(plan_steps)}")

    # --------------------------------------------------
    # MAIN STEP LOOP
    # --------------------------------------------------
    while state.has_more_steps():
        step_text = state.current_step()
        step_number = state.step_index + 1

        print(f"\n▶ Executing Step {step_number}")
        print("-" * 60)
        print(step_text)

        last_error = None

        # --------------------------------------------------
        # RETRY LOOP (BREAK ON FIRST SUCCESS)
        # --------------------------------------------------
        while state.attempt <= MAX_RETRIES:
            print(f"  Attempt {state.attempt}")

            before_df = state.df.copy()

            code = generate_code_for_step(
                full_plan=plan_steps,
                step_number=step_number,
                step_text=step_text,
                df=state.df,
                feedback=last_error,
            )

            if DEBUG_LLM_CODE:
                print("\n🧠 LLM GENERATED CODE:")
                print("-" * 50)
                print(code)
                print("-" * 50)

            result = execute_analysis_code.run({
                "df": state.df,
                "code": code,
            })

            # -----------------------------
            # EXECUTION ERROR → RETRY
            # -----------------------------
            if result["status"] == "error":
                last_error = result["error"]
                state.record({
                    "step": step_number,
                    "attempt": state.attempt,
                    "status": "execution_error",
                    "error": last_error,
                    "generated_code": code,
                })
                state.attempt += 1
                continue

            after_df = result["df"]
            metrics = evaluate_step(before_df, after_df)

            # -----------------------------
            # HARD FAILURE: EMPTY DF
            # -----------------------------
            if after_df.empty:
                last_error = "Resulting DataFrame is empty"
                state.record({
                    "step": step_number,
                    "attempt": state.attempt,
                    "status": "rejected",
                    "metrics": metrics,
                    "generated_code": code,
                })
                state.attempt += 1
                continue

            # -----------------------------
            # SOFT WARNING ONLY
            # -----------------------------
            if metrics["row_drop_pct"] > 90:
                print(
                    f"⚠️ Large row reduction detected "
                    f"({metrics['row_drop_pct']}%) — expected for analysis"
                )

            # -----------------------------
            # ✅ ACCEPT & BREAK (CRITICAL FIX)
            # -----------------------------
            state.df = after_df
            state.record({
                "step": step_number,
                "attempt": state.attempt,
                "status": "accepted",
                "metrics": metrics,
                "generated_code": code,
            })

            print("  ✅ Step accepted")
            print(f"  📐 Shape before: {before_df.shape}")
            print(f"  📐 Shape after : {after_df.shape}")

            break  # 🔥 STOP RETRIES ON FIRST SUCCESS

        else:
            # This executes ONLY if the loop never broke
            raise AnalysisExecutionAgentError(
                f"Step {step_number} failed after {MAX_RETRIES} attempts"
            )

        # --------------------------------------------------
        # MOVE TO NEXT STEP
        # --------------------------------------------------
        state.advance_step()

    # --------------------------------------------------
    # SAVE FINAL RESULT
    # --------------------------------------------------
    output_path = (
        Path(output_path)
        if output_path
        else Path(__file__).parent / "analysis_result.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state.df.to_csv(output_path, index=False)

    print("\n🎉 ANALYSIS EXECUTION COMPLETE")
    print(f"💾 Result saved to: {output_path}")
    print(f"Final shape: {state.df.shape}")

    return state.df


# ----------------------------------------------------------------------
# LOCAL MANUAL RUN (DEVELOPER ONLY)
# ----------------------------------------------------------------------
if __name__ == "__main__":

    print("\n🚀 MANUAL RUN: ANALYSIS EXECUTION AGENT\n")

    CLEANED_DATASET_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS"
        r"\data_cleaning\execution_agent\cleaned_dataset.csv"
    )

    ANALYSIS_PLAN_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS"
        r"\data_analysis\outputs\analysis_plan.txt"
    )

    OUTPUT_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS"
        r"\data_analysis\execution_agent\outputs\analysis_result.csv"
    )

    try:
        final_df = run_analysis_execution_agent(
            cleaned_dataset_path=CLEANED_DATASET_PATH,
            analysis_plan_path=ANALYSIS_PLAN_PATH,
            output_path=OUTPUT_PATH,
        )

        print("\n📊 FINAL DATAFRAME PREVIEW")
        print(final_df.head())

    except Exception as e:
        print("\n❌ ANALYSIS EXECUTION FAILED")
        print(e)
