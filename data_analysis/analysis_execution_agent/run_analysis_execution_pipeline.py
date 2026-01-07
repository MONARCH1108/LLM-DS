from pathlib import Path

# -------------------------------
# IMPORT EXISTING MODULES
# -------------------------------
from analysis_plan_reconciler import run_analysis_plan_reconciler
from analysis_execution_loop import AnalysisExecutionLoop


# -------------------------------
# PATH CONFIGURATION (SINGLE SOURCE OF TRUTH)
# -------------------------------
from pathlib import Path

# -------------------------------
# BASE PATHS
# -------------------------------
PROJECT_ROOT = Path(
    r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS"
)

DATA_ANALYSIS_DIR = PROJECT_ROOT / "data_analysis"
OUTPUT_DIR = DATA_ANALYSIS_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# PIPELINE-1 OUTPUTS (AUTHORITATIVE INPUTS)
# -------------------------------
DATASET_PROFILE_PATH = OUTPUT_DIR / "dataset_profile.txt"
ANALYSIS_UNDERSTANDING_PATH = OUTPUT_DIR / "analysis_understanding.txt"
ANALYSIS_PLAN_PATH = OUTPUT_DIR / "analysis_plan.txt"

# -------------------------------
# PIPELINE-2 ARTIFACTS
# -------------------------------
NORMALIZED_PLAN_PATH = (
    DATA_ANALYSIS_DIR
    / "analysis_execution_agent"
    / "analysis_plan_normalized_raw.txt"
)

CLEANED_DATASET_PATH = (
    PROJECT_ROOT
    / "data_cleaning"
    / "execution_agent"
    / "cleaned_dataset.csv"
)

# -------------------------------
# PIPELINE EXECUTION
# -------------------------------
def run_execution_pipeline():
    print("\n🚀 STARTING ANALYSIS EXECUTION PIPELINE\n")

    # --------------------------------------------------
    # 1️⃣ RECONCILE ANALYSIS PLAN
    # --------------------------------------------------
    print("🔁 Reconciling analysis plan...")

    run_analysis_plan_reconciler(
        dataset_profile_path=str(DATASET_PROFILE_PATH),
        analysis_understanding_path=str(ANALYSIS_UNDERSTANDING_PATH),
        analysis_plan_path=str(ANALYSIS_PLAN_PATH),
        output_path=str(NORMALIZED_PLAN_PATH),
    )

    print(f"✅ Normalized plan saved → {NORMALIZED_PLAN_PATH}")

    # --------------------------------------------------
    # 2️⃣ EXECUTE ANALYSIS PLAN
    # --------------------------------------------------
    print("\n⚙️ Executing analysis steps...")

    loop = AnalysisExecutionLoop(
        cleaned_dataset_path=CLEANED_DATASET_PATH,
        normalized_plan_path=str(NORMALIZED_PLAN_PATH),
        output_dir=str(OUTPUT_DIR),
    )

    loop.run()

    print("\n🎯 ANALYSIS EXECUTION PIPELINE COMPLETED")
    print(f"📁 Outputs available at: {OUTPUT_DIR}\n")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run_execution_pipeline()
