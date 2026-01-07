from pathlib import Path
import pandas as pd

# -------------------------------
# IMPORT YOUR EXISTING MODULES
# -------------------------------
from dataset_profile import dataset_profile
from profile_to_text import profile_to_text
from analysis_thinker import run_analysis_thinker
from analysis_plan_generator import run_analysis_plan_generator


# -------------------------------
# CONFIGURATION (ONLY PLACE TO EDIT PATHS)
# -------------------------------
BASE_DIR = Path(__file__).parent

CLEANED_DATASET_PATH = Path(
    r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\execution_agent\cleaned_dataset.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_PROFILE_TXT = OUTPUT_DIR / "dataset_profile.txt"
ANALYSIS_UNDERSTANDING_TXT = OUTPUT_DIR / "analysis_understanding.txt"
ANALYSIS_PLAN_TXT = OUTPUT_DIR / "analysis_plan.txt"

USER_QUERY = "Analyze what factors influence popularity"


# -------------------------------
# PIPELINE EXECUTION
# -------------------------------
def run_pipeline():
    print("\n🚀 Starting Full Analysis Pipeline\n")

    # 1️⃣ Load dataset
    print("📥 Loading cleaned dataset...")
    df = pd.read_csv(CLEANED_DATASET_PATH)
    print(f"✅ Dataset loaded | Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # 2️⃣ Generate dataset profile (dict)
    print("\n📊 Generating dataset profile...")
    profile = dataset_profile(df)
    print("✅ Dataset profile generated")

    # 3️⃣ Convert profile to deterministic text
    print("\n📝 Converting profile to text...")
    profile_to_text(
        profile=profile,
        output_path=str(DATASET_PROFILE_TXT),
    )
    print(f"✅ Dataset profile text saved → {DATASET_PROFILE_TXT}")

    # 4️⃣ Run Analysis Thinker
    print("\n🧠 Running Analysis Thinker...")
    run_analysis_thinker(
        dataset_profile_path=str(DATASET_PROFILE_TXT),
        user_query=USER_QUERY,
        output_path=str(ANALYSIS_UNDERSTANDING_TXT),
    )
    print(f"✅ Analysis understanding saved → {ANALYSIS_UNDERSTANDING_TXT}")

    # 5️⃣ Run Analysis Plan Generator
    print("\n📋 Running Analysis Plan Generator...")
    run_analysis_plan_generator(
        dataset_profile_path=str(DATASET_PROFILE_TXT),
        analysis_understanding_path=str(ANALYSIS_UNDERSTANDING_TXT),
        output_path=str(ANALYSIS_PLAN_TXT),
    )
    print(f"✅ Analysis plan saved → {ANALYSIS_PLAN_TXT}")

    print("\n🎯 PIPELINE COMPLETED SUCCESSFULLY\n")


# -------------------------------
# ENTRY POINT
# -------------------------------
if __name__ == "__main__":
    run_pipeline()
