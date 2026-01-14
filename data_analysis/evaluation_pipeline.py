# data_analysis/evaluation_pipeline.py

from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
# IMPORT PIPELINE COMPONENTS
# ----------------------------------------------------------------------

from dataset_profile import dataset_profile
from profile_to_text import profile_to_text
from analysis_thinker import run_analysis_thinker
from analysis_plan_generator import run_analysis_plan_generator


# ----------------------------------------------------------------------
# PIPELINE ERROR
# ----------------------------------------------------------------------

class EvaluationPipelineError(Exception):
    """Fatal error in evaluation pipeline."""
    pass


# ----------------------------------------------------------------------
# CORE PIPELINE
# ----------------------------------------------------------------------

def run_evaluation_pipeline(
    cleaned_dataset_path: str,
    user_query: str,
    outputs_dir: str | None = None,
) -> dict:
    """
    Runs the full data_analysis generation pipeline in a fixed order.

    This pipeline:
    - Generates dataset_profile.txt
    - Generates analysis_understanding.txt
    - Generates analysis_plan.txt

    NO execution.
    NO parsing.
    NO interpretation.

    Returns:
        dict containing output file paths
    """

    # ------------------------------------------------------------------
    # PATH RESOLUTION
    # ------------------------------------------------------------------

    cleaned_dataset_path = Path(cleaned_dataset_path)

    if not cleaned_dataset_path.exists():
        raise EvaluationPipelineError(
            f"Cleaned dataset not found: {cleaned_dataset_path}"
        )

    if outputs_dir is None:
        outputs_dir = Path(__file__).parent / "outputs"
    else:
        outputs_dir = Path(outputs_dir)

    outputs_dir.mkdir(parents=True, exist_ok=True)

    dataset_profile_path = outputs_dir / "dataset_profile.txt"
    analysis_understanding_path = outputs_dir / "analysis_understanding.txt"
    analysis_plan_path = outputs_dir / "analysis_plan.txt"

    # ------------------------------------------------------------------
    # STEP 1: LOAD DATASET
    # ------------------------------------------------------------------

    try:
        df = pd.read_csv(cleaned_dataset_path)
    except Exception as e:
        raise EvaluationPipelineError(
            f"Failed to load cleaned dataset: {e}"
        )

    # ------------------------------------------------------------------
    # STEP 2: DATASET PROFILE (FACTUAL)
    # ------------------------------------------------------------------

    try:
        profile = dataset_profile(df)
    except Exception as e:
        raise EvaluationPipelineError(
            f"Dataset profiling failed: {e}"
        )

    try:
        profile_to_text(
            profile=profile,
            output_path=dataset_profile_path,
        )
    except Exception as e:
        raise EvaluationPipelineError(
            f"Failed to write dataset_profile.txt: {e}"
        )

    # ------------------------------------------------------------------
    # STEP 3: ANALYSIS THINKER (INTENT NORMALIZATION)
    # ------------------------------------------------------------------

    try:
        run_analysis_thinker(
            dataset_profile_path=str(dataset_profile_path),
            user_query=user_query,
            output_path=str(analysis_understanding_path),
        )
    except Exception as e:
        raise EvaluationPipelineError(
            f"Analysis thinker failed: {e}"
        )

    # ------------------------------------------------------------------
    # STEP 4: ANALYSIS PLAN GENERATOR (EXECUTION BLUEPRINT)
    # ------------------------------------------------------------------

    try:
        run_analysis_plan_generator(
            dataset_profile_path=str(dataset_profile_path),
            analysis_understanding_path=str(analysis_understanding_path),
            output_path=str(analysis_plan_path),
        )
    except Exception as e:
        raise EvaluationPipelineError(
            f"Analysis plan generation failed: {e}"
        )

    # ------------------------------------------------------------------
    # PIPELINE COMPLETE
    # ------------------------------------------------------------------

    return {
        "dataset_profile": str(dataset_profile_path),
        "analysis_understanding": str(analysis_understanding_path),
        "analysis_plan": str(analysis_plan_path),
    }


# ----------------------------------------------------------------------
# LOCAL TEST (RUN 3 TIMES MANUALLY)
# ----------------------------------------------------------------------

if __name__ == "__main__":

    CLEANED_DATASET_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\execution_agent\cleaned_dataset.csv"
    )
    USER_QUERY = "For each country, what is the average popularity and total streamcount of tracks released there?"

    print("\nRunning evaluation pipeline...\n")

    result = run_evaluation_pipeline(
        cleaned_dataset_path=CLEANED_DATASET_PATH,
        user_query=USER_QUERY,
    )

    print("Pipeline completed successfully\n")
    for k, v in result.items():
        print(f"{k}: {v}")
