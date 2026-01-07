from pathlib import Path
from typing import Optional
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# PROMPTS (LOCKED — FINAL)
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an Analysis Plan Reconciliation Agent.

Your task is to convert a natural-language ANALYSIS PLAN
into a STRICT, MACHINE-EXECUTABLE STRUCTURED REPRESENTATION.

IMPORTANT:
- You MUST NOT execute anything
- You MUST NOT invent steps
- You MUST NOT change intent
- You MUST NOT choose statistics or thresholds
- You MUST NOT optimize or simplify steps

You MUST:
- Normalize column names to match the dataset profile exactly
- Remove formatting artifacts (backticks, bullets, markdown)
- Explicitly name derived columns
- Preserve step order exactly
- Explicitly state uncertainty where present

OUTPUT RULES:
- Output MUST be STRUCTURED TEXT
- One section per field
- No markdown
- No explanations
- No code blocks

This output will be parsed later by a deterministic parser.
"""

USER_PROMPT_TEMPLATE = """
DATASET PROFILE (AUTHORITATIVE):
--------------------------------
{dataset_profile}

ANALYSIS UNDERSTANDING:
----------------------
{analysis_understanding}

ANALYSIS PLAN (RAW):
-------------------
{analysis_plan}

Produce a normalized, execution-ready plan
using clear structured text.
"""


# ----------------------------------------------------------------------
# CORE FUNCTION
# ----------------------------------------------------------------------

def run_analysis_plan_reconciler(
    dataset_profile_path: str,
    analysis_understanding_path: str,
    analysis_plan_path: str,
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
) -> str:
    """
    Uses LLM to normalize analysis_plan.txt into
    a structured TEXT representation (not JSON).
    """

    dataset_profile_text = Path(dataset_profile_path).read_text(encoding="utf-8")
    analysis_understanding_text = Path(analysis_understanding_path).read_text(
        encoding="utf-8"
    )
    analysis_plan_text = Path(analysis_plan_path).read_text(encoding="utf-8")

    final_prompt = USER_PROMPT_TEMPLATE.format(
        dataset_profile=dataset_profile_text,
        analysis_understanding=analysis_understanding_text,
        analysis_plan=analysis_plan_text,
    )

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\n{final_prompt}",
        generation_config={
            "temperature": 0.1,  # max determinism
        },
    )

    reconciled_text = response.text.strip()

    # ------------------------------------------------------------------
    # OUTPUT PATH
    # ------------------------------------------------------------------
    if output_path is None:
        output_path = Path(__file__).parent / "analysis_plan_normalized_raw.txt"
    else:
        output_path = Path(output_path)

    output_path.write_text(reconciled_text, encoding="utf-8")

    return reconciled_text


# ----------------------------------------------------------------------
# LOCAL TEST (DEVELOPER ONLY)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    BASE_PATH = r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_analysis"

    result = run_analysis_plan_reconciler(
        dataset_profile_path=f"{BASE_PATH}\\dataset_profile.txt",
        analysis_understanding_path=f"{BASE_PATH}\\analysis_understanding.txt",
        analysis_plan_path=f"{BASE_PATH}\\analysis_plan.txt",
    )

    print("\n🔁 ANALYSIS PLAN RECONCILED (RAW TEXT)\n")
    print(result[:1000])  # preview only
    print("\n✅ Saved as analysis_plan_normalized_raw.txt")
