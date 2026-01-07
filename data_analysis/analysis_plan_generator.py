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
You are an Analysis Plan Generation Agent.

Your task is to convert:
1) A factual dataset profile
2) A structured analysis understanding
3) A user analysis query

into a clear, ordered, executable ANALYSIS PLAN.

The plan must describe WHAT operations are required,
NOT HOW they are computed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT NON-NEGOTIABLE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- You must NOT write code
- You must NOT compute statistics
- You must NOT choose statistical measures (e.g., mean, median, sum)
- You must NOT suggest plots, models, or algorithms
- You must NOT define thresholds, bins, or numeric cutoffs
- You must NOT assume causality
- You must NOT invent columns or data
- You must NOT introduce external knowledge
- You must NOT optimize, simplify, or recommend execution strategies

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALLOWED BEHAVIOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MAY:
- Translate supported analysis capabilities into ordered steps
- Reference operations ONLY at a structural level
  (e.g., grouping, aggregation, segmentation, comparison)
- Describe required inputs and expected outputs per step
- Note execution considerations without recommendations
- Explicitly acknowledge uncertainty when present

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DISCIPLINE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1) AGGREGATION DISCIPLINE
   - Aggregation may be referenced ONLY structurally
   - You must NOT name or imply any statistical function
   - Use neutral phrasing such as:
     “Produce an aggregated representation”
     “Apply an aggregation operation”

2) SEGMENTATION DISCIPLINE
   - Segmentation must be described structurally
   - You must NOT define thresholds, bins, ranges, or labels
   - Use phrasing such as:
     “Partition records into discrete groups using a consistent rule”
   - If segmentation criteria are unknown, state this explicitly

3) HIGH-CARDINALITY AWARENESS
   - If a step involves high-cardinality columns
     (e.g., identifiers, names, artists),
     you MUST explicitly note this as an execution consideration
   - You must NOT suggest how to handle it

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUNDING REQUIREMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Every step MUST be grounded in:
  - the dataset profile
  - the analysis understanding
- If something is not supported, state exactly:
  “Not evident from provided inputs”

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST produce output using the following structure EXACTLY:

ANALYSIS PLAN
=============

PLAN OBJECTIVE
--------------

PLAN ASSUMPTIONS
----------------

PLAN STEPS
----------
Step 1:
- Description:
- Input Columns:
- Operation Type:
- Expected Output:

(repeat for each step)

PLAN CONSTRAINTS
----------------

END OF ANALYSIS PLAN

Deviation from this format is NOT allowed.
"""

USER_PROMPT_TEMPLATE = """
DATASET PROFILE (FACT AUTHORITY):
--------------------------------
{dataset_profile}

ANALYSIS UNDERSTANDING (INTENT AUTHORITY):
-----------------------------------------
{analysis_understanding}

Produce the analysis plan strictly using the format below.

OUTPUT FORMAT (EXACT):

ANALYSIS PLAN
=============

PLAN OBJECTIVE
--------------
- Objective:

PLAN ASSUMPTIONS
----------------
- Assumption 1:
- Assumption 2:

PLAN STEPS
----------
Step 1:
- Description:
- Input Columns:
- Operation Type:
- Expected Output:

Step 2:
- Description:
- Input Columns:
- Operation Type:
- Expected Output:

Step 3:
- Description:
- Input Columns:
- Operation Type:
- Expected Output:

PLAN CONSTRAINTS
----------------
- Constraint 1:
- Constraint 2:

END OF ANALYSIS PLAN
"""

# ----------------------------------------------------------------------
# CORE FUNCTION
# ----------------------------------------------------------------------

def run_analysis_plan_generator(
    dataset_profile_path: str,
    analysis_understanding_path: str,
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
) -> str:
    """
    Reads dataset_profile.txt and analysis_understanding.txt,
    sends them to Gemini, and writes analysis_plan.txt.

    This function performs NO execution and NO analysis.
    """

    dataset_profile_text = Path(dataset_profile_path).read_text(encoding="utf-8")
    analysis_understanding_text = Path(analysis_understanding_path).read_text(
        encoding="utf-8"
    )

    final_prompt = f"""
{SYSTEM_PROMPT}

{USER_PROMPT_TEMPLATE.format(
    dataset_profile=dataset_profile_text,
    analysis_understanding=analysis_understanding_text
)}
"""

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        final_prompt,
        generation_config={"temperature": 0.2},
    )

    plan_text = response.text.strip()

    if output_path is None:
        output_path = Path(__file__).parent / "analysis_plan.txt"
    else:
        output_path = Path(output_path)

    output_path.write_text(plan_text, encoding="utf-8")

    return plan_text


# ----------------------------------------------------------------------
# LOCAL TEST (DEVELOPER ONLY)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    DATASET_PROFILE_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_analysis\dataset_profile.txt"
    )

    ANALYSIS_UNDERSTANDING_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_analysis\analysis_understanding.txt"
    )

    result = run_analysis_plan_generator(
        dataset_profile_path=DATASET_PROFILE_PATH,
        analysis_understanding_path=ANALYSIS_UNDERSTANDING_PATH,
    )

    print("\n📋 ANALYSIS PLAN OUTPUT\n" + "-" * 60)
    print(result)
    print("\n✅ Saved to same directory as analysis_plan_generator.py")
