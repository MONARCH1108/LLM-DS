from pathlib import Path
from typing import Optional
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------
# SYSTEM PROMPT (LOCKED)
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an Analysis Runtime Interpreter Agent.

Your task is to interpret an ANALYSIS PLAN
and determine the NEXT EXECUTABLE STEP only.

You operate at STEP LEVEL.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES (NON-NEGOTIABLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- You must NOT execute anything
- You must NOT write code
- You must NOT invent steps
- You must NOT skip steps
- You must NOT reorder steps
- You must NOT optimize
- You must NOT assume results
- You must NOT modify intent

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU MUST DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Given:
1) A normalized analysis plan (text)
2) A list of already completed step numbers

You must decide:
- What is the NEXT step to execute
OR
- That execution is COMPLETE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST output STRUCTURED TEXT ONLY.

If a next step exists, output EXACTLY:

NEXT_STEP
STEP_NUMBER: <number>
OPERATION: <operation_name>
INPUT_COLUMNS:
- col1
- col2
DERIVED_COLUMNS:
- (if any, else NONE)
EXECUTION_NOTES:
- notes or uncertainty (if any)

If no steps remain, output EXACTLY:

EXECUTION_COMPLETE

No markdown.
No explanations.
No extra text.
"""

# ----------------------------------------------------------------------
# USER PROMPT TEMPLATE
# ----------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """
ANALYSIS PLAN (NORMALIZED, RAW TEXT):
------------------------------------
{analysis_plan}

COMPLETED STEPS:
----------------
{completed_steps}

Determine the NEXT executable step.
"""

# ----------------------------------------------------------------------
# CORE FUNCTION
# ----------------------------------------------------------------------

def run_analysis_runtime_interpreter(
    analysis_plan_normalized_path: str,
    completed_steps: Optional[list[int]] = None,
    model_name: str = "gemini-2.5-pro",
) -> str:
    """
    Determines the next executable analysis step using LLM reasoning.

    Returns structured TEXT instructions or EXECUTION_COMPLETE.
    """

    if completed_steps is None:
        completed_steps = []

    plan_text = Path(analysis_plan_normalized_path).read_text(
        encoding="utf-8"
    )

    completed_steps_text = (
        ", ".join(str(s) for s in completed_steps)
        if completed_steps
        else "NONE"
    )

    final_prompt = USER_PROMPT_TEMPLATE.format(
        analysis_plan=plan_text,
        completed_steps=completed_steps_text,
    )

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\n{final_prompt}",
        generation_config={
            "temperature": 0.1,  # max determinism
        },
    )

    return response.text.strip()


# ----------------------------------------------------------------------
# LOCAL TEST (DEVELOPER ONLY)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    BASE_PATH = r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_analysis\analysis_execution_agent\analysis_plan_normalized_raw.txt"

    output = run_analysis_runtime_interpreter(
        analysis_plan_normalized_path=BASE_PATH,
        completed_steps=[10],
    )

    print("\n🧭 NEXT STEP INTERPRETER OUTPUT\n" + "-" * 60)
    print(output)
