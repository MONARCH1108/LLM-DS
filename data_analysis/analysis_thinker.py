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
You are an Analysis Thinking Agent with recursive reasoning capabilities.

Your task is to produce a STRUCTURED, FACT-BOUNDED understanding of a dataset
based ONLY on:
1) A factual dataset profile
2) A user analysis query

You MUST reason internally using multiple recursive passes, but ALL reasoning
MUST remain internal and MUST NOT be revealed.

INTERNAL REASONING PASSES (DO NOT OUTPUT):
1) INITIAL PASS
   - Identify only factual signals present in the dataset profile.
   - Note column names, data types, cardinalities, and structural signals.
   - Tentatively infer dataset domain ONLY if strongly supported.

2) VERIFICATION PASS
   - Verify every inference strictly against the dataset profile.
   - Remove, weaken, or qualify any statement not explicitly supported.
   - If meaning or intent is unclear, mark it as:
     "Not evident from profile".

3) REFINEMENT PASS
   - Remove causal, associative, or predictive language.
   - Replace interpretation with capability-based descriptions.
   - Ensure strict adherence to the output schema and wording rules.

STRICT RULES (NON-NEGOTIABLE):
- You must NOT write code
- You must NOT compute or derive statistics
- You must NOT suggest plots, models, or algorithms
- You must NOT assume causality, influence, impact, or effect
- You must NOT invent column meanings or semantics
- You must NOT use external domain knowledge
- You must NOT perform execution

ALLOWED (WITH DISCIPLINE):
- Infer dataset domain ONLY if strongly supported by column names
- Assign column roles ONLY using structural evidence
- Describe SUPPORTED ANALYSIS CAPABILITIES (not directions, not conclusions)
- Note data quality observations grounded strictly in the profile

LANGUAGE CONSTRAINTS (VERY IMPORTANT):
- Use "supports analysis involving…" instead of "relationship", "association", or "impact"
- Use "can be grouped by", "can be compared across", "can be examined by"
- NEVER say: affects, influences, drives, explains, predicts
- If a semantic meaning is not explicitly evident, say:
  "Not evident from profile"

OUTPUT FORMAT MUST BE FOLLOWED EXACTLY.
Deviation is not allowed.
"""

USER_PROMPT_TEMPLATE = """
DATASET PROFILE:
----------------
{dataset_profile}

USER QUERY:
-----------
{user_query}

Produce the analysis understanding strictly using the format below.

OUTPUT FORMAT (EXACT):

### DATASET DOMAIN (IF SUPPORTED)
- Domain:
- Evidence:

### COLUMN ROLE ASSIGNMENT
- Identifier Columns:
- Candidate Target Columns (if any):
- Descriptive Metadata Columns:
- Categorical Columns:
- Numerical Measurement Columns:
- Temporal Columns:

### SUPPORTED ANALYSIS CAPABILITIES
- Capability 1:
- Capability 2:
- Capability 3:

### DATA QUALITY OBSERVATIONS
- Observation 1:
- Observation 2:

### LIMITATIONS AND UNKNOWN AREAS
- Limitation 1:
- Limitation 2:
"""

# ----------------------------------------------------------------------
# CORE FUNCTION
# ----------------------------------------------------------------------

def run_analysis_thinker(
    dataset_profile_path: str,
    user_query: str,
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
) -> str:
    """
    Reads dataset_profile.txt, sends it to Gemini,
    and writes analysis_understanding.txt.

    This function performs NO analysis itself.
    """

    # Load dataset profile text
    profile_text = Path(dataset_profile_path).read_text(encoding="utf-8")

    # Compose final prompt
    final_prompt = f"""
{SYSTEM_PROMPT}

{USER_PROMPT_TEMPLATE.format(
    dataset_profile=profile_text,
    user_query=user_query
)}
"""

    # Configure Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(
        final_prompt,
        generation_config={
            "temperature": 0.2,  # disciplined, low creativity
        },
    )

    analysis_text = response.text.strip()

    # ------------------------------------------------------------------
    # OUTPUT PATH RESOLUTION
    # ------------------------------------------------------------------
    if output_path is None:
        output_path = Path(__file__).parent / "analysis_understanding.txt"
    else:
        output_path = Path(output_path)

    output_path.write_text(analysis_text, encoding="utf-8")

    return analysis_text


# ----------------------------------------------------------------------
# LOCAL TEST (DEVELOPER ONLY)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    DATASET_PROFILE_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_analysis\dataset_profile.txt"
    )

    USER_QUERY = "Analyze what factors influence popularity"

    result = run_analysis_thinker(
        dataset_profile_path=DATASET_PROFILE_PATH,
        user_query=USER_QUERY,
    )

    print("\n🧠 ANALYSIS THINKER OUTPUT\n" + "-" * 60)
    print(result)
    print("\n✅ Saved to same directory as analysis_thinker.py")
