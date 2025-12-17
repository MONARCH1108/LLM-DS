from L1_metrics import run_level_1_checks
from L2_metrics import run_level_2_metrics, summarize_level_2_for_llm

"""
from data_cleaning.L1_metrics import run_level_1_checks
from data_cleaning.L2_metrics import run_level_2_metrics, summarize_level_2_for_llm
"""

import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# Gemini Planner LLM
# =========================================================

def query_gemini(prompt: str) -> str:
    """
    Calls Gemini for planning-only tasks.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in environment")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(prompt)
    return response.text.strip()


# =========================================================
# Cleaning Plan Generator (TEXT MODE)
# =========================================================

def generate_cleaning_plan(dataset_path: str) -> str:
    """
    Generates a human-readable, step-by-step data cleaning plan.
    Saves the plan to a text file for user review.
    """

    # ---------- Run Metrics ----------
    level_1 = run_level_1_checks(dataset_path)
    level_2_raw = run_level_2_metrics(dataset_path)
    level_2 = summarize_level_2_for_llm(level_2_raw)

    # ---------- Build LLM Prompt ----------
    prompt = f"""
You are a senior data engineer.

Your task is to create a DATA CLEANING PLAN.

STRICT RULES:
- Do NOT write code
- Do NOT output JSON
- Do NOT analyze raw statistics
- ONLY produce a clear, numbered, step-by-step PLAN
- Each step must be concise and actionable

Dataset Overview:
- Rows: {level_1['row_count']}
- Columns: {level_1['column_count']}

Level-1 Data Quality Signals:
{level_1}

Level-2 Diagnostic Signals:
{level_2}

Instructions:
1. Write numbered steps (Step 1, Step 2, …).
2. Each step must include:
   - What to clean
   - Which columns are affected
   - When to skip the step
   - Severity (low / medium / high)
3. End with a short summary of overall cleaning complexity.

Output format example:

Step 1: Handle missing values  
Affected columns: ...  
Condition: ...  
Action: ...  
Severity: ...

DO NOT include anything else.
"""

    # ---------- Call Gemini ----------
    plan_text = query_gemini(prompt)

    # ---------- Persist Plan as TEXT ----------
    output_path = "cleaning_plan.txt"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "cleaning_plan.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plan_text)

    print(f"\n✅ Cleaning plan saved to: {output_path}")

    return plan_text


# =========================================================
# Temporary Test Runner
# =========================================================

if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\data\spotify.csv"

    plan_text = generate_cleaning_plan(DATASET_PATH)

    print("\n=== CLEANING PLAN (TEXT) ===\n")
    print(plan_text)