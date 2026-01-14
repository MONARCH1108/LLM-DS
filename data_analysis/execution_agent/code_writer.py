import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def generate_code_for_step(
    full_plan: list[str],
    step_number: int,
    step_text: str,
    df,
    feedback: str | None = None,
) -> str:
    """
    LLM writes pandas code for ONE step,
    aware of entire plan and current dataframe.
    """

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    feedback_block = ""
    if feedback:
        feedback_block = f"""
PREVIOUS ATTEMPT FAILED WITH ERROR:
{feedback}

You MUST correct this in the new code.
"""

    prompt = f"""
You are a senior data analyst executing an analysis plan.

You are executing STEP {step_number} of the plan.

FULL ANALYSIS PLAN:
-------------------
{chr(10).join(full_plan)}

CURRENT STEP:
-------------
{step_text}

CURRENT DATAFRAME:
------------------
Columns: {list(df.columns)}
Shape: {df.shape}
Preview:
{df.head(5)}

STRICT RULES (MANDATORY):
- Use ONLY pandas
- NO imports
- NO file I/O
- NO prints
- NO plotting
- You MUST reassign df
- Output ONLY valid Python code

{feedback_block}

Think internally. Do NOT explain.

OUTPUT:
Return ONLY executable Python code.
The final line MUST assign the result to `df`.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You write safe, production pandas code."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    # --------------------------------------------------
    # SANITIZE LLM OUTPUT (CRITICAL FIX)
    # --------------------------------------------------

    code = response.choices[0].message.content.strip()

    # Remove markdown fences ```python ... ```
    if code.startswith("```"):
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()

    # Remove leading 'python'
    if code.strip().lower().startswith("python"):
        code = code.strip()[6:].strip()

    # Remove any import statements (hard safety)
    clean_lines = []
    for line in code.splitlines():
        if not line.strip().startswith("import"):
            clean_lines.append(line)

    code = "\n".join(clean_lines).strip()

    return code

