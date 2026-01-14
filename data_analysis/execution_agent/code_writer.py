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
You are a senior data analyst executing an analysis plan step-by-step.

You are executing STEP {step_number} of the plan.

FULL ANALYSIS PLAN:
-------------------
{chr(10).join(full_plan)}

CURRENT STEP (AUTHORITATIVE):
-----------------------------
{step_text}

CURRENT DATAFRAME (AUTHORITATIVE):
---------------------------------
- Type: pandas.DataFrame
- Columns: {list(df.columns)}
- Shape: {df.shape}
- Preview:
{df.head(5)}

======================
ABSOLUTE EXECUTION CONTRACT (NON-NEGOTIABLE)
======================

1. `df` MUST ALWAYS remain a pandas.DataFrame.
2. `df` MUST be fully materialized after this step.
3. Lazy objects are FORBIDDEN.

THE FOLLOWING ARE STRICTLY FORBIDDEN:
- df = df.groupby(...)
- df = some_groupby_object
- Creating synthetic or dummy data
- Using variables other than `df`
- Referring to `original_df`
- Returning partial or intermediate objects

IF GROUPING IS REQUIRED:
- You MUST combine it with an aggregation or transformation
- You MUST call `.reset_index()`
- The final result MUST be a DataFrame

VALID EXAMPLES:
- df = df.groupby([...]).agg(...).reset_index()
- df = df.groupby([...])[...].mean().reset_index(name="...")

INVALID EXAMPLES:
- df = df.groupby(...)
- grouped = df.groupby(...)

======================
SELF-CHECK (SILENT, INTERNAL)
======================

Before returning code, verify internally:
- Is `df` a pandas DataFrame at the end?
- Would `type(df)` be DataFrame (not GroupBy)?
- Does this code change ONLY the columns/rows implied by the step?
- Is there a simpler DataFrame-producing alternative if grouping feels ambiguous?

If the step description is underspecified:
- Choose the MOST CONSERVATIVE valid aggregation
- Prefer `count` or `mean` depending on column type

======================
STRICT RULES (MANDATORY):
======================
- Use ONLY pandas
- NO imports
- NO file I/O
- NO prints
- NO plotting
- You MUST reassign df exactly once
- Output ONLY valid Python code
- The FINAL LINE MUST assign the result to `df`

{feedback_block}

Do NOT explain.
Do NOT comment.
Do NOT add extra variables.

OUTPUT:
Return ONLY executable Python code.
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

