import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def generate_code_for_step(
    step_text: str,
    df_sample,
    feedback: str | None = None
) -> str:
    """
    Uses Groq LLM to generate Pandas cleaning code for ONE step.

    Rules enforced:
    - df MUST be reassigned
    - NO inplace=True
    - NO prints, NO explanations
    - Code-only output
    - Retry-aware (feedback injected)
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    # ---- Optional retry feedback ----
    feedback_block = ""
    if feedback:
        feedback_block = f"""
PREVIOUS ATTEMPT FAILED WITH THIS FEEDBACK:
{feedback}

You MUST correct this failure in the new code.
"""

    # ---- Prompt ----
    prompt = f"""
You are a senior data engineer writing SAFE Pandas code.

You MUST follow ALL rules strictly.

=====================
TASK
=====================
Apply the following cleaning step to a pandas DataFrame named `df`.

Cleaning Step (plain text):
{step_text}

=====================
CRITICAL RULES (MANDATORY)
=====================
- You MUST reassign df (example: df = df.copy())
- NEVER use inplace=True
- NEVER rely on chained assignment
- NEVER print anything
- NEVER explain anything
- NEVER import new libraries
- Use ONLY pandas (pd) and df
- If no action is required, return df unchanged explicitly

=====================
SAFETY RULES
=====================
- Avoid dropping rows unless explicitly required
- Text imputation → use "Unknown"
- Numeric imputation → use median unless stated otherwise
- If condition says "skip", implement a conditional guard

=====================
RETRY CONTEXT
=====================
{feedback_block}

=====================
THINKING INSTRUCTION
=====================
Think step-by-step INTERNALLY to decide the safest operation.
DO NOT output your reasoning.
ONLY output executable Python code.

=====================
OUTPUT FORMAT
=====================
Return ONLY valid Python code.
The FINAL line must leave the cleaned dataframe assigned to `df`.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You write strict, safe, production Pandas code."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    code = response.choices[0].message.content.strip()

    # ---- Strip markdown fences if present ----
    if code.startswith("```"):
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()
    
    # --- Sanitize output ---
    code = response.choices[0].message.content.strip()
    
    # Remove markdown fences
    if code.startswith("```"):
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()
    
    # Remove leading "python" if present
    if code.lower().startswith("python"):
        code = code[len("python"):].strip()
    
    # Remove any import statements (LLM sometimes violates rules)
    lines = []
    for line in code.splitlines():
        if not line.strip().startswith("import"):
            lines.append(line)
    
    code = "\n".join(lines).strip()

    return code
