# orchestrator.py

from typing import Dict, TypedDict, List, Union
import pandas as pd

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from utils.llm import query_groq_llm

# ----------------------------
# Cleaning imports
# ----------------------------
from data_cleaning.plan_generator import generate_cleaning_plan
from data_cleaning.execution_agent.agent import run_execution_agent

# ----------------------------
# Evaluation pipeline import
# ----------------------------
from data_analysis.evaluation_pipeline import run_evaluation_pipeline

# ----------------------------
# Analysis execution agent import
# ----------------------------
from data_analysis.execution_agent.agent import run_analysis_execution_agent


# ============================================================
# SESSION STATE (ADD-ON)
# ============================================================

class SessionState:
    """
    Maintains state across multiple user queries in one session.
    """
    def __init__(self):
        self.dataset_path: str | None = None
        self.cleaned_dataset_path: str | None = None
        self.is_cleaned: bool = False


# ============================================================
# TOOLS (ONLY WHERE LLM IS REQUIRED)
# ============================================================

@tool
def greet_tool() -> str:
    """
    Tool: ask the LLM to produce a short greeting asking for dataset + query.
    """
    prompt = (
        "Greet the user briefly and ask them to provide:\n"
        "1) Path to a CSV dataset\n"
        "2) The question they want answered\n"
        "Keep it short."
    )
    return query_groq_llm(
        user_input=prompt,
        system_prompt="You are a helpful data assistant."
    )


@tool
def receive_data_and_query(path: str, query: str) -> str:
    """
    Tool: Load CSV from path and return an acknowledgement string.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"Failed to load the dataset from '{path}'. Error: {e}"

    return (
        f"Dataset loaded successfully from '{path}'.\n"
        f"Rows: {len(df)}, Columns: {len(df.columns)}.\n"
        f"User query: '{query}'."
    )


@tool
def ingest_tool(path: str, query: str) -> str:
    """
    Tool: Minimal ingest wrapper.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"Ingest failed for '{path}'. Error: {e}"

    return f"Ingested '{path}'. Rows: {len(df)}, Columns: {len(df.columns)}."


@tool
def cleaning_pipeline_tool(dataset_path: str) -> str:
    """
    Tool: Generates a cleaning plan and executes the cleaning agent.
    """
    generate_cleaning_plan(dataset_path)

    cleaned_df = run_execution_agent(
        dataset_path=dataset_path,
        plan_path=r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\cleaning_plan.txt"
    )

    return (
        "Data cleaning completed successfully.\n"
        f"Final cleaned dataset shape: {cleaned_df.shape}"
    )


# ============================================================
# EVALUATION ORCHESTRATION
# ============================================================

def run_evaluation_after_cleaning(
    cleaned_dataset_path: str,
    user_query: str,
) -> Dict[str, str]:

    print("\n--- Running Evaluation Pipeline ---\n")

    result = run_evaluation_pipeline(
        cleaned_dataset_path=cleaned_dataset_path,
        user_query=user_query,
    )

    print("Evaluation pipeline completed successfully:\n")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result


# ============================================================
# ANALYSIS EXECUTION ORCHESTRATION
# ============================================================

def run_analysis_execution_after_evaluation(
    cleaned_dataset_path: str,
    analysis_plan_path: str,
) -> pd.DataFrame:

    print("\n--- Running Analysis Execution Agent ---\n")

    OUTPUT_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS"
        r"\data_analysis\execution_agent\outputs\analysis_result.csv"
    )

    final_df = run_analysis_execution_agent(
        cleaned_dataset_path=cleaned_dataset_path,
        analysis_plan_path=analysis_plan_path,
        output_path=OUTPUT_PATH,
    )

    print("\nAnalysis execution completed successfully.")
    print(f"Final dataframe shape: {final_df.shape}")
    print(f"Result saved to: {OUTPUT_PATH}")

    return final_df


# ============================================================
# MAIN ORCHESTRATOR (SESSION-BASED)
# ============================================================

def Agent():
    """
    Interactive session-based orchestrator.

    - Cleans dataset ONCE
    - Accepts multiple user queries
    - Reuses cleaned dataset
    """

    session = SessionState()

    # 1) Greet user
    greeting = greet_tool.run({})
    print("AI:", greeting)

    # 2) Dataset input (ONCE)
    session.dataset_path = input("Give path to your dataset: ").strip()

    # 3) Validate dataset
    ack = receive_data_and_query.run({
        "path": session.dataset_path,
        "query": "initial"
    })
    print("AI:", ack)

    ingest_ack = ingest_tool.run({
        "path": session.dataset_path,
        "query": "initial"
    })
    print("AI (ingest):", ingest_ack)

    # 4) CLEAN DATASET ONCE
    print("\n🧹 Cleaning dataset (one-time)...\n")

    cleaning_ack = cleaning_pipeline_tool.run({
        "dataset_path": session.dataset_path
    })
    print("AI (cleaning):", cleaning_ack)

    session.cleaned_dataset_path = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS"
        r"\data_cleaning\execution_agent\cleaned_dataset.csv"
    )
    session.is_cleaned = True

    # --------------------------------------------------
    # 🔁 QUERY LOOP (NO MORE CLEANING)
    # --------------------------------------------------
    while True:
        print("\n--------------------------------------")
        user_query = input("Ask a new question (or type 'exit'): ").strip()

        if user_query.lower() in {"exit", "quit"}:
            print("👋 Session ended.")
            break

        # Evaluation
        evaluation_result = run_evaluation_after_cleaning(
            cleaned_dataset_path=session.cleaned_dataset_path,
            user_query=user_query,
        )

        # Execution
        analysis_df = run_analysis_execution_after_evaluation(
            cleaned_dataset_path=session.cleaned_dataset_path,
            analysis_plan_path=evaluation_result["analysis_plan"],
        )

        print("\n✅ Query completed.")
        print(f"Result shape: {analysis_df.shape}")


# ============================================================
# OPTIONAL: LANGGRAPH VERSION (UNCHANGED)
# ============================================================

def build_graph():
    g = StateGraph(dict, input_schema=dict, output_schema=dict)

    def greet_node(state: dict) -> dict:
        reply = greet_tool.run({})
        state.setdefault("messages", []).append(AIMessage(content=reply))
        return state

    def ingest_node(state: dict) -> dict:
        resp = receive_data_and_query.run({
            "path": state.get("dataset_path", ""),
            "query": state.get("user_query", "")
        })
        state.setdefault("messages", []).append(AIMessage(content=resp))
        return state

    def cleaning_node(state: dict) -> dict:
        resp = cleaning_pipeline_tool.run({
            "dataset_path": state.get("dataset_path", "")
        })
        state.setdefault("messages", []).append(AIMessage(content=resp))
        return state

    def evaluation_node(state: dict) -> dict:
        result = run_evaluation_after_cleaning(
            cleaned_dataset_path=(
                r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS"
                r"\data_cleaning\execution_agent\cleaned_dataset.csv"
            ),
            user_query=state.get("user_query", ""),
        )
        state["evaluation_result"] = result
        return state

    def analysis_execution_node(state: dict) -> dict:
        df = run_analysis_execution_after_evaluation(
            cleaned_dataset_path=(
                r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS"
                r"\data_cleaning\execution_agent\cleaned_dataset.csv"
            ),
            analysis_plan_path=state["evaluation_result"]["analysis_plan"],
        )
        state["analysis_df_shape"] = df.shape
        return state

    g.add_node("greet", greet_node)
    g.add_node("ingest", ingest_node)
    g.add_node("cleaning", cleaning_node)
    g.add_node("evaluation", evaluation_node)
    g.add_node("analysis_execution", analysis_execution_node)

    g.add_edge(START, "greet")
    g.add_edge("greet", "ingest")
    g.add_edge("ingest", "cleaning")
    g.add_edge("cleaning", "evaluation")
    g.add_edge("evaluation", "analysis_execution")
    g.add_edge("analysis_execution", END)

    return g.compile()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    Agent()
