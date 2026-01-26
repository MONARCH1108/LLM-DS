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
# Analysis execution agent import (ADD-ON)
# ----------------------------
from data_analysis.execution_agent.agent import run_analysis_execution_agent


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
# EVALUATION ORCHESTRATION (NO TOOL, NO LLM)
# ============================================================

def run_evaluation_after_cleaning(
    cleaned_dataset_path: str,
    user_query: str,
) -> Dict[str, str]:
    """
    Runs evaluation pipeline on cleaned dataset.
    NO execution. NO interpretation.
    """

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
# ANALYSIS EXECUTION ORCHESTRATION (ADD-ON)
# ============================================================

def run_analysis_execution_after_evaluation(
    cleaned_dataset_path: str,
    analysis_plan_path: str,
) -> pd.DataFrame:
    """
    Runs the analysis execution agent AFTER evaluation.
    """

    print("\n--- Running Analysis Execution Agent ---\n")

    OUTPUT_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_analysis\execution_agent\outputs\analysis_result.csv"
    )

    final_df = run_analysis_execution_agent(
        cleaned_dataset_path=cleaned_dataset_path,
        analysis_plan_path=analysis_plan_path,
        output_path=OUTPUT_PATH,   # ✅ FIX
    )

    print("\nAnalysis execution completed successfully.")
    print(f"Final dataframe shape: {final_df.shape}")
    print(f"Result saved to: {OUTPUT_PATH}")

    return final_df



# ============================================================
# MAIN ORCHESTRATOR (LINEAR, DEFAULT)
# ============================================================

def Agent():
    """
    Central orchestrator (CURRENT STAGE):

    User Input
      → Cleaning (plan + execution)
      → Evaluation pipeline (profile → understanding → plan)
      → Analysis execution agent
      → STOP
    """

    # 1) Greet user
    greeting = greet_tool.run({})
    print("AI:", greeting)

    # 2) User inputs
    dataset_path = input("Give path to your dataset: ").strip()
    user_query = input("Please provide your query: ").strip()

    # 3) Dataset validation
    ack = receive_data_and_query.run({
        "path": dataset_path,
        "query": user_query
    })
    print("AI:", ack)

    # 4) Ingest validation
    ingest_ack = ingest_tool.run({
        "path": dataset_path,
        "query": user_query
    })
    print("AI (ingest):", ingest_ack)

    # 5) Cleaning pipeline
    cleaning_ack = cleaning_pipeline_tool.run({
        "dataset_path": dataset_path
    })
    print("AI (cleaning):", cleaning_ack)

    CLEANED_DATASET_PATH = (
        r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\execution_agent\cleaned_dataset.csv"
    )

    # 6) Evaluation pipeline
    evaluation_result = run_evaluation_after_cleaning(
        cleaned_dataset_path=CLEANED_DATASET_PATH,
        user_query=user_query,
    )

    # 7) Analysis execution agent
    analysis_df = run_analysis_execution_after_evaluation(
        cleaned_dataset_path=CLEANED_DATASET_PATH,
        analysis_plan_path=evaluation_result["analysis_plan"],
    )

    return {
        "greeting": greeting,
        "ack": ack,
        "ingest_ack": ingest_ack,
        "cleaning_ack": cleaning_ack,
        "evaluation_result": evaluation_result,
        "analysis_df_shape": analysis_df.shape,
    }


# ============================================================
# OPTIONAL: LANGGRAPH VERSION (ADD-ON EXTENDED)
# ============================================================

def build_graph():
    """
    Optional LangGraph-based orchestrator.
    Mirrors:
      greet → ingest → cleaning → evaluation → analysis_execution → END
    """

    g = StateGraph(dict, input_schema=dict, output_schema=dict)

    def greet_node(state: dict) -> dict:
        reply = greet_tool.run({})
        state.setdefault("messages", []).append(AIMessage(content=reply))
        state["last_action"] = "greet"
        state["last_response"] = reply
        return state

    def ingest_node(state: dict) -> dict:
        path = state.get("dataset_path", "")
        query = state.get("user_query", "")
        resp = receive_data_and_query.run({
            "path": path,
            "query": query
        })
        state.setdefault("messages", []).append(AIMessage(content=resp))
        state["last_action"] = "ingest"
        state["last_response"] = resp
        return state

    def cleaning_node(state: dict) -> dict:
        path = state.get("dataset_path", "")
        resp = cleaning_pipeline_tool.run({
            "dataset_path": path
        })
        state.setdefault("messages", []).append(AIMessage(content=resp))
        state["last_action"] = "cleaning"
        state["last_response"] = resp
        return state

    def evaluation_node(state: dict) -> dict:
        cleaned_path = (
            r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\execution_agent\cleaned_dataset.csv"
        )
        query = state.get("user_query", "")

        result = run_evaluation_after_cleaning(
            cleaned_dataset_path=cleaned_path,
            user_query=query,
        )

        state["evaluation_result"] = result
        state["last_action"] = "evaluation"
        return state

    def analysis_execution_node(state: dict) -> dict:
        cleaned_path = (
            r"C:\Users\abhay\OneDrive\Desktop\LLM-DS\LLM-DS\data_cleaning\execution_agent\cleaned_dataset.csv"
        )

        plan_path = state["evaluation_result"]["analysis_plan"]

        df = run_analysis_execution_after_evaluation(
            cleaned_dataset_path=cleaned_path,
            analysis_plan_path=plan_path,
        )

        state["analysis_df_shape"] = df.shape
        state["last_action"] = "analysis_execution"
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
