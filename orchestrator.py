from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import query_groq_llm
from typing import TypedDict, List, Union
from langchain_core.tools import tool
import pandas as pd

from data_cleaning.plan_generator import generate_cleaning_plan
from data_cleaning.execution_agent.agent import run_execution_agent
    
# -------- Tools (explicit params) --------

@tool
def greet_tool() -> str:
    """
    Tool: ask the LLM to produce a short greeting asking for dataset + query.
    Call with greet_tool.run({}).
    """
    prompt = (
        "Greet the user briefly and ask them to provide (1) the dataset (CSV text or file path) "
        "and (2) the specific question/query they want answered. Keep it short."
    )
    return query_groq_llm(user_input=prompt, system_prompt="You are a helpful data assistant.")

@tool
def receive_data_and_query(path: str, query: str) -> str:
    """
    Tool: Load CSV from path and return an acknowledgement string.
    Call with receive_data_and_query.run({'path': path, 'query': query})
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
    Tool: Minimal ingest wrapper (same as receive_data_and_query but separate).
    Call with ingest_tool.run({'path': path, 'query': query})
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"Ingest failed for '{path}'. Error: {e}"
    return f"Ingested '{path}'. Rows: {len(df)}, Columns: {len(df.columns)}."

@tool
def cleaning_pipeline_tool(dataset_path: str)->str:
    """
    Tool: Generates a cleaning plan and executes the cleaning agent.
    
    :param dataset_path: Description
    :type dataset_path: str
    :return: Description
    :rtype: str
    """
    generate_cleaning_plan(dataset_path)
    cleaned_df = run_execution_agent(
        dataset_path=dataset_path,
        plan_path="data_cleaning/cleaning_plan.txt"
    )
    return (
        "Data cleaning completed successfully.\n"
        f"Final cleaned dataset shape: {cleaned_df.shape}"
    )

# -------- Agent runtime --------

def Agent():
    """
    Minimal interactive agent that:
      1) calls greet_tool (LLM) and prints the greeting
      2) asks the user for dataset path and query
      3) calls receive_data_and_query tool and prints acknowledgement
      4) calls ingest_tool and prints ack
      5) runs cleaning pipeline (plan + execution)
      6) asks the LLM to answer the user's query
    """

    # 1) Greet
    greeting = greet_tool.run({})
    print("AI:", greeting)

    # 2) User inputs
    dataset_path = input("Give Path to your dataset: ").strip()
    user_query = input("Please Provide Your Query: ").strip()

    # 3) Receive dataset + query
    ack = receive_data_and_query.run({
        "path": dataset_path,
        "query": user_query
    })
    print("AI:", ack)

    # 4) Ingest
    ingest_ack = ingest_tool.run({
        "path": dataset_path,
        "query": user_query
    })
    print("AI (ingest):", ingest_ack)

    # 5) Cleaning pipeline (PLAN → EXECUTION)
    cleaning_ack = cleaning_pipeline_tool.run({
        "dataset_path": dataset_path
    })
    print("AI (cleaning):", cleaning_ack)

    # 6) Final LLM answer
    prompt = (
        f"The dataset has been cleaned.\n"
        f"Original dataset path: {dataset_path}\n"
        f"User question: {user_query}\n\n"
        "Answer the user's question based on the cleaned dataset. "
        "If more analysis or visualization is required, explain."
    )

    final_answer = query_groq_llm(
        user_input=prompt,
        system_prompt="You are a data assistant. Answer concisely."
    )
    print("AI (final):", final_answer)

    return {
        "greeting": greeting,
        "ack": ack,
        "ingest_ack": ingest_ack,
        "cleaning_ack": cleaning_ack,
        "final_answer": final_answer,
    }

# If you still want a tiny graph version (optional)
def build_graph():
    g = StateGraph(dict, input_schema=dict, output_schema=dict)

    def greet_node(state: dict) -> dict:
        reply = greet_tool.run({})
        state.setdefault("messages", []).append(AIMessage(content=reply))
        state["last_action"] = "greet"
        state["last_response"] = reply
        return state

    def ingest_node_graph(state: dict) -> dict:
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

    def cleaning_node_graph(state: dict) -> dict:
        path = state.get("dataset_path", "")
        resp = cleaning_pipeline_tool.run({
            "dataset_path": path
        })
        state.setdefault("messages", []).append(AIMessage(content=resp))
        state["last_action"] = "cleaning"
        state["last_response"] = resp
        return state

    g.add_node("greet", greet_node)
    g.add_node("ingest", ingest_node_graph)
    g.add_node("cleaning", cleaning_node_graph)

    g.add_edge(START, "greet")
    g.add_edge("greet", "ingest")
    g.add_edge("ingest", "cleaning")
    g.add_edge("cleaning", END)

    return g.compile()

if __name__ == "__main__":
    Agent()
