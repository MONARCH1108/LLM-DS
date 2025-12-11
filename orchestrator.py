# orchestrator.py
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm import query_groq_llm
from typing import TypedDict, List, Union
from langchain_core.tools import tool
import pandas as pd
    
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

# -------- Agent runtime --------

def Agent():
    """
    Minimal interactive agent that:
      1) calls greet_tool (LLM) and prints the greeting
      2) asks the user for dataset path and query
      3) calls receive_data_and_query tool and prints acknowledgement
      4) calls ingest_tool and prints ack
      5) asks the LLM to answer the user's query using the dataset summary
    """
    # 1) Greet (call structured tool with empty input)
    greeting = greet_tool.run({})   # StructuredTool expects a dict input
    print("AI:", greeting)

    # 2) Get user inputs
    dataset_path = input("Give Path to your dataset: ").strip()
    user_query = input("Please Provide Your Query: ").strip()

    # 3) Call the reception tool
    ack = receive_data_and_query.run({"path": dataset_path, "query": user_query})
    print("AI:", ack)

    # 4) Call ingest tool
    ingest_ack = ingest_tool.run({"path": dataset_path, "query": user_query})
    print("AI (ingest):", ingest_ack)

    # 5) Final LLM answer using the short summary
    prompt = (
        f"The user provided a dataset at path: {dataset_path}.\n"
        f"Dataset summary: {ack}\n"
        f"User's question: {user_query}\n\n"
        "Answer the user's question concisely based on the dataset summary above. "
        "If you need more data, ask for it."
    )
    final_answer = query_groq_llm(user_input=prompt, system_prompt="You are a data assistant. Answer concisely.")
    print("AI (final):", final_answer)

    return {
        "greeting": greeting,
        "ack": ack,
        "ingest_ack": ingest_ack,
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
        resp = receive_data_and_query.run({"path": path, "query": query})
        state.setdefault("messages", []).append(AIMessage(content=resp))
        state["last_action"] = "ingest"
        state["last_response"] = resp
        return state

    g.add_node("greet", greet_node)
    g.add_node("ingest", ingest_node_graph)
    g.add_edge(START, "greet")
    g.add_edge("greet", "ingest")
    g.add_edge("ingest", END)
    return g.compile()

if __name__ == "__main__":
    Agent()
