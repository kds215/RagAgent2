from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever

"""
def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
"""

def retrieve(state: dict):
    print("---RETRIEVE---")
    question = state["question"]
    # ... your retrieval logic ...
    documents = retriever.invoke(question)
    print(f"---RETRIEVED {len(documents)} DOCUMENTS ---")

    # You MUST return a dictionary with the key defined in your State
    return {"documents": documents, "question": question}

