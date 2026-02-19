from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState

"""
def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
"""

def generate(state: dict):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0) + 1

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": state["documents"], "question": question, "generation": generation, "retry_count": retry_count}