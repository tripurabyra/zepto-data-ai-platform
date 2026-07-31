from typing import TypedDict
from langgraph.graph import StateGraph, END

from sentence_transformers import SentenceTransformer
import chromadb


# -------------------------
# State
# -------------------------

class State(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list
    confidence: float



# -------------------------
# ChromaDB setup
# -------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


client = chromadb.PersistentClient(
    path="vectorstore/chroma_db"
)


collection = client.get_collection(
    "zepto_policy_collection"
)



# -------------------------
# Node 1: Intent classifier
# -------------------------

def classify_intent(state: State):

    keywords = [
    "delivery",
    "return",
    "refund",
    "membership",
    "pass",
    "zepto pass",
    "pass+",
    "pricing",
    "cost",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
    "replacement",
    "damaged",
    "order"
]


    query = state["query"].lower()


    if any(word in query for word in keywords):
        state["intent"] = "policy_question"

    else:
        state["intent"] = "general_question"


    return state



# -------------------------
# Node 2: Retrieval answer
# -------------------------


def retrieve_and_answer(state: State):

    embedding = model.encode(
        state["query"]
    ).tolist()


    result = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )


    top_chunk = result["documents"][0][0]
    source = result["ids"][0][0]


    state["answer"] = (
        "Based on the retrieved context: "
        + top_chunk[:200]
    )


    state["sources"] = [
        source
    ]

    state["confidence"] = 1.0


    return state



# -------------------------
# Node 3: Direct answer
# -------------------------

def direct_answer(state: State):

    state["answer"] = (
        "I can only answer questions about Zepto policies right now."
    )

    state["sources"] = []

    state["confidence"] = 1.0


    return state



# -------------------------
# Build LangGraph
# -------------------------

workflow = StateGraph(State)


workflow.add_node(
    "classify_intent",
    classify_intent
)


workflow.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)


workflow.add_node(
    "direct_answer",
    direct_answer
)



workflow.set_entry_point(
    "classify_intent"
)



workflow.add_conditional_edges(
    "classify_intent",
    lambda state: state["intent"],
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)



workflow.add_edge(
    "retrieve_and_answer",
    END
)


workflow.add_edge(
    "direct_answer",
    END
)



# IMPORTANT:
# This is what main.py imports

app_graph = workflow.compile()

print("Graph loaded successfully")