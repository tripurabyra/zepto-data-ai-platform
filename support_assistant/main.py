from fastapi import FastAPI
from schemas import AskRequest, AskResponse
from graph import app_graph


app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant is running"
    }


@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):

    result = app_graph.invoke(
        {
            "query": request.query,
            "intent": "",
            "answer": "",
            "sources": [],
            "confidence": 0.0
        }
    )


    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )