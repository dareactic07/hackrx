# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time

from rag import load_and_split_pdf, create_vectorstore, build_qa_chain, get_answers

app = FastAPI()

class QueryInput(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class QueryOutput(BaseModel):
    answers: List[dict]
    total_execution_time_seconds: float

@app.post("/api/v1/hackrx/run", response_model=QueryOutput)
def run_query(query: QueryInput):
    start_time = time.time()

    try:
        docs = load_and_split_pdf(query.documents)
        vectorstore = create_vectorstore(docs)
        chain = build_qa_chain(vectorstore)
        answers = get_answers(chain, query.questions)

        total_time = round(time.time() - start_time, 2)

        return {
            "answers": answers,
            "total_execution_time_seconds": total_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# main.py

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

