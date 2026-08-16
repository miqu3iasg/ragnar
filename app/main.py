# refs:
# docs: https://fastapi.tiangolo.com/#alternative-api-docs
# installation guide: https://fastapi.tiangolo.com/tutorial/#install-fastapi
from fastapi import FastAPI
from domain.research.question import Question

# You can ran the application using `uv run fastapi dev`
# server runs at http://127.0.0.1:8000 and Swagger documentation at http://127.0.0.1:8000/docs
app = FastAPI()


@app.post("/ask")
def ask_question(question: Question):
    return
