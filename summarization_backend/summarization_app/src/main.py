from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder

import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from src import parallel_summarization_pipeline

app = FastAPI()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model = joblib.load("./src/model_pipeline.pkl")
model = SentenceTransformer('all-mpnet-base-v2')
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

class ToS(BaseModel):
    Company: str
    URL: str | None = None
    Industry: str | None = None
    ToS: str

class Privacy(BaseModel):
    Company: str
    URL: str | None = None
    Industry: str | None = None
    Privacy: str

@app.get("/")
async def root():
    raise HTTPException(status_code=404, detail="not implemented")  # endpoint not found

@app.post("/tos_summarize")
async def generate_summary(tos:ToS):
    json_compatible_tos = jsonable_encoder(tos)
    return parallel_summarization_pipeline.pooled_paragraph_summary_pipeline(json_compatible_tos["ToS"], model, summarizer)

@app.post("/privacy_summarize")
async def generate_summary(privacy:Privacy):
    json_compatible_privacy = jsonable_encoder(privacy)
    return parallel_summarization_pipeline.pooled_paragraph_summary_pipeline(json_compatible_privacy["Privacy"], model, summarizer)

@app.get("/health")
async def health():
    return datetime.now().isoformat()

