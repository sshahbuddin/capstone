from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
from datetime import datetime

from src import summary_pipeline

app = FastAPI()

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the retriever model from huggingface model hub
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum", device = 0) if torch.cuda.is_available() else pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

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
    return summary_pipeline.paragraph_summary_pipeline(json_compatible_tos["ToS"], model = retriever, summarizaer = summarizer)

@app.post("/privacy_summarize")
async def generate_summary(privacy:Privacy):
    json_compatible_privacy = jsonable_encoder(privacy)
    return summary_pipeline.paragraph_summary_pipeline(json_compatible_privacy["Privacy"], model = retriever, summarizer = summarizer)

@app.get("/health")
async def health():
    return datetime.now().isoformat()

