from typing import Optional

from requests import Response
from transformers import AutoModelForSeq2SeqLM

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers import pipeline
import logging
import json


model = AutoModelForSeq2SeqLM.from_pretrained(
        "model")

tokenizer = AutoTokenizer.from_pretrained(
        "model")

class Item(BaseModel):
    article: str

app = FastAPI()


@app.post("/items")
def read_item(item: Item):
    try:

        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

        return(summarizer(item.article))


    except Exception as e:

        return (str(e))
