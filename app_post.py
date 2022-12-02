from typing import Optional

from requests import Response
from transformers import AutoModelForSeq2SeqLM

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
from transformers import pipeline
import uvicorn
import logging
import json
import logging


# model = AutoModelForSeq2SeqLM.from_pretrained(
#         "./TEXT_SUMMARY_MODEL")
#
# tokenizer = AutoTokenizer.from_pretrained(
#         "./TEXT_SUMMARY_MODEL")
#
# tokenizer.save_pretrained("C:\\Users\\Ashish.khandelwal\\ProjectWorkspace\\TextSummary\\TEXT_SUMMARY_MODEL_LOCAL")
# model.save_pretrained("C:\\Users\\Ashish.khandelwal\\ProjectWorkspace\\TextSummary\\TEXT_SUMMARY_MODEL_LOCAL")


# tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\Ashish.khandelwal\\ProjectWorkspace\\TextSummary\\TEXT_SUMMARY_MODEL_LOCAL")
# model = AutoModelForSeq2SeqLM.from_pretrained("C:\\Users\\Ashish.khandelwal\\ProjectWorkspace\\TextSummary\\TEXT_SUMMARY_MODEL_LOCAL")


logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained("TestZee/t5-small-finetuned-pytorch-final")
model = AutoModelForSeq2SeqLM.from_pretrained("TestZee/t5-small-finetuned-pytorch-final")

# tokenizer = AutoTokenizer.from_pretrained("my_model")
# model = AutoModelForSeq2SeqLM.from_pretrained("my_model")




class Item(BaseModel):
    article: str

app = FastAPI()


@app.post("/items")
def read_item(item: Item):
    try:
        logger.info("Entered into try")
        print(("Entered into try"))

        logger.info("creating summary pipeline")
        print(("creating summary pipeline"))

        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

        logger.info("summary pipeline created")
        print(("summary pipeline created"))

        return(summarizer(item.article))


    except Exception as e:

        return (str(e))


if __name__== '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')