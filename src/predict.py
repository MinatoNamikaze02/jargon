from typing import List

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


def load_ner_pipeline(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def predict(text: str, model_dir: str):
    ner = load_ner_pipeline(model_dir)
    return ner(text)

