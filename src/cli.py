import json
import os
from pathlib import Path
from typing import List

import click
from openai import OpenAI

from .annotate import annotate_long_text
from .data import load_entities, load_policy_texts
from .prepare import to_bio_tags, preprocess_records
from .train import NERConfig, train_ner

@click.group()
def cli():
    """LLM self-annotation -> supervised NER pipeline"""


@cli.command()
@click.option("--sqlite-path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--entities-path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--limit", type=int, default=5000)
@click.option("--out", "out_dir", type=click.Path(file_okay=False), required=True)
@click.option("--model", "llm_model", type=str, default="gpt-4o-mini")
def annotate(sqlite_path: str, entities_path: str, limit: int, out_dir: str, llm_model: str):
    os.makedirs(out_dir, exist_ok=True)
    client = OpenAI()
    entities = load_entities(entities_path)
    records = load_policy_texts(sqlite_path, limit=limit)

    records = preprocess_records(records)

    out_fp = Path(out_dir) / "annotations.jsonl"
    with out_fp.open("w", encoding="utf-8") as f:
        for rec in records:
            click.echo(f"Annotating {rec.id}...")
            res = annotate_long_text(
                rec.policy_text,
                entities=entities,
                client=client,
                model=llm_model,
            )
            line = {
                "id": rec.id,
                "domain": rec.site_domain,
                "year": rec.year,
                "phase": rec.phase,
                "text": rec.policy_text,
                "entities": res.get("entities", {}),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    click.echo(f"Wrote annotations to {out_fp}")


@cli.command()
@click.option("--annotations", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--entities-path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--train-out", type=click.Path(file_okay=False), required=True)
@click.option("--eval-split", type=float, default=0.1)
def prepare(annotations: str, entities_path: str, train_out: str, eval_split: float):
    os.makedirs(train_out, exist_ok=True)
    entities = load_entities(entities_path)
    texts: List[List[str]] = []
    tags: List[List[str]] = []
    with open(annotations, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            tok, lab = to_bio_tags(item["text"], item.get("entities", {}))
            texts.append(tok)
            tags.append(lab)
    # simple split
    n = len(texts)
    k = max(1, int(n * (1 - eval_split)))
    train_data = {"tokens": texts[:k], "tags": tags[:k]}
    eval_data = {"tokens": texts[k:], "tags": tags[k:]}
    with open(os.path.join(train_out, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f)
    with open(os.path.join(train_out, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(eval_data, f)
    with open(os.path.join(train_out, "entities.json"), "w", encoding="utf-8") as f:
        json.dump(entities, f)
    click.echo(f"Prepared train/eval in {train_out}")


@cli.command()
@click.option("--prepared", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", type=click.Path(file_okay=False), required=True)
@click.option("--base-model", type=str, default="bert-base-cased")
@click.option("--epochs", type=int, default=2)
@click.option("--lr", type=float, default=5e-5)
@click.option("--batch-size", type=int, default=8)
def train(prepared: str, output: str, base_model: str, epochs: int, lr: float, batch_size: int):
    with open(prepared, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(os.path.dirname(prepared), "eval.json"), "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(os.path.join(os.path.dirname(prepared), "entities.json"), "r", encoding="utf-8") as f:
        entities = json.load(f)

    cfg = NERConfig(
        base_model=base_model, num_epochs=epochs, learning_rate=lr, per_device_batch_size=batch_size
    )
    os.makedirs(output, exist_ok=True)
    train_ner(
        data["tokens"],
        data["tags"],
        eval_data.get("tokens", data["tokens"][-max(1, len(data["tokens"]) // 10):]),
        eval_data.get("tags", data["tags"][-max(1, len(data["tags"]) // 10):]),
        entities,
        output,
        cfg,
    )
    click.echo(f"Saved model to {output}")


@cli.command()
@click.option("--text", type=str, required=True)
@click.option("--model-dir", type=click.Path(exists=True, file_okay=True), required=True)
def predict(text: str, model_dir: str):
    from .predict import predict as do_predict

    out = do_predict(text, model_dir)
    print(out)

if __name__ == "__main__":
    cli()

