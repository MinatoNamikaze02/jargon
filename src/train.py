from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import f1_score, precision_score, recall_score


@dataclass
class NERConfig:
    base_model: str = "bert-base-cased"
    num_epochs: int = 2
    learning_rate: float = 5e-5
    per_device_batch_size: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1


def build_label_list(entities: List[str]) -> List[str]:
    labels = ["O"]
    for ent in entities:
        labels.append(f"B-{ent}")
        labels.append(f"I-{ent}")
    return labels


def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512,
    )
    labels = []
    for i, label_seq in enumerate(examples["tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label_seq[word_idx]])
            else:
                # For subword tokens, assign I- tag if B- or I-
                label_str = label_seq[word_idx]
                if label_str.startswith("B-"):
                    label_str = label_str.replace("B-", "I-")
                label_ids.append(label_to_id[label_str])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def make_compute_metrics(id_to_label: Dict[int, str]):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_predictions = []
        for prediction, label in zip(predictions, labels):
            true_prediction = []
            true_label = []
            for p_i, l_i in zip(prediction, label):
                if l_i != -100:
                    true_prediction.append(id_to_label[int(p_i)])
                    true_label.append(id_to_label[int(l_i)])
            true_predictions.append(true_prediction)
            true_labels.append(true_label)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def train_ner(
    train_texts: List[List[str]],
    train_tags: List[List[str]],
    eval_texts: List[List[str]],
    eval_tags: List[List[str]],
    entity_list: List[str],
    output_dir: str,
    config: Optional[NERConfig] = None,
):
    if config is None:
        config = NERConfig()

    label_list = build_label_list(entity_list)
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model, num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id
    )

    train_ds = Dataset.from_dict({"tokens": train_texts, "tags": train_tags})
    eval_ds = Dataset.from_dict({"tokens": eval_texts, "tags": eval_tags})
    dsd = DatasetDict({"train": train_ds, "validation": eval_ds})

    tokenized = dsd.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
        remove_columns=["tokens", "tags"],
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id_to_label),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
