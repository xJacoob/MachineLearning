"""
Polish sentiment - fine-tuning Transformer encoders (HerBERT, XLM-R) with the
Hugging Face Trainer on PolEmo 2.0 (KLEJ).

Train on polemo2-in/train, pick the best epoch on polemo2-in/validation,
evaluate once on polemo2-in/test (in-domain) and polemo2-out/test (out-of-domain).

Set SMOKE_TEST = True for a ~1 minute end-to-end check on the 200 longest
training texts (worst case for GPU memory).
"""

import numpy as np
import torch
import wandb
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

MAX_LENGTH = 512
SMOKE_TEST = False
SEED = 42

# --- Shared for all models: data and labels ----------------------------------

ds_in = load_dataset("allegro/klej-polemo2-in")
ds_out = load_dataset("allegro/klej-polemo2-out")

label_names = sorted(set(ds_in["train"]["target"]))
label2id = {name: idx for idx, name in enumerate(label_names)}
id2label = {idx: name for idx, name in enumerate(label_names)}


def encode_labels(example):
    example["labels"] = label2id[example["target"]]
    return example


ds_in = ds_in.map(encode_labels)
ds_out = ds_out.map(encode_labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

# --- Model-specific: tokenizer, model, training, evaluation ------------------

def train(model_name: str, run_name: str, epochs: int = 2, lr: float = 2e-5,
          batch_size: int = 8, grad_accum: int = 2) -> dict:
    """Fine-tune one model and evaluate it in- and out-of-domain."""
    set_seed(SEED)
    if SMOKE_TEST:
        run_name = f"{run_name}-smoke"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # token length statistics (each model has its own tokenizer)
    lengths = np.array([len(ids) for ids in tokenizer(list(ds_in["train"]["sentence"]))["input_ids"]])
    print(f"\n[{run_name}] tokens - mean: {lengths.mean():.0f}, max: {lengths.max()}, "
          f"longer than {MAX_LENGTH}: {(lengths > MAX_LENGTH).mean():.1%}")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)

    tok_in = ds_in.map(tokenize, batched=True)
    tok_out = ds_out.map(tokenize, batched=True)

    train_ds = tok_in["train"]
    val_ds = tok_in["validation"]
    test_in_ds = tok_in["test"]
    test_out_ds = tok_out["test"]

    if SMOKE_TEST:
        longest = np.argsort(lengths)[-200:]          # worst case for GPU memory
        train_ds = train_ds.select(longest.tolist())
        val_ds = val_ds.select(range(100))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=f"checkpoints/{run_name}",
        run_name=run_name,
        num_train_epochs=1 if SMOKE_TEST else epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=grad_accum,
        weight_decay=0.01,
        warmup_steps=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        logging_steps=20,
        fp16=True,
        report_to="wandb",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = {
        "val": trainer.evaluate(val_ds, metric_key_prefix="val"),
        "test_in": trainer.evaluate(test_in_ds, metric_key_prefix="test_in"),
        "test_out": trainer.evaluate(test_out_ds, metric_key_prefix="test_out"),
    }

    if not SMOKE_TEST:
        trainer.save_model(f"models/{run_name}")
        tokenizer.save_pretrained(f"models/{run_name}")

    # close this W&B run, otherwise the next model would log into the same run
    wandb.finish()

    # free GPU memory before the next model
    del model, trainer
    torch.cuda.empty_cache()

    return results


def print_summary(run_name: str, results: dict) -> None:
    v, ti, to = results["val"], results["test_in"], results["test_out"]
    print(f"\n=== {run_name} ===")
    print(f"val      macro F1: {v['val_f1_macro']:.3f}")
    print(f"in-test  accuracy: {ti['test_in_accuracy']:.3f}, macro F1: {ti['test_in_f1_macro']:.3f}")
    print(f"out-test accuracy: {to['test_out_accuracy']:.3f}, macro F1: {to['test_out_f1_macro']:.3f}")


if __name__ == "__main__":
    experiments = [
        ("allegro/herbert-base-cased", "herbert-2ep"),
        ("FacebookAI/xlm-roberta-base", "xlmr-2ep"),
    ]
    all_results = {}
    for model_name, run_name in experiments:
        all_results[run_name] = train(model_name, run_name)

    for run_name, results in all_results.items():
        print_summary(run_name, results)