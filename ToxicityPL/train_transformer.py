import numpy as np
import wandb
from datasets import load_dataset
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed

MAX_LENGTH = 256
SEED = 42
MODEL_NAME = 'allegro/herbert-base-cased'
DIR_DATA = Path(__file__).parent / "data" / "clean_data"
SMOKE_TEST = False

data = load_dataset(
    'csv',
    data_files={
        "train": str(DIR_DATA / "train.csv"),
        "val": str(DIR_DATA / "val.csv"),
        "test": str(DIR_DATA / "test.csv"),
    }
)

data = data.rename_column("Class", "label")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch['Text'], truncation=True, max_length=MAX_LENGTH)

tokenized = data.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

set_seed(SEED)
id2label = {0: "non-harmful", 1: "harmful"}
label2id = {label: idx for idx, label in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id.keys()),
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


if SMOKE_TEST:
    train_ds = tokenized["train"].select(range(64))
    val_ds = None

    args = TrainingArguments(
        output_dir="smoke_test",
        num_train_epochs=15,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )

else:
    train_ds = tokenized['train']
    val_ds = tokenized['val']

    args = TrainingArguments(
        output_dir="checkpoints/herbert",
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=0.1,
        weight_decay=0.01,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="wandb",
        seed=SEED,
        logging_steps=50,
    )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()