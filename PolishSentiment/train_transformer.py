"""
Polish sentiment - fine-tuning HerBERT with the Hugging Face Trainer.

Setup (once, in the MachineLearning project):
    uv add datasets transformers scikit-learn wandb accelerate
    uv run wandb login          # paste the API key from wandb.ai/authorize

Workflow:
1. First run with SMOKE_TEST = True  -> tiny subset, 1 epoch, ~1 minute.
   It only checks that the whole pipeline works end to end.
2. Then set SMOKE_TEST = False       -> the real training run.

Fill in every TODO. Each section ends with a CHECK.
"""

from collections import Counter

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "allegro/herbert-base-cased"
MAX_LENGTH = 512        # update after SECTION 3 (token length statistics)
SMOKE_TEST = False         # set to False for the real run
SEED = 42                 # fixed seed -> reproducible results (a known gap in Pneumonia)

# =============================================================================
# SECTION 1 - Data
# =============================================================================
# CONCEPT
# Train ONLY on polemo2-in. polemo2-out is used ONLY for the final test
# (out-of-domain). Validation (in) is used to pick the best epoch.

ds_in = load_dataset("allegro/klej-polemo2-in")
ds_out = load_dataset("allegro/klej-polemo2-out")

# =============================================================================
# SECTION 2 - Labels: strings -> integers
# =============================================================================
# CONCEPT
# The model outputs 4 logits, one per class, so every label must be an
# integer 0..3. We also keep both directions of the mapping:
#   label2id: "__label__meta_plus_m" -> 0
#   id2label: 0 -> "__label__meta_plus_m"
# The model stores id2label in its config, so later it can return names
# (exactly what you read from model.config.id2label in the HF exercises).

# TODO 2a: sorted list of the unique values in ds_in["train"]["target"]
label_names = sorted(set(ds_in["train"]["target"]))
# TODO 2b: dict comprehension: name -> index
label2id = {name: idx for idx, name in enumerate(label_names)}
# TODO 2c: dict comprehension: index -> name
id2label = {idx: name for idx, name in enumerate(label_names)}

def encode_labels(example):
    # TODO 2d: add a new field "labels" with the integer id of example["target"]
    #          and return the example
    example["labels"] = label2id[example["target"]]
    return example


ds_in = ds_in.map(encode_labels)
ds_out = ds_out.map(encode_labels)

print("\nSECTION 2")
print(label2id)
for split_name, split in [("in/train", ds_in["train"]), ("in/test", ds_in["test"]),
                          ("out/test", ds_out["test"])]:
    counts = Counter(split["labels"])
    total = len(split)
    print(split_name, {id2label[k]: f"{v} ({v / total:.1%})" for k, v in sorted(counts.items())})
# CHECK: 4 classes, labels 0..3; look at the percentages - is the data balanced?
#        Is the class distribution in out/test similar to in/test?

# =============================================================================
# SECTION 3 - Tokenizer and token length statistics
# =============================================================================
# CONCEPT
# HerBERT accepts at most 512 tokens. Longer texts are truncated.
# Shorter MAX_LENGTH = faster training and less GPU memory, but more texts
# lose their ending. Look at the numbers and choose MAX_LENGTH consciously.

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# TODO 3: list of token counts for every sentence in ds_in["train"]
#         (tokenize WITHOUT truncation; take len(...) of "input_ids")
lengths = [len(tokenizer(text)["input_ids"]) for text in ds_in["train"]["sentence"]]

lengths = np.array(lengths)
print("\nSECTION 3")
print(f"mean: {lengths.mean():.0f}, median: {np.median(lengths):.0f}, max: {lengths.max()}")
for limit in (128, 256, 512):
    print(f"longer than {limit}: {(lengths > limit).mean():.1%}")
# CHECK: choose MAX_LENGTH so that only a small share of texts is truncated,
#        then update the constant at the top of the file.

# =============================================================================
# SECTION 4 - Tokenize the whole dataset
# =============================================================================
# CONCEPT
# .map(..., batched=True) tokenizes many examples at once (fast).
# NO padding here: DataCollatorWithPadding pads each batch to the longest
# text IN THAT BATCH during training (dynamic padding) - much less wasted
# compute than padding everything to MAX_LENGTH.


def tokenize(batch):
    # TODO 4: tokenize batch["sentence"] with truncation=True and max_length=MAX_LENGTH
    tokens = tokenizer(batch["sentence"], truncation=True, max_length=MAX_LENGTH)
    return tokens


tok_in = ds_in.map(tokenize, batched=True)
tok_out = ds_out.map(tokenize, batched=True)

train_ds = tok_in["train"]
val_ds = tok_in["validation"]
test_in_ds = tok_in["test"]
test_out_ds = tok_out["test"]

if SMOKE_TEST:
    train_ds = train_ds.shuffle(seed=SEED).select(range(200))
    val_ds = val_ds.select(range(100))

collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("\nSECTION 4")
print(train_ds)
# CHECK: train_ds has columns input_ids, attention_mask and labels.

# =============================================================================
# SECTION 5 - Model
# =============================================================================
# CONCEPT
# Same idea as model.fc = nn.Linear(512, 2) in Pneumonia: the body is
# pretrained, the classification head is new and randomly initialized.
# You will see a warning that some weights were "newly initialized" - expected.

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id,
)


# =============================================================================
# SECTION 6 - Metrics
# =============================================================================
# CONCEPT
# Trainer calls compute_metrics after each evaluation with:
#   eval_pred.predictions -> logits, shape (N, 4)
#   eval_pred.label_ids   -> true labels, shape (N,)
# Use macro F1 as the main metric: it averages F1 over the 4 classes equally,
# so a weak minority class (e.g. "amb") cannot hide behind accuracy.


def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # TODO 6a: predicted class for each example (argmax over the class axis)
    preds = np.argmax(logits, axis=-1)
    # TODO 6b: return a dict with "accuracy" and "f1_macro"
    #          (accuracy_score, f1_score(..., average="macro"))
    metrics_dict = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }
    return metrics_dict


# =============================================================================
# SECTION 7 - Training arguments and training
# =============================================================================
# CONCEPT
# eval_strategy and save_strategy must match ("epoch") so the best epoch
# can be found and loaded at the end. fp16 = mixed precision on the GPU:
# roughly 2x faster and uses less memory.

args = TrainingArguments(
    output_dir="checkpoints/herbert",
    run_name="herbert-smoke" if SMOKE_TEST else "herbert",
    num_train_epochs=1 if SMOKE_TEST else 3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    warmup_steps=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # TODO 7a: which metric decides the best epoch? (key from compute_metrics)
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
    data_collator=collator,
    compute_metrics=compute_metrics,
)

trainer.train()


# =============================================================================
# SECTION 8 - Final evaluation: in-domain vs out-of-domain
# =============================================================================
# CONCEPT
# The test sets are used ONCE, at the very end, after the best epoch was
# chosen on validation. This is the answer to the project's research question.

# TODO 8: evaluate the trainer on test_in_ds and test_out_ds
#         (trainer.evaluate(dataset, metric_key_prefix="test_in") etc.)
results_in = trainer.evaluate(test_in_ds, metric_key_prefix="test_in")
results_out = trainer.evaluate(test_out_ds, metric_key_prefix="test_out")

print("\nSECTION 8")
print("in-domain :", results_in)
print("out-domain:", results_out)
# CHECK: compare test_in_f1_macro with test_out_f1_macro.
#        How much does the model lose outside its training domains?

if not SMOKE_TEST:
    trainer.save_model("models/herbert-polemo")
    tokenizer.save_pretrained("models/herbert-polemo")