from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer
from pathlib import Path
from datasets import load_dataset, Dataset
from sklearn.metrics import classification_report
from prepare_data import data_process

MAX_LENGTH = 256

TEST_DIR = Path(__file__).parent / "data" / "clean_data"
ROOT = Path(__file__).parent
CHECKPOINT_DIR = str(ROOT / "checkpoints" / "herbert" / "checkpoint-2400")

def tokenize(batch):
    return tokenizer(batch['Text'], truncation=True, max_length=MAX_LENGTH)

def evaluate_model(dataset):
    tokenized = dataset.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    output = trainer.predict(tokenized)
    preds = output.predictions.argmax(-1)
    print(classification_report(output.label_ids, preds, target_names=["non-harmful", "harmful"], digits=3))

if __name__ == "__main__":
    cbd = load_dataset("mteb/CBD", split="test")

    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    test_dataset = load_dataset("csv", data_files=str(TEST_DIR / "test.csv"), split="train")
    test_dataset = test_dataset.rename_column("Class", "label")
    print("=== BAN-PL ===")
    evaluate_model(test_dataset)

    cbd_df = cbd.to_pandas()
    cbd_df = cbd_df.rename(columns={"text": "Text"})
    cbd_df = data_process(cbd_df)
    cbd_clean = Dataset.from_pandas(cbd_df, preserve_index=False)

    print("=== CBD (original) ===")
    evaluate_model(cbd_clean)

    cbd_swapped = cbd_df.copy()
    cbd_swapped['Text'] = cbd_swapped['Text'].str.replace("@anonymized_account", "{USERNAME}", regex=False)
    cbd_swapped_clean = Dataset.from_pandas(data_process(cbd_swapped), preserve_index=False)
    print("=== CBD (@anonymized_account → {USERNAME}) ===")
    evaluate_model(cbd_swapped_clean)
