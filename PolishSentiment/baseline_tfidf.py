from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report


data_in = load_dataset("allegro/klej-polemo2-in")
data_out = load_dataset("allegro/klej-polemo2-out")

label_names = sorted(set(data_in["train"]["target"]))
label2id = {name: idx for idx, name in enumerate(label_names)}
id2label = {idx: name for idx, name in enumerate(label_names)}


def encode_labels(example):
    example["labels"] = label2id[example["target"]]
    return example


data_in = data_in.map(encode_labels)
data_out = data_out.map(encode_labels)

X_train = data_in["train"]["sentence"]
y_train = [label2id[target] for target in data_in["train"]["target"]]
X_test_in = data_in["test"]["sentence"]
y_test_in = [label2id[target] for target in data_in["test"]["target"]]
X_test_out = data_out["test"]["sentence"]
y_test_out = [label2id[target] for target in data_out["test"]["target"]]
X_val_in = data_in["validation"]["sentence"]
y_val_in = [label2id[target] for target in data_in["validation"]["target"]]

pipeline = {
    "default": Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000)),
    ]),
    "bigrams_sublinear_tf": Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000)),
    ]),
    "balanced": Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
}

val_scores = {}
for name, pipe in pipeline.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val_in)
    macro_f1 = f1_score(y_val_in, preds, average="macro")
    val_scores[name] = macro_f1
    print(f"{name}: val macro {macro_f1:.3f}")

best_name = max(val_scores, key=val_scores.get)
print(f"Best on validation: {best_name}")


def evaluate(pipeline, X, y, name):
    preds = pipeline.predict(X)

    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")

    print(f"Accuracy {name}: {accuracy:.3f}, F1 score {name}: {f1:.3f}")
    print(classification_report(y, preds, target_names=label_names))


best = pipeline[best_name]
evaluate(best, X_test_in, y_test_in, "in-domain")
evaluate(best, X_test_out, y_test_out, "out-of-domain")
