from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'clean_data'

data_train = pd.read_csv(DATA_DIR / 'train.csv')
X_train = data_train["Text"]
y_train = data_train['Class']

data_val = pd.read_csv(DATA_DIR / 'val.csv')
X_val = data_val["Text"]
y_val = data_val['Class']

data_test = pd.read_csv(DATA_DIR / 'test.csv')
X_test = data_test["Text"]
y_test = data_test['Class']

pipeline = {
    'default': Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000)),
    ]),
    'bigrams_sublinear_tf': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=1000)),
    ]),
    'analyzer_tf': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(2, 5), sublinear_tf=True, analyzer='char_wb')),
        ('clf', LogisticRegression(max_iter=1000)),
    ])
}

val_scores = {}

for name, pipe in pipeline.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_train)
    f1 = f1_score(y_train, preds)
    print(f"Name: {name} | f1_score: {f1}")

for name, pipe in pipeline.items():
    preds = pipe.predict(X_val)
    f1 = f1_score(y_val, preds)
    acc = accuracy_score(y_val, preds)
    val_scores[name] = f1
    print(f"Name: {name} | f1_score: {f1} | Accuracy: {acc}")

best_baseline_model = max(val_scores, key=val_scores.get)
print(f"Best baseline model: {best_baseline_model}")


best_model = pipeline[best_baseline_model]

print(f"\n================= Best model: {best_baseline_model} =================")
preds = best_model.predict(X_test)
f1 = f1_score(y_test, preds)
acc = accuracy_score(y_test, preds)
print(f"f1_score: {f1} | Accuracy: {acc}")
print(f"\n{classification_report(y_test, preds, digits=3)}")
