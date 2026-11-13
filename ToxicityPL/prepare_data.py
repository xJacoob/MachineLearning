import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_ROOT = Path(__file__).resolve().parent
DATA_DIR = DATA_ROOT / "data" / "BAN-PL" / "data"
data = pd.read_csv(DATA_DIR / "BAN_PL_1" / "BAN-PL.csv")
RESULT_DIR = Path(__file__).resolve().parent / "data" / "clean_data"

def data_process(df):
    df = df.copy()
    df['Text'] = df['Text'].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset='Text').reset_index(drop=True)
    return df

def split_data(df):
    df = df.copy()
    train, temp = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['Class'], random_state=42)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(RESULT_DIR / "train.csv", index=False, encoding='utf-8')
    val.to_csv(RESULT_DIR / "val.csv", index=False, encoding='utf-8')
    test.to_csv(RESULT_DIR / "test.csv", index=False, encoding='utf-8')
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(name, len(split), split["Class"].value_counts(normalize=True).round(3).to_dict())

clean = data_process(data)
split_data(clean)
print(len(data), '->', len(clean))
