import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

SEED = 42
TARGET, ID_COL = "Churn", "customerID"
FEATURES = ["tenure", "MonthlyCharges"]
LABELS = [0, 1]  # 0 = No churn, 1 = Churn (positive class)

source = Path("data/raw/telco.csv")
out_dir = Path("data/output")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)

# --- 60/20/20 stratified split, one seed, test set reserved ---
train_df, rest_df = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=SEED)
valid_df, test_df = train_test_split(rest_df, test_size=0.5, stratify=rest_df[TARGET], random_state=SEED)

# split checks: disjoint rows, same class balance everywhere
assert len({*train_df.index, *valid_df.index, *test_df.index}) == len(df)
print("sizes:", len(train_df), len(valid_df), len(test_df))
for name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    print(f"  {name} churn rate: {(part[TARGET] == 'Yes').mean():.4f}")
id_overlap = len(set(train_df[ID_COL]) & set(valid_df[ID_COL])) + len(set(train_df[ID_COL]) & set(test_df[ID_COL]))
print("customerID shared between train and valid/test:", id_overlap)

# save the allocation so every later TP step reuses exactly the same customers
(out_dir / "split_ids.json").write_text(json.dumps({
    "seed": SEED,
    "train": train_df[ID_COL].tolist(),
    "valid": valid_df[ID_COL].tolist(),
    "test": test_df[ID_COL].tolist(),
}), encoding="utf-8")


def xy(part):
    X = part[FEATURES].apply(pd.to_numeric, errors="coerce")  # ensure numeric
    y = part[TARGET].map({"No": 0, "Yes": 1})
    return X, y


X_train, y_train = xy(train_df)
X_valid, y_valid = xy(valid_df)
assert X_train.notna().all().all() and X_valid.notna().all().all(), "NaN in features"
assert y_train.notna().all() and y_valid.notna().all(), "unexpected target value"

models = {
    "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=1000),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  # fitted on TRAIN only
    pred = model.predict(X_valid)  # evaluated on VALIDATION
    cm = confusion_matrix(y_valid, pred, labels=LABELS)
    results[name] = {
        "accuracy": float(accuracy_score(y_valid, pred)),
        "precision_churn": float(precision_score(y_valid, pred, zero_division=0)),
        "recall_churn": float(recall_score(y_valid, pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }
    print(f"\n== {name} ==")
    print("accuracy :", round(results[name]["accuracy"], 4))
    print("precision:", round(results[name]["precision_churn"], 4), "| recall:", round(results[name]["recall_churn"], 4))
    print(pd.DataFrame(cm, index=["actual 0 (No)", "actual 1 (Yes)"],
                       columns=["pred 0 (No)", "pred 1 (Yes)"]))

(out_dir / "first_model_report.json").write_text(json.dumps({
    "positive_class": "Churn = Yes (1)",
    "matrix_labels": {"order": LABELS, "rows": "actual", "columns": "predicted"},
    "features": FEATURES,
    "seed": SEED,
    "sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
    "validation_results": results,
}, indent=2), encoding="utf-8")
print("\nsaved reports in", out_dir.resolve())