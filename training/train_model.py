import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DATA_FILE = ROOT / "training" / "heart.csv"

# 7 simple features that a home device or patient could actually provide.
FEATURES = ["age", "sex", "trestbps", "chol", "thalach", "exang", "oldpeak"]

# Friendly names + units for the sliders on the web page.
LABELS = {
    "age": "Age (years)",
    "sex": "Sex (0 = female, 1 = male)",
    "trestbps": "Resting blood pressure (mmHg)",
    "chol": "Cholesterol (mg/dl)",
    "thalach": "Max heart rate (bpm)",
    "exang": "Chest pain during exercise (0 = no, 1 = yes)",
    "oldpeak": "ST depression (ECG measure)",
}


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            "Put heart.csv (with its header line) into the training folder."
        )
    # In this file, '?' marks a missing value (only in the ca and thal columns).
    df = pd.read_csv(DATA_FILE, na_values="?")
    needed = FEATURES + ["num"]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"heart.csv is missing columns: {missing_cols}. "
            "Check that the header line is the first line of the file."
        )
    return df


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_data()
    n_raw = len(df)
    df = df.dropna(subset=FEATURES + ["num"]).copy()
    df["target"] = (df["num"] > 0).astype(int)  # 1 = heart disease present

    X = df[FEATURES].astype(float).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(Xs_train, y_train)

    probs = clf.predict_proba(Xs_test)[:, 1]
    acc = accuracy_score(y_test, probs > 0.5)
    auc = roc_auc_score(y_test, probs)

    weights = clf.coef_[0]
    bias = float(clf.intercept_[0])

    # Range of the model's raw score (logit) on all data.
    # Used to pick the interval for the sigmoid approximation.
    all_logits = scaler.transform(X) @ weights + bias

    # Slider ranges and defaults for the web page
    ranges = {}
    for i, f in enumerate(FEATURES):
        col = X[:, i]
        ranges[f] = {
            "label": LABELS[f],
            "min": float(np.floor(col.min())),
            "max": float(np.ceil(col.max())),
            "default": float(np.round(np.median(col), 1)),
        }

    (MODEL_DIR / "weights.json").write_text(json.dumps({
        "features": FEATURES,
        "weights": [float(w) for w in weights],
        "bias": bias,
    }, indent=2))

    (MODEL_DIR / "scaler.json").write_text(json.dumps({
        "mean": [float(m) for m in scaler.mean_],
        "std": [float(s) for s in scaler.scale_],
        "ranges": ranges,
    }, indent=2))

    (MODEL_DIR / "metrics.json").write_text(json.dumps({
        "dataset": "UCI Heart Disease (Cleveland)",
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(float(acc), 4),
        "auc": round(float(auc), 4),
        "logit_min": float(all_logits.min()),
        "logit_max": float(all_logits.max()),
        "seed": SEED,
    }, indent=2))

    print(f"Rows in file: {n_raw}, used: {len(df)}  (train {len(X_train)}, test {len(X_test)})")
    print(f"Test accuracy: {acc:.3f}   AUC: {auc:.3f}")
    print(f"Logit range: [{all_logits.min():.2f}, {all_logits.max():.2f}]")
    print("Saved: model/weights.json, model/scaler.json, model/metrics.json")


if __name__ == "__main__":
    main()