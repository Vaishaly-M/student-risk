import io
import json
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data")
CSV = DATA_DIR / "student-mat.csv"
URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
PASS_MARK = 10  # final grade G3 is out of 20

# True  = "after first assessment" scenario (uses G1)
# False = "before any assessment" scenario (background data only)
USE_G1 = True


def fetch_data():
    DATA_DIR.mkdir(exist_ok=True)
    if CSV.exists():
        return
    print("Downloading dataset...")
    raw = urllib.request.urlopen(URL).read()
    outer = zipfile.ZipFile(io.BytesIO(raw))
    for name in outer.namelist():
        if name.endswith("student-mat.csv"):
            CSV.write_bytes(outer.read(name))
            return
    for name in outer.namelist():
        if name.endswith(".zip"):
            inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
            for n in inner.namelist():
                if n.endswith("student-mat.csv"):
                    CSV.write_bytes(inner.read(n))
                    return
    raise SystemExit("Could not find student-mat.csv. Put it in data/ manually.")


def main():
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    fetch_data()
    df = pd.read_csv(CSV, sep=";")

    y = (df["G3"] >= PASS_MARK).astype(int)
    drop_cols = ["G2", "G3"] if USE_G1 else ["G1", "G2", "G3"]
    X = df.drop(columns=drop_cols)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, model in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        scores[name] = float(s.mean())
        print(f"{name}: CV ROC-AUC = {s.mean():.3f} (+/- {s.std():.3f})")

    best_name = max(scores, key=scores.get)
    print(f"\nBest model: {best_name}")

    best = Pipeline([("pre", pre), ("model", candidates[best_name])])
    best.fit(X_train, y_train)

    proba = best.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    test_auc = float(roc_auc_score(y_test, proba))
    print(f"\nTest ROC-AUC: {test_auc:.3f}")
    print(classification_report(y_test, preds, target_names=["Fail", "Pass"]))

    imp = permutation_importance(
        best, X_test, y_test, n_repeats=15, random_state=42, scoring="roc_auc"
    )
    imp_df = (
        pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    imp_df.to_csv("reports/feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    plt.xlabel("Permutation importance (drop in ROC-AUC)")
    plt.title("Top 10 factors influencing pass/fail prediction")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)

    with open("reports/metrics.json", "w") as f:
        json.dump({"scenario": "after first assessment (G1)" if USE_G1
                   else "background data only",
                   "cv_roc_auc": scores, "best_model": best_name,
                   "test_roc_auc": test_auc}, f, indent=2)

    bundle = {
        "pipeline": best,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_ranges": {c: (int(X[c].min()), int(X[c].max()), int(X[c].median()))
                       for c in num_cols},
        "cat_options": {c: sorted(X[c].unique().tolist()) for c in cat_cols},
    }
    joblib.dump(bundle, "models/model.joblib")
    print("\nSaved model to models/model.joblib")


if __name__ == "__main__":
    main()