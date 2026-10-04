"""
Disease Prediction System - training script (Diabetes + Heart Disease)

Usage:  python train.py
Data :  data/diabetes.csv  (auto-downloaded if missing)
        data/heart.csv     (download manually from Kaggle: johnsmith88/heart-disease-dataset)
Output: models/*.joblib, outputs/*.csv, outputs/*.png
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import (DIABETES_FEATURES, DIABETES_TARGET, HEART_FEATURES, HEART_TARGET,
                    IQRClipper, clean_diabetes)

BASE = Path(__file__).parent
DATA_DIR, MODEL_DIR, OUT_DIR = BASE / "data", BASE / "models", BASE / "outputs"
for d in (DATA_DIR, MODEL_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

RS = 42
PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


# ---------- 2. Collect data ----------
def load_diabetes():
    path = DATA_DIR / "diabetes.csv"
    if not path.exists():
        print("diabetes.csv not found, downloading PIMA dataset...")
        df = pd.read_csv(PIMA_URL, header=None, names=DIABETES_FEATURES + [DIABETES_TARGET])
        df.to_csv(path, index=False)
    return pd.read_csv(path)


def load_heart():
    path = DATA_DIR / "heart.csv"
    if not path.exists():
        print("\n[!] data/heart.csv not found - skipping heart disease.")
        print("    Download from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
        print("    and place heart.csv inside the data/ folder, then run again.\n")
        return None
    df = pd.read_csv(path)
    before = len(df)
    df = df.drop_duplicates()  # this Kaggle file has many duplicate rows -> would leak into the test set
    print(f"Heart: removed {before - len(df)} duplicate rows ({before} -> {len(df)})")
    return df


# ---------- 3-4. Preprocess + feature selection live inside one pipeline (no data leakage) ----------
def build_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", IQRClipper()),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif)),
        ("clf", clf),
    ])


def model_zoo():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RS),
                                {"clf__C": [0.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [5, 9, 15]}),
        "SVM": (SVC(probability=True, random_state=RS),
                {"clf__C": [0.5, 1, 5], "clf__kernel": ["rbf", "linear"]}),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=RS),
                          {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=RS),
                              {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                               "clf__max_depth": [2, 3]}),
    }


def run(name, df, features, target):
    print(f"\n{'=' * 60}\n{name.upper()}  ({len(df)} rows)\n{'=' * 60}")
    X, y = df[features], df[target]
    print("Class balance:", y.value_counts(normalize=True).round(3).to_dict())

    # correlation heatmap (EDA)
    corr = df[features + [target]].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im)
    ax.set_title(f"{name} - correlation matrix")
    fig.savefig(OUT_DIR / f"{name}_correlation.png", bbox_inches="tight")
    plt.close(fig)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    n = len(features)
    ks = list(dict.fromkeys([max(2, n // 2), max(2, int(n * 0.75)), "all"]))

    rows, fitted, probas = [], {}, {}
    for mname, (clf, grid) in model_zoo().items():
        gs = GridSearchCV(build_pipeline(clf), {**grid, "selector__k": ks},
                          cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        est = gs.best_estimator_
        # ---------- 8. cross-validation with medical metrics ----------
        cvres = cross_validate(est, X_tr, y_tr, cv=cv,
                               scoring=["accuracy", "recall", "f1", "roc_auc"])
        pred = est.predict(X_te)
        proba = est.predict_proba(X_te)[:, 1]
        rows.append({
            "model": mname,
            "cv_accuracy": cvres["test_accuracy"].mean(),
            "cv_recall": cvres["test_recall"].mean(),
            "cv_f1": cvres["test_f1"].mean(),
            "cv_auc": cvres["test_roc_auc"].mean(),
            "test_accuracy": accuracy_score(y_te, pred),
            "test_precision": precision_score(y_te, pred, zero_division=0),
            "test_recall": recall_score(y_te, pred),
            "test_f1": f1_score(y_te, pred),
            "test_auc": roc_auc_score(y_te, proba),
            "best_params": str(gs.best_params_),
        })
        fitted[mname], probas[mname] = est, proba
        print(f"  {mname:<20} CV AUC {rows[-1]['cv_auc']:.3f} | test acc {rows[-1]['test_accuracy']:.3f} "
              f"| recall {rows[-1]['test_recall']:.3f} | AUC {rows[-1]['test_auc']:.3f}")

    res = pd.DataFrame(rows).round(4)
    res.to_csv(OUT_DIR / f"{name}_results.csv", index=False)

    # best model chosen on cross-validation AUC only (test set stays untouched for selection)
    best_name = res.loc[res.cv_auc.idxmax(), "model"]
    best = fitted[best_name]
    sel = best.named_steps["selector"].get_support()
    used = [f for f, keep in zip(features, sel) if keep]
    print(f"\nBest model: {best_name}\nFeatures used: {used}")

    ConfusionMatrixDisplay.from_predictions(y_te, best.predict(X_te), cmap="Blues")
    plt.title(f"{name} - {best_name}")
    plt.savefig(OUT_DIR / f"{name}_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for mname, p in probas.items():
        RocCurveDisplay.from_predictions(y_te, p, ax=ax, name=mname)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"{name} - ROC curves")
    fig.savefig(OUT_DIR / f"{name}_roc.png", bbox_inches="tight")
    plt.close(fig)

    metrics = res[res.model == best_name].iloc[0].drop("best_params").to_dict()
    joblib.dump({"pipeline": best, "features": features, "model_name": best_name,
                 "features_used": used, "metrics": metrics}, MODEL_DIR / f"{name}.joblib")


if __name__ == "__main__":
    diabetes = clean_diabetes(load_diabetes())
    run("diabetes", diabetes, DIABETES_FEATURES, DIABETES_TARGET)

    heart = load_heart()
    if heart is not None:
        run("heart", heart, HEART_FEATURES, HEART_TARGET)
    print("\nDone. Now run:  streamlit run app.py")