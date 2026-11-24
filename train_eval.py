"""CP467 Group 14 - KNN vs SVM on the saved food features (Person A).

Loads the .npy features made by the preprocessing notebook, tunes KNN and SVM
with the same 5-fold stratified CV, tests each once on the test set, and saves
the numbers + confusion matrices to a results folder next to this script.
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# change this if your features folder is somewhere else
FEATURES_DIR = Path(r"C:\Users\garyc\Desktop\New folder\food-identification-cp467\features")

# True = tune on a 3,000 image sample
# False = tune on all 15,000 training images
QUICK_MODE = True
QUICK_TUNE_SIZE = 3000

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "results"


def load_features():
    """Task 1: Load the saved feature matrices, labels and class names."""
    if not (FEATURES_DIR / "X_train.npy").exists():
        raise FileNotFoundError(f"Can't find X_train.npy in {FEATURES_DIR}. Fix FEATURES_DIR at the top.")

    X_train = np.load(FEATURES_DIR / "X_train.npy")
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")
    category_map = joblib.load(FEATURES_DIR / "category_map.pkl")
    names = [category_map[i] for i in range(len(category_map))]

    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    return X_train, X_test, y_train, y_test, names


def make_search_spaces():
    """Task 2: Build the pipelines and hyperparameter grids from the proposal."""
    knn = Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())])
    knn_grid = {
        "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
        "clf__metric": ["euclidean", "manhattan"],
    }

    svm = Pipeline([("scale", StandardScaler()), ("clf", SVC())])
    svm_grid = [
        {
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": [0.001, 0.01, 0.1],
        },
        {
            "clf__kernel": ["linear"],
            "clf__C": [0.1, 1, 10, 100],
        },
    ]
    return {"KNN": (knn, knn_grid), "SVM": (svm, svm_grid)}


def tune(name, pipe, grid, X, y, cv):
    """Task 3: Grid search with 5-fold stratified CV, scored by macro F1."""
    search = GridSearchCV(
        pipe, grid, scoring="f1_macro", cv=cv,
        n_jobs=-1, refit=False, verbose=1,
    )
    start = time.time()
    search.fit(X, y)
    print(f"{name} tuning took {time.time() - start:.1f}s")
    print(f"{name} best params: {search.best_params_}")
    print(f"{name} best CV macro F1: {search.best_score_:.4f}")
    return search


def train_and_test(name, pipe, best_params, X_train, y_train, X_test, y_test):
    """Task 4: Retrain the best setting on all training data, test once."""
    model = clone(pipe).set_params(**best_params)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    preds = model.predict(X_test)
    predict_time = time.time() - t0

    results = {
        "best_params": best_params,
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "train_time_s": train_time,
        "predict_time_total_s": predict_time,
        "predict_time_per_image_ms": predict_time / len(y_test) * 1000,
    }
    print(f"{name} test accuracy: {results['accuracy']:.4f}  macro F1: {results['macro_f1']:.4f}")
    print(f"{name} train {train_time:.1f}s, predict {results['predict_time_per_image_ms']:.3f} ms/image")
    return results, preds


def save_outputs(name, results, preds, y_test, names):
    """Task 5: Save the per-class report and the confusion matrix plot."""
    report = classification_report(y_test, preds, target_names=names, digits=3)
    print(f"\n{name} per-class report:\n{report}")
    (OUT_DIR / f"{name}_report.txt").write_text(report)

    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(11, 11))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(
        ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{name} confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    worst = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:5], off.shape))[0]
    results["top_confusions"] = [
        f"{names[t]} -> {names[p]}: {off[t, p]}" for t, p in worst
    ]
    print(f"{name} top confusions (true -> predicted):", results["top_confusions"])


def main():
    """Task 6: Run everything for KNN and SVM and save a summary."""
    OUT_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test, names = load_features()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    if QUICK_MODE:
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=QUICK_TUNE_SIZE,
            stratify=y_train, random_state=SEED,
        )
        print(f"QUICK_MODE: tuning on {len(y_tune)} images")
    else:
        X_tune, y_tune = X_train, y_train

    summary = {"quick_mode": QUICK_MODE, "tune_size": int(len(y_tune)),
               "n_features": int(X_train.shape[1])}

    for name, (pipe, grid) in make_search_spaces().items():
        print(f"\n===== {name} =====")
        search = tune(name, pipe, grid, X_tune, y_tune, cv)
        results, preds = train_and_test(
            name, pipe, search.best_params_, X_train, y_train, X_test, y_test
        )
        results["best_cv_macro_f1"] = search.best_score_
        save_outputs(name, results, preds, y_test, names)
        summary[name] = results

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone. Results saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
