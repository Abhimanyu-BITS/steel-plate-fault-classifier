"""
train_models.py
---------------
Step 2 of the pipeline: train and evaluate the five required classifiers on the
Steel Plates Faults dataset, then persist each fitted estimator so that the
Streamlit front-end can score new data *without ever retraining*.

Required models
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbour Classifier
    4. Naive Bayes Classifier (Gaussian)
    5. Support Vector Machine

Required metrics (all reported on the held-out test split)
    Accuracy, AUC, Precision, Recall, F1, Matthews Correlation Coefficient

Design notes
    * Every model is wrapped in a Pipeline, so any scaling is fitted on
      training folds only -- there is no leakage from the test split.
    * The target has 7 imbalanced classes, so Precision / Recall / F1 are
      macro-averaged (each fault weighs the same regardless of frequency) and
      AUC is the macro average of the one-vs-rest curves.
    * Hyper-parameters are selected by 5-fold stratified GridSearchCV
      optimising macro-F1, which is the honest criterion under class imbalance
      -- plain accuracy would simply reward predicting Other_Faults.

Run:  python train_models.py
Out:  model/*.joblib, model/metadata.json,
      test_data.csv, data/train_data.csv,
      reports/metrics_comparison.csv, reports/metrics_comparison.md,
      reports/classification_reports.txt, reports/confusion_matrices.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
TIDY_CSV = PROJECT_ROOT / "data" / "steel_faults_full.csv"
MODEL_DIR = PROJECT_ROOT / "model"
REPORT_DIR = PROJECT_ROOT / "reports"
TEST_CSV = PROJECT_ROOT / "test_data.csv"
TRAIN_CSV = PROJECT_ROOT / "data" / "train_data.csv"

TARGET_COLUMN = "Fault_Type"
RANDOM_SEED = 42
TEST_FRACTION = 0.25
CV_FOLDS = 5

# Order used in every table / dropdown across the whole project.
MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Support Vector Machine",
]


def build_registry() -> dict:
    """Registry of display name -> file stem, pipeline and search grid."""
    return {
        "Logistic Regression": {
            "stem": "logistic_regression",
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=4000,
                            solver="lbfgs",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            "grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "Decision Tree": {
            "stem": "decision_tree",
            # Trees are scale-invariant, so no scaler is attached.
            "pipeline": Pipeline(
                [("clf", DecisionTreeClassifier(random_state=RANDOM_SEED))]
            ),
            "grid": {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [None, 6, 10, 16, 24],
                "clf__min_samples_leaf": [1, 2, 5, 10],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "KNN": {
            "stem": "knn",
            "pipeline": Pipeline(
                [("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]
            ),
            "grid": {
                "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        },
        "Naive Bayes": {
            "stem": "naive_bayes",
            # Gaussian NB is chosen over Multinomial because the 27 descriptors
            # are continuous and several take negative values (for example
            # Luminosity_Index), which Multinomial NB cannot accept.
            "pipeline": Pipeline(
                [("scaler", StandardScaler()), ("clf", GaussianNB())]
            ),
            "grid": {"clf__var_smoothing": np.logspace(-11, -3, 9).tolist()},
        },
        "Support Vector Machine": {
            "stem": "svm",
            "pipeline": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", random_state=RANDOM_SEED)),
                ]
            ),
            "grid": {
                "clf__C": [1.0, 10.0, 100.0],
                "clf__gamma": ["scale", 0.01, 0.1],
                "clf__class_weight": [None, "balanced"],
            },
        },
    }


def build_splits(frame: pd.DataFrame):
    """Stratified train / test split, persisted to disk for reproducibility."""
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_FRACTION,
        stratify=target,
        random_state=RANDOM_SEED,
    )

    # The CSV handed to the Streamlit app must carry the features *and* the
    # target column, per the assignment brief.
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_CSV, index=False)
    pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_CSV, index=False)

    print(f"[train] train split : {X_train.shape[0]:>5} rows -> {TRAIN_CSV.name}")
    print(f"[train] test  split : {X_test.shape[0]:>5} rows -> {TEST_CSV.name}")
    return X_train, X_test, y_train, y_test


def score_predictions(y_true, y_pred, y_proba, class_labels) -> dict:
    """Compute the six assignment metrics for one model."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=class_labels
        ),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def fit_one(name: str, spec: dict, X_train, y_train):
    """Grid-search one model and return (fitted_estimator, best_params, cv_f1)."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = GridSearchCV(
        estimator=spec["pipeline"],
        param_grid=spec["grid"],
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    estimator = search.best_estimator_

    # SVC is grid-searched without Platt scaling (far cheaper); the winning
    # configuration is then refitted with probability=True so the app can
    # produce an AUC score.
    if name == "Support Vector Machine":
        tuned = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=RANDOM_SEED,
                        **tuned,
                    ),
                ),
            ]
        )
        estimator.fit(X_train, y_train)

    return estimator, search.best_params_, float(search.best_score_)


def plot_confusion_grid(matrices: dict, class_labels, destination: Path) -> None:
    """One row-normalised confusion matrix per model, saved as a single figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for ax, name in zip(axes, MODEL_ORDER):
        matrix = matrices[name].astype(float)
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalised = np.divide(
            matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0
        )
        ax.imshow(normalised, cmap="BuPu", vmin=0, vmax=1)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(class_labels, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                ax.text(
                    j,
                    i,
                    f"{normalised[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if normalised[i, j] > 0.55 else "#333333",
                )

    axes[-1].axis("off")
    fig.suptitle(
        "Row-normalised confusion matrices - Steel Plates Faults test split",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not TIDY_CSV.exists():
        print(f"[train] {TIDY_CSV} is missing -- run prepare_dataset.py first")
        return 1

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    registry = build_registry()
    frame = pd.read_csv(TIDY_CSV)
    print(f"[train] loaded {frame.shape[0]:,} rows x {frame.shape[1] - 1} features\n")

    X_train, X_test, y_train, y_test = build_splits(frame)
    class_labels = sorted(frame[TARGET_COLUMN].unique())
    feature_names = [c for c in frame.columns if c != TARGET_COLUMN]

    metric_rows: list[dict] = []
    tuning_notes: dict[str, dict] = {}
    confusion_grids: dict[str, np.ndarray] = {}
    text_reports: list[str] = []

    for name in MODEL_ORDER:
        spec = registry[name]
        print(f"\n[train] ---- {name} ----")
        started = time.perf_counter()

        estimator, best_params, cv_score = fit_one(name, spec, X_train, y_train)
        elapsed = time.perf_counter() - started

        y_pred = estimator.predict(X_test)
        y_proba = estimator.predict_proba(X_test)

        # Align probability columns with the canonical class order.
        proba_frame = pd.DataFrame(y_proba, columns=list(estimator.classes_))
        y_proba_aligned = proba_frame[class_labels].to_numpy()

        scores = score_predictions(y_test, y_pred, y_proba_aligned, class_labels)
        metric_rows.append({"ML Model Name": name, **scores})

        confusion_grids[name] = confusion_matrix(y_test, y_pred, labels=class_labels)
        tuning_notes[name] = {
            "best_params": {k: str(v) for k, v in best_params.items()},
            "cv_macro_f1": round(cv_score, 4),
            "fit_seconds": round(elapsed, 2),
        }

        print(f"[train] best params : {best_params}")
        print(f"[train] cv macro-F1 : {cv_score:.4f}   ({elapsed:.1f}s)")
        print(
            "[train] test        : "
            + "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
        )

        artefact = MODEL_DIR / (spec["stem"] + ".joblib")
        joblib.dump(estimator, artefact, compress=3)
        print(f"[train] saved       : model/{artefact.name}")

        text_reports.append(
            "=" * 78
            + f"\n{name}\n"
            + "=" * 78
            + f"\nbest params : {best_params}\ncv macro-F1 : {cv_score:.4f}\n\n"
            + classification_report(
                y_test, y_pred, labels=class_labels, zero_division=0, digits=4
            )
        )

    # ---------------------------------------------------------------- reports
    metrics = pd.DataFrame(metric_rows).set_index("ML Model Name")
    metrics_rounded = metrics.round(4)
    metrics_rounded.to_csv(REPORT_DIR / "metrics_comparison.csv")

    header = "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |"
    divider = "| --- | --- | --- | --- | --- | --- | --- |"
    body = [
        "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            name,
            row["Accuracy"],
            row["AUC"],
            row["Precision"],
            row["Recall"],
            row["F1"],
            row["MCC"],
        )
        for name, row in metrics.iterrows()
    ]
    (REPORT_DIR / "metrics_comparison.md").write_text(
        "# Model comparison - Steel Plates Faults (held-out test split)\n\n"
        + "\n".join([header, divider] + body)
        + "\n\nPrecision / Recall / F1 are macro-averaged; AUC is macro one-vs-rest.\n",
        encoding="utf-8",
    )

    (REPORT_DIR / "classification_reports.txt").write_text(
        "\n\n".join(text_reports), encoding="utf-8"
    )
    plot_confusion_grid(
        confusion_grids, class_labels, REPORT_DIR / "confusion_matrices.png"
    )

    metadata = {
        "dataset_name": "Steel Plates Faults (UCI ML Repository, id=198)",
        "source_url": "https://archive.ics.uci.edu/dataset/198/steel+plates+faults",
        "target_column": TARGET_COLUMN,
        "class_labels": class_labels,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_instances": int(len(frame)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_fraction": TEST_FRACTION,
        "random_seed": RANDOM_SEED,
        "cv_folds": CV_FOLDS,
        "selection_metric": "f1_macro",
        "averaging": {
            "precision_recall_f1": "macro",
            "auc": "macro one-vs-rest",
        },
        "model_files": {n: registry[n]["stem"] + ".joblib" for n in MODEL_ORDER},
        "model_order": MODEL_ORDER,
        "tuning": tuning_notes,
        "reference_metrics": {
            name: {k: round(float(v), 4) for k, v in row.items()}
            for name, row in metrics.iterrows()
        },
        "sklearn_version": sklearn.__version__,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n\n" + "=" * 78)
    print("FINAL COMPARISON TABLE (held-out test split)")
    print("=" * 78)
    print(metrics_rounded.to_string())
    print("\n[train] artefacts written to model/ and reports/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
