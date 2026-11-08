"""
app.py
------
Steel Plate Surface-Fault Inspector -- Streamlit front-end.

The app loads the five classifiers that were fitted offline by
`train_models.py` and scores an uploaded test CSV with them. It never calls
`.fit()`: the uploaded file is treated purely as held-out evaluation data, as
the assignment requires.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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

APP_ROOT = Path(__file__).resolve().parent
MODEL_DIR = APP_ROOT / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"
BUNDLED_TEST_CSV = APP_ROOT / "test_data.csv"

COMPARE_ALL = "Compare all five models"

# A muted, colour-blind-safe ramp used consistently for every chart.
FAULT_PALETTE = ["#4C6EF5", "#15AABF", "#2F9E44", "#F08C00", "#E8590C", "#C2255C", "#7048E8"]
HEAT_CMAP = "BuPu"


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner=False)
def load_metadata() -> dict:
    """Training-time facts: feature order, class labels, tuned parameters."""
    with METADATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def load_estimators(file_map: dict) -> dict:
    """Deserialise every saved pipeline exactly once per server process."""
    estimators = {}
    for display_name, filename in file_map.items():
        estimators[display_name] = joblib.load(MODEL_DIR / filename)
    return estimators


@st.cache_data(show_spinner=False)
def read_csv_bytes(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(payload))


@st.cache_data(show_spinner=False)
def read_bundled_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


# ------------------------------------------------------------------ scoring
def present_labels(y_true) -> list:
    """Classes that actually occur in the file being scored."""
    return sorted(pd.unique(np.asarray(y_true)))


def macro_ovr_auc(y_true, proba_frame: pd.DataFrame) -> float | None:
    """
    Macro one-vs-rest AUC, accumulated one class at a time.

    Each class contributes the binary AUC of "is this class" against that
    class's own probability column. On a complete file this is numerically
    identical to `roc_auc_score(..., multi_class="ovr", average="macro")`,
    but it also survives a partial upload: a class that is absent simply
    contributes no term, and no row-wise renormalisation is required.
    (Renormalising would be wrong anyway -- row sums vary, so rescaling
    reorders samples within a class and changes its AUC.)

    Returns None when fewer than two classes are present, since one-vs-rest
    is undefined there.
    """
    labels = present_labels(y_true)
    if len(labels) < 2:
        return None

    truth = np.asarray(y_true)
    per_class: list[float] = []
    for label in labels:
        indicator = (truth == label).astype(int)
        # Needs both a positive and a negative to define an ROC curve.
        if indicator.min() == indicator.max():
            continue
        per_class.append(float(roc_auc_score(indicator, proba_frame[label].to_numpy())))

    return float(np.mean(per_class)) if per_class else None


def evaluate(estimator, X, y_true, class_labels) -> dict:
    """Run one fitted pipeline over the test frame and score the six metrics."""
    y_pred = estimator.predict(X)

    proba = estimator.predict_proba(X)
    proba_frame = pd.DataFrame(proba, columns=list(estimator.classes_))
    # Guarantee a column per known class, in the canonical order.
    for label in class_labels:
        if label not in proba_frame.columns:
            proba_frame[label] = 0.0
    proba_frame = proba_frame[class_labels]

    # Macro-average over the classes present in this file. On the full test
    # split that is all seven, so the figures reproduce the README table; on a
    # partial upload it keeps the average from being dragged down by classes
    # the file never contained.
    scored_labels = present_labels(y_true)

    return {
        "y_pred": y_pred,
        "proba": proba_frame,
        "metrics": {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": macro_ovr_auc(y_true, proba_frame),
            "Precision": precision_score(
                y_true, y_pred, labels=scored_labels, average="macro", zero_division=0
            ),
            "Recall": recall_score(
                y_true, y_pred, labels=scored_labels, average="macro", zero_division=0
            ),
            "F1": f1_score(
                y_true, y_pred, labels=scored_labels, average="macro", zero_division=0
            ),
            "MCC": matthews_corrcoef(y_true, y_pred),
        },
    }


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


# -------------------------------------------------------------------- charts
def draw_confusion(matrix, class_labels, title: str, normalise: bool):
    frame = pd.DataFrame(matrix, index=class_labels, columns=class_labels)
    if normalise:
        totals = frame.sum(axis=1).replace(0, np.nan)
        frame = frame.div(totals, axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sns.heatmap(
        frame,
        annot=True,
        fmt=".2f" if normalise else "d",
        cmap=HEAT_CMAP,
        vmin=0,
        vmax=1 if normalise else None,
        cbar_kws={"label": "row share" if normalise else "samples"},
        linewidths=0.4,
        linecolor="white",
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_xlabel("Predicted fault", fontsize=10)
    ax.set_ylabel("Actual fault", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    plt.setp(ax.get_xticklabels(), rotation=38, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    return fig


def draw_metric_bars(table: pd.DataFrame, metric: str):
    ordered = table[metric].dropna().sort_values()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bars = ax.barh(
        ordered.index,
        ordered.to_numpy(),
        color=FAULT_PALETTE[: len(ordered)],
        height=0.6,
    )
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel(metric, fontsize=10)
    ax.set_title(f"{metric} by model", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def draw_class_support(y_true, class_labels):
    counts = pd.Series(y_true).value_counts().reindex(class_labels).fillna(0)
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    bars = ax.bar(counts.index, counts.to_numpy(), color=FAULT_PALETTE, width=0.62)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_ylabel("rows in file", fontsize=9)
    ax.set_title("Class balance of the evaluated file", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=38, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ page
st.set_page_config(
    page_title="Steel Plate Surface-Fault Inspector",
    page_icon="🪓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.1rem; padding-bottom: 3rem; }
      .sp-hero {
          background: linear-gradient(118deg, #16233d 0%, #2b3f66 58%, #41506e 100%);
          border-radius: 16px; padding: 1.5rem 1.8rem; margin-bottom: 1.3rem;
          color: #f4f6fb;
      }
      .sp-hero h1 { margin: 0 0 .35rem 0; font-size: 1.72rem; letter-spacing: -.4px; }
      .sp-hero p  { margin: 0; opacity: .88; font-size: .95rem; line-height: 1.45; }
      .sp-chip {
          display: inline-block; margin: .55rem .4rem 0 0; padding: .17rem .62rem;
          border-radius: 999px; background: rgba(255,255,255,.14);
          font-size: .74rem; letter-spacing: .3px;
      }
      div[data-testid="stMetricValue"] { font-size: 1.42rem; }
      div[data-testid="stMetricLabel"] { font-size: .78rem; opacity: .75; }
      .sp-note {
          border-left: 3px solid #4C6EF5; background: #f3f5fd;
          padding: .6rem .9rem; border-radius: 6px; font-size: .86rem;
          color: #22304d;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    meta = load_metadata()
    models = load_estimators(meta["model_files"])
except FileNotFoundError as exc:
    st.error(
        "Model artefacts are missing. Run `python prepare_dataset.py` and then "
        f"`python train_models.py` to regenerate them.\n\nDetail: {exc}"
    )
    st.stop()

FEATURES: list[str] = meta["feature_names"]
TARGET: str = meta["target_column"]
CLASSES: list[str] = meta["class_labels"]

st.markdown(
    f"""
    <div class="sp-hero">
      <h1>Steel Plate Surface-Fault Inspector</h1>
      <p>Five pre-trained classifiers scoring surface-defect readings from a steel
         rolling line. Upload a test CSV and the app evaluates it with the saved
         pipelines &mdash; no model is ever refitted in the browser.</p>
      <span class="sp-chip">UCI id 198</span>
      <span class="sp-chip">{meta['n_instances']:,} instances</span>
      <span class="sp-chip">{meta['n_features']} features</span>
      <span class="sp-chip">{len(CLASSES)} fault classes</span>
      <span class="sp-chip">scikit-learn {meta['sklearn_version']}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV",
        type=["csv"],
        help=(
            "The file must contain the 27 feature columns and the "
            f"`{TARGET}` target column."
        ),
    )
    use_bundled = st.checkbox(
        "Use the bundled test_data.csv instead",
        value=uploaded is None,
        help="The 486-row stratified hold-out split created during training.",
    )

    st.divider()
    st.header("2 · Model")
    choice = st.selectbox(
        "Classifier to evaluate",
        options=meta["model_order"] + [COMPARE_ALL],
        index=len(meta["model_order"]),
        help="Pick one classifier, or compare all five side by side.",
    )

    st.divider()
    st.header("3 · Display")
    normalise_cm = st.toggle(
        "Normalise confusion matrix by row", value=True,
        help="Shows per-class recall on the diagonal."
    )
    show_head = st.toggle("Preview the loaded data", value=False)

    st.divider()
    with st.expander("Tuned hyper-parameters"):
        for name in meta["model_order"]:
            note = meta["tuning"][name]
            st.caption(f"**{name}** — CV macro-F1 {note['cv_macro_f1']:.4f}")
            st.code(
                "\n".join(f"{k.replace('clf__', '')} = {v}"
                          for k, v in note["best_params"].items())
                or "defaults",
                language="text",
            )

    st.divider()
    if BUNDLED_TEST_CSV.exists():
        st.download_button(
            "Download test_data.csv",
            data=BUNDLED_TEST_CSV.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )
    st.caption(
        f"Artefacts generated {meta['generated_at']} · seed {meta['random_seed']} · "
        f"{meta['cv_folds']}-fold CV on macro-F1"
    )

# ------------------------------------------------------------- data loading
source_label = None
frame = None

if uploaded is not None and not use_bundled:
    try:
        frame = read_csv_bytes(uploaded.getvalue())
        source_label = f"uploaded file `{uploaded.name}`"
    except Exception as exc:
        st.error(f"That CSV could not be parsed: {exc}")
        st.stop()
elif use_bundled:
    if not BUNDLED_TEST_CSV.exists():
        st.error("test_data.csv is missing from the repository.")
        st.stop()
    frame = read_bundled_csv(str(BUNDLED_TEST_CSV))
    source_label = "bundled `test_data.csv`"
else:
    st.info(
        "Upload a test CSV in the sidebar, or tick **Use the bundled "
        "test_data.csv instead** to score the stratified hold-out split."
    )
    st.stop()

# --------------------------------------------------------------- validation
missing_features = [c for c in FEATURES if c not in frame.columns]
if missing_features:
    st.error(
        f"The file is missing {len(missing_features)} required feature column(s): "
        + ", ".join(f"`{c}`" for c in missing_features[:8])
        + (" …" if len(missing_features) > 8 else "")
    )
    st.stop()

if TARGET not in frame.columns:
    st.error(
        f"The file has no `{TARGET}` column. The assignment requires the test CSV "
        "to carry the target so that evaluation metrics can be computed."
    )
    st.stop()

working = frame.dropna(subset=FEATURES + [TARGET])
dropped = len(frame) - len(working)
if working.empty:
    st.error("No complete rows remain after dropping rows with missing values.")
    st.stop()

unknown = sorted(set(working[TARGET].astype(str)) - set(CLASSES))
if unknown:
    st.warning(
        "Ignoring "
        + ", ".join(f"`{u}`" for u in unknown)
        + " — these labels were not seen during training."
    )
    working = working[working[TARGET].astype(str).isin(CLASSES)]
    if working.empty:
        st.error("No rows left with a recognised fault label.")
        st.stop()

X_eval = working[FEATURES]
y_eval = working[TARGET].astype(str)

caption = (
    f"Scoring **{len(X_eval):,} rows** from {source_label} · "
    f"{len(FEATURES)} features · {y_eval.nunique()} of {len(CLASSES)} classes present"
)
if dropped:
    caption += f" · {dropped} incomplete row(s) dropped"
st.markdown(caption)

if show_head:
    with st.expander("Loaded data preview", expanded=True):
        st.dataframe(working.head(25), width="stretch", height=300)

st.markdown(
    '<div class="sp-note">The saved pipelines are only used for inference. '
    "Scaling parameters, tree splits, neighbours and support vectors all come "
    "from the training split — nothing is re-estimated from this file.</div>",
    unsafe_allow_html=True,
)
st.write("")

# ------------------------------------------------------------------ compute
with st.spinner("Scoring with the saved pipelines…"):
    results = {
        name: evaluate(estimator, X_eval, y_eval, CLASSES)
        for name, estimator in models.items()
    }

metric_table = pd.DataFrame(
    {name: res["metrics"] for name, res in results.items()}
).T.reindex(meta["model_order"])

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# ------------------------------------------------------------------- output
if choice == COMPARE_ALL:
    st.subheader("Comparison across all five classifiers")

    st.dataframe(
        metric_table.style.format("{:.4f}", na_rep="n/a")
        .background_gradient(cmap=HEAT_CMAP, axis=0, vmin=0, vmax=1)
        .set_properties(**{"font-size": "13px"}),
        width="stretch",
    )

    leaders = ", ".join(
        f"**{m}** → {metric_table[m].idxmax()}"
        for m in METRIC_COLUMNS
        if metric_table[m].notna().any()
    )
    st.caption("Best per metric: " + leaders)

    tab_bars, tab_cms, tab_support, tab_export = st.tabs(
        ["Metric charts", "Confusion matrices", "Class balance", "Export"]
    )

    with tab_bars:
        picked = st.selectbox(
            "Metric to plot", METRIC_COLUMNS, index=0, key="cmp_metric"
        )
        if metric_table[picked].notna().any():
            st.pyplot(draw_metric_bars(metric_table, picked), width="stretch")
        else:
            st.info(f"{picked} could not be computed for this file.")

    with tab_cms:
        cm_choice = st.radio(
            "Classifier",
            meta["model_order"],
            horizontal=True,
            key="cmp_cm_model",
        )
        matrix = confusion_matrix(y_eval, results[cm_choice]["y_pred"], labels=CLASSES)
        st.pyplot(
            draw_confusion(
                matrix,
                CLASSES,
                f"{cm_choice} — confusion matrix",
                normalise_cm,
            ),
            width="stretch",
        )

    with tab_support:
        st.pyplot(draw_class_support(y_eval, CLASSES), width="stretch")

    with tab_export:
        export = metric_table.copy()
        export.index.name = "ML Model Name"
        st.download_button(
            "Download the comparison table (CSV)",
            data=export.round(4).to_csv().encode("utf-8"),
            file_name="model_comparison_metrics.csv",
            mime="text/csv",
        )
        predictions = working[[TARGET]].rename(columns={TARGET: "actual"}).copy()
        for name in meta["model_order"]:
            predictions[f"pred__{name}"] = results[name]["y_pred"]
        st.download_button(
            "Download every model's predictions (CSV)",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="all_model_predictions.csv",
            mime="text/csv",
        )
        st.dataframe(predictions.head(20), width="stretch")

else:
    chosen = results[choice]
    scores = chosen["metrics"]

    st.subheader(choice)
    tuned = meta["tuning"][choice]
    st.caption(
        "Tuned configuration: "
        + ", ".join(
            f"`{k.replace('clf__', '')}={v}`" for k, v in tuned["best_params"].items()
        )
        + f" · CV macro-F1 {tuned['cv_macro_f1']:.4f}"
    )

    columns = st.columns(6)
    reference = meta["reference_metrics"].get(choice, {})
    for column, metric in zip(columns, METRIC_COLUMNS):
        value = scores[metric]
        delta = None
        if value is not None and metric in reference:
            gap = value - reference[metric]
            delta = f"{gap:+.4f} vs report" if abs(gap) > 5e-5 else "matches report"
        column.metric(metric, fmt(value), delta=delta, delta_color="off")

    if scores["AUC"] is None:
        st.info(
            "AUC needs at least two fault classes present in the uploaded file."
        )

    tab_cm, tab_report, tab_perclass, tab_pred = st.tabs(
        ["Confusion matrix", "Classification report", "Per-class scores", "Predictions"]
    )

    with tab_cm:
        matrix = confusion_matrix(y_eval, chosen["y_pred"], labels=CLASSES)
        st.pyplot(
            draw_confusion(
                matrix, CLASSES, f"{choice} — confusion matrix", normalise_cm
            ),
            width="stretch",
        )
        with st.expander("Raw counts"):
            st.dataframe(
                pd.DataFrame(matrix, index=CLASSES, columns=CLASSES),
                width="stretch",
            )

    with tab_report:
        st.text(
            classification_report(
                y_eval,
                chosen["y_pred"],
                labels=CLASSES,
                zero_division=0,
                digits=4,
            )
        )

    with tab_perclass:
        per_class = pd.DataFrame(
            classification_report(
                y_eval,
                chosen["y_pred"],
                labels=CLASSES,
                zero_division=0,
                output_dict=True,
            )
        ).T.loc[CLASSES, ["precision", "recall", "f1-score", "support"]]
        st.dataframe(
            per_class.style.format(
                {"precision": "{:.4f}", "recall": "{:.4f}",
                 "f1-score": "{:.4f}", "support": "{:.0f}"}
            ).background_gradient(
                cmap=HEAT_CMAP, subset=["precision", "recall", "f1-score"],
                vmin=0, vmax=1,
            ),
            width="stretch",
        )
        weakest = per_class["f1-score"].idxmin()
        st.caption(
            f"Weakest class for this model: **{weakest}** "
            f"(F1 {per_class.loc[weakest, 'f1-score']:.4f}, "
            f"{int(per_class.loc[weakest, 'support'])} rows)."
        )

    with tab_pred:
        detail = pd.DataFrame(
            {
                "actual": y_eval.to_numpy(),
                "predicted": chosen["y_pred"],
            },
            index=working.index,
        )
        detail["correct"] = detail["actual"] == detail["predicted"]
        detail["confidence"] = chosen["proba"].to_numpy().max(axis=1)

        only_errors = st.checkbox("Show misclassified rows only", value=False)
        view = detail[~detail["correct"]] if only_errors else detail
        st.dataframe(
            view.style.format({"confidence": "{:.3f}"}),
            width="stretch",
            height=380,
        )
        st.download_button(
            "Download these predictions (CSV)",
            data=detail.to_csv(index_label="row").encode("utf-8"),
            file_name=f"predictions_{choice.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "M.Tech (AIML) · Machine Learning Assignment 2 · "
    "Dataset: Steel Plates Faults, UCI ML Repository (id 198). "
    "Precision / Recall / F1 are macro-averaged; AUC is macro one-vs-rest."
)
