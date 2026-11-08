# Steel Plate Surface-Fault Inspector

**M.Tech (AIML) · Machine Learning · Assignment 2**

Five classification models trained on the UCI *Steel Plates Faults* dataset, served
through an interactive Streamlit application that scores an uploaded test CSV with
the saved pipelines — without retraining.

---

## a. Problem statement

A steel rolling mill inspects every finished plate with an automated surface-vision
rig. The rig segments each suspected defect and reduces it to **27 numeric
descriptors** — bounding-box geometry, pixel area and perimeter, luminosity
statistics, conveyer geometry, steel grade and a set of derived shape indices.
A human metallurgist then labels the defect with one of **seven fault categories**.

Manual labelling is the bottleneck: it is slow, needs a trained inspector on every
shift, and is inconsistent between inspectors. The task addressed here is therefore:

> **Given the 27 descriptors produced automatically for a surface defect, predict
> which of the seven fault types it is.**

This is a **multi-class (7-class) classification** problem. It is a genuinely hard
one, for two reasons that shape every result in this report:

1. **The classes are heavily imbalanced** — from 55 `Dirtiness` samples up to 673
   `Other_Faults` samples, a 12:1 ratio.
2. **`Other_Faults` is a catch-all bucket**, not a coherent defect type. It holds
   34.7% of the data and overlaps every other class in feature space, which puts a
   hard ceiling on achievable accuracy.

Because the classes are imbalanced, plain accuracy is a misleading headline: a model
that simply predicted `Other_Faults` for everything would score 34.7% while being
useless. Every averaged metric in this report is therefore **macro-averaged**, so
that a rare fault such as `Dirtiness` counts exactly as much as the common bucket.

**Business framing of the errors.** The two error directions cost differently. A
missed `K_Scatch` or `Stains` ships a defective plate to a customer. A false alarm
on a good plate costs a re-inspection. Macro-recall tracks the first; macro-precision
tracks the second — both are reported, rather than collapsing to accuracy alone.

---

## b. Dataset description

| Property | Value |
| --- | --- |
| Name | Steel Plates Faults |
| Source | UCI Machine Learning Repository, dataset id **198** |
| URL | https://archive.ics.uci.edu/dataset/198/steel+plates+faults |
| Donor | Semeion, Research Center of Sciences of Communication, Rome, Italy |
| **Instances** | **1,941** (assignment minimum 500 ✔) |
| **Features** | **27**, all numeric (assignment minimum 12 ✔) |
| Target column | `Fault_Type` |
| Classes | 7 (multi-class) |
| Missing values | None |
| Train / test split | 1,455 / 486, stratified, `random_state=42` |

### Target distribution

| Fault type | Instances | Share | What it is |
| --- | ---: | ---: | --- |
| `Other_Faults` | 673 | 34.67% | Catch-all bucket for anything not in the six named classes |
| `Bumps` | 402 | 20.71% | Raised deformations on the plate surface |
| `K_Scatch` | 391 | 20.14% | Large scratch-type scoring, visually distinctive |
| `Z_Scratch` | 190 | 9.79% | Z-shaped scratch marks |
| `Pastry` | 158 | 8.14% | Pastry-like surface texture defect |
| `Stains` | 72 | 3.71% | Discolouration / staining |
| `Dirtiness` | 55 | 2.83% | Foreign particles embedded in the surface |

### Feature groups (27 total)

| Group | Columns |
| --- | --- |
| Defect bounding box (4) | `X_Minimum`, `X_Maximum`, `Y_Minimum`, `Y_Maximum` |
| Size and outline (3) | `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter` |
| Luminosity (4) | `Sum_of_Luminosity`, `Minimum_of_Luminosity`, `Maximum_of_Luminosity`, `Luminosity_Index` |
| Plate / line geometry (3) | `Length_of_Conveyer`, `Steel_Plate_Thickness`, `Orientation_Index` |
| Steel grade (2, binary) | `TypeOfSteel_A300`, `TypeOfSteel_A400` |
| Derived shape indices (7) | `Edges_Index`, `Empty_Index`, `Square_Index`, `Outside_X_Index`, `Edges_X_Index`, `Edges_Y_Index`, `Outside_Global_Index` |
| Log / sigmoid transforms (4) | `LogOfAreas`, `Log_X_Index`, `Log_Y_Index`, `SigmoidOfAreas` |

### Preprocessing

The raw UCI archive ships the label as a **one-hot block of 7 indicator columns**
(`Faults.NNA`, 34 tab-separated columns, no header; names in `Faults27x7_var`).
`prepare_dataset.py` collapses that block into the single categorical `Fault_Type`
column and asserts that every row is single-labelled.

Feature scaling matters a great deal here, because the raw columns span
wildly different magnitudes — `Y_Minimum` reaches ~1.3 × 10⁷ while
`Luminosity_Index` sits in [−1, 1]. Any distance- or gradient-based learner would
otherwise be dominated by the coordinate columns alone. `StandardScaler` is
therefore attached to Logistic Regression, KNN, Naive Bayes and the SVM. The
Decision Tree is deliberately left unscaled, since axis-aligned splits are
scale-invariant and scaling it would only obscure the saved model.

**Every scaler lives inside a `Pipeline`**, so it is re-fitted on the training folds
of each cross-validation split and never sees the test data. There is no leakage.

---

## c. GitHub repository link

> **https://github.com/Abhimanyu-BITS/steel-plate-fault-classifier**

**Live Streamlit app:** `<PASTE-YOUR-STREAMLIT-APP-URL-HERE>`

### Repository contents

```
steel-plate-fault-classifier/
├── app.py                        # Streamlit front-end (inference only)
├── requirements.txt              # pinned dependencies
├── README.md                     # this file
├── test_data.csv                 # 486-row stratified hold-out split
├── prepare_dataset.py            # downloads UCI data, builds the tidy CSV
├── train_models.py               # trains, tunes, evaluates and saves all 5 models
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metadata.json             # feature order, class labels, tuned params, metrics
├── data/
│   ├── raw/                      # untouched UCI files, kept for provenance
│   ├── steel_faults_full.csv     # all 1,941 rows after one-hot collapse
│   ├── train_data.csv            # 1,455-row training split
│   └── dataset_summary.json
└── reports/
    ├── metrics_comparison.csv    # the comparison table below, as data
    ├── metrics_comparison.md
    ├── classification_reports.txt  # per-class precision/recall/F1 for all 5 models
    └── confusion_matrices.png
```

### Reproducing the results

```bash
pip install -r requirements.txt
python prepare_dataset.py     # fetch + reshape the UCI data
python train_models.py        # tune, evaluate, save model/*.joblib
streamlit run app.py          # launch the UI at http://localhost:8501
```

With `random_state=42` fixed throughout, `train_models.py` reproduces the table
below exactly.

---

## d. Models used

All five classifiers are trained on the **same** 1,455-row training split and scored
on the **same** 486-row hold-out split, which is the file shipped as `test_data.csv`.

Hyper-parameters were chosen by **5-fold stratified `GridSearchCV` optimising
macro-F1**. Macro-F1 — not accuracy — is the selection criterion precisely because
of the class imbalance described above; tuning on accuracy would have rewarded models
for abandoning the rare faults. Notably, the grid was free to pick
`class_weight="balanced"` for Logistic Regression, the Decision Tree and the SVM, and
in all three cases cross-validation preferred the **unweighted** fit.

| Model | Tuned configuration | CV macro-F1 |
| --- | --- | ---: |
| Logistic Regression | `C=1.0`, `class_weight=None`, `lbfgs`, `max_iter=4000` | 0.7145 |
| Decision Tree | `criterion=entropy`, `max_depth=None`, `min_samples_leaf=5` | 0.7495 |
| KNN | `n_neighbors=7`, `weights=distance`, `p=1` (Manhattan) | 0.7650 |
| Naive Bayes | Gaussian, `var_smoothing=1e-3` | 0.6269 |
| Support Vector Machine | RBF kernel, `C=10.0`, `gamma=0.01` | 0.7798 |

**Why Gaussian Naive Bayes rather than Multinomial?** The 27 descriptors are
continuous, and several — `Luminosity_Index`, `Orientation_Index` — legitimately take
negative values. Multinomial NB models non-negative counts and cannot accept that
input, so Gaussian NB is the only correct member of the family here.

### Comparison table — evaluation metrics on the held-out test split

Precision, Recall and F1 are **macro-averaged**; AUC is the **macro one-vs-rest** AUC
across the seven classes. Best value in each column is shown in bold.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7222 | 0.9386 | 0.7632 | 0.7113 | 0.7316 | 0.6404 |
| Decision Tree | 0.7449 | 0.8819 | 0.7585 | 0.7157 | 0.7349 | 0.6697 |
| KNN | **0.7510** | 0.9359 | 0.7443 | **0.7717** | 0.7542 | **0.6827** |
| Naive Bayes | 0.6235 | 0.9250 | 0.6480 | 0.7101 | 0.6424 | 0.5611 |
| Support Vector Machine | 0.7428 | **0.9511** | **0.8018** | 0.7378 | **0.7654** | 0.6672 |

Per-class precision/recall/F1 for every model is in
[`reports/classification_reports.txt`](reports/classification_reports.txt), and the
five confusion matrices are rendered in
[`reports/confusion_matrices.png`](reports/confusion_matrices.png).

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | A respectable linear baseline — 0.7222 accuracy and a strong 0.9386 AUC — which confirms the classes are *largely* linearly separable once features are standardised. Its weakness is exactly where a linear boundary should fail: `Bumps` recall collapses to 0.5347 and `Pastry` to 0.4615, because both are textural faults that sit *inside* the `Other_Faults` cloud rather than on one side of a hyperplane. The gap between its high AUC and mediocre accuracy is informative: the model *ranks* classes well but places its hard decision boundaries poorly. It needed `max_iter=4000` to converge, and cross-validation chose `C=1.0`, i.e. moderate regularisation. |
| **Decision Tree** | Second-best accuracy (0.7449) but by far the **worst AUC (0.8819)** — 0.043 below even the next-worst model. This is the single most instructive result in the table: a single tree assigns every sample in a leaf the same probability, so its score distribution is coarse and piecewise-constant, which is poor for *ranking* even when the arg-max prediction is right. Accuracy and AUC are measuring genuinely different things here. The tree is the best model for `Z_Scratch` precision (0.9762), a fault defined by sharp geometric thresholds that axis-aligned splits capture naturally. `min_samples_leaf=5` was needed to stop it memorising the training set. |
| **KNN** | The strongest model on three of the six metrics: **accuracy 0.7510, macro-recall 0.7717 and MCC 0.6827**. Distance weighting with the Manhattan metric (`p=1`) is what makes this work — L1 is more robust than Euclidean in 27 dimensions, where L2 distances concentrate. Its real advantage is on the *rare* classes: `Dirtiness` recall reaches 0.9286 (13 of 14 samples), the best of any model, because a local neighbourhood of 7 can represent a small, tight cluster that a global parametric model averages away. The trade-off is the lowest macro-precision of the four strong models (0.7443) — it over-reaches into `Bumps`. It is also the only model whose cost is paid at prediction time, and its 196 KB artefact is the largest because it must store the whole training set. |
| **Naive Bayes** | Clearly the **weakest on hard predictions** — accuracy 0.6235, F1 0.6424 and MCC 0.5611, some 12 points behind the leaders. The cause is its conditional-independence assumption, which this dataset violates about as badly as possible: `Pixels_Areas` and `X_Perimeter` correlate at r = 0.97, and `LogOfAreas` with `SigmoidOfAreas` at r = 0.88, so the same evidence about defect size is counted several times over and the posteriors become badly overconfident. The damage is visible in `Other_Faults`, where recall drops to 0.3095 — it scatters the catch-all bucket across the named classes, inflating recall on small classes (`Dirtiness` recall 0.7857) at a ruinous precision cost (0.3056). Yet its **AUC is a healthy 0.9250**, above the Decision Tree: the *ranking* it induces is sound even though its calibrated probabilities and therefore its arg-max decisions are not. Its redeeming feature is speed — 0.12 s to fit, versus 6.3 s for Logistic Regression. |
| **Support Vector Machine** | The best overall model: top **AUC (0.9511)**, top **macro-precision (0.8018)** and top **macro-F1 (0.7654)**, plus the best cross-validated score (0.7798), so its lead is not an artefact of one lucky split. An RBF kernel with `C=10` and `gamma=0.01` gives a boundary flexible enough for the non-linear fault regions but smooth enough not to chase noise. It is the most *conservative* model — `Dirtiness` precision is a perfect 1.0000 and `K_Scatch` precision 0.9780 — which is why it leads on precision while trailing KNN on recall: when it names a fault, it is right, but it abstains into `Other_Faults` more readily. Note its accuracy (0.7428) is marginally *below* KNN's despite the higher F1, which is the imbalance showing through: the SVM spreads its competence more evenly across the seven classes. |
| **Overall, winner for your dataset?** | **Support Vector Machine (RBF)** — but narrowly, and with a caveat worth stating. The SVM wins AUC, precision and F1; KNN wins accuracy, recall and MCC. The tie-breakers favour the SVM: it has the best macro-F1 (the fairest single summary under imbalance), the best AUC by 0.0125 over the runner-up, and the best cross-validated macro-F1, which is the only estimate not computed on the test split. On 486 test rows the accuracy gap to KNN (0.7510 vs 0.7428, ~4 samples) is well inside sampling noise, so the two should be read as **statistically indistinguishable on hard predictions** and separated mainly by ranking quality, where the SVM is genuinely ahead. Practical recommendation: deploy the **SVM** if the mill wants few false alarms, and **KNN** if it would rather catch every rare `Dirtiness` defect and accept the re-inspections. No model exceeds 0.78 accuracy, and the ceiling is the `Other_Faults` bucket itself — the most valuable next step would be relabelling it, not swapping in a fancier classifier. |

---

## Streamlit application

The app loads the five `.joblib` pipelines and scores whatever CSV it is given.
It **never calls `.fit()`** — the uploaded file is treated strictly as held-out
evaluation data, as the assignment requires.

### Required features

| # | Requirement | Where it is implemented |
| --- | --- | --- |
| a | **Dataset upload option (CSV)** | Sidebar → *1 · Test data* → file uploader. A checkbox falls back to the bundled 486-row `test_data.csv`, which is also downloadable from the sidebar. |
| b | **Model selection dropdown** | Sidebar → *2 · Model* → all five classifiers plus a **Compare all five models** mode. |
| c | **Display of evaluation metrics** | Six metric tiles (Accuracy, AUC, Precision, Recall, F1, MCC) per model; in compare mode, a colour-graded table of all five models with a "best per metric" summary and a sortable bar chart. |
| d | **Confusion matrix / classification report** | Dedicated tabs: an annotated confusion-matrix heatmap (row-normalised or raw counts), the full scikit-learn classification report, and a per-class precision/recall/F1 table that flags each model's weakest class. |

### Additional features

- **Metric deltas against the report.** Each tile shows the gap between the live
  score and the value recorded in `metadata.json` at training time, so a grader can
  see at a glance that the deployed artefacts reproduce this README exactly.
- **Input validation.** Missing feature columns, a missing target column, rows with
  NaNs and unseen class labels each produce a specific, readable message instead of a
  stack trace.
- **Graceful partial uploads.** Macro one-vs-rest AUC is accumulated class by class,
  so uploading a two-class slice of the test set still yields a valid AUC; with only
  one class present the app reports `n/a` and explains why, rather than crashing.
  All macro averages are taken over the classes actually present in the file, so the
  full `test_data.csv` reproduces the table above while a subset stays coherent.
- **Prediction inspection and export.** A per-row table of actual vs predicted with
  model confidence, a "misclassified rows only" filter, and CSV downloads of both the
  comparison table and every model's predictions.
- **Tuned hyper-parameters** for all five models are shown in a sidebar expander.

### Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. **New app** → select the repository → branch `main` → main file `app.py`.
4. Under **Advanced settings**, set the Python version to **3.12** or **3.13** — the
   pinned dependency versions in `requirements.txt` need one of these.
5. **Deploy**.

---

## Notes on methodology

- **Random seed 42** is fixed for the split, the CV folds and every stochastic
  estimator, so all results are reproducible.
- **No leakage**: scalers are fitted inside pipelines, on training folds only.
- **The test split is never used for tuning.** Hyper-parameters come from 5-fold CV
  on the training split alone; the 486 test rows are touched exactly once, to produce
  the final table.
- **Macro averaging throughout**, because of the 12:1 class imbalance.
- **MCC** is included as the most conservative summary: it accounts for all cells of
  the confusion matrix and stays meaningful under imbalance, which is why the
  0.56–0.68 MCC band is a fairer picture of difficulty than the 0.62–0.75 accuracy band.

## Dataset citation

> Buscema, M., Terzi, S., & Tastle, W. (2010). *Steel Plates Faults* [Dataset].
> UCI Machine Learning Repository. https://doi.org/10.24432/C5J88N
> Donated by Semeion, Research Center of Sciences of Communication, Rome, Italy.
