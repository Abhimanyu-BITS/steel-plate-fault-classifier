# Model comparison - Steel Plates Faults (held-out test split)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7222 | 0.9386 | 0.7632 | 0.7113 | 0.7316 | 0.6404 |
| Decision Tree | 0.7449 | 0.8819 | 0.7585 | 0.7157 | 0.7349 | 0.6697 |
| KNN | 0.7510 | 0.9359 | 0.7443 | 0.7717 | 0.7542 | 0.6827 |
| Naive Bayes | 0.6235 | 0.9250 | 0.6480 | 0.7101 | 0.6424 | 0.5611 |
| Support Vector Machine | 0.7428 | 0.9511 | 0.8018 | 0.7378 | 0.7654 | 0.6672 |

Precision / Recall / F1 are macro-averaged; AUC is macro one-vs-rest.
