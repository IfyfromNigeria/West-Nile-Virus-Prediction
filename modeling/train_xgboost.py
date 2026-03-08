import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_xgb(X_train, y_train, X_test, y_test):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        eval_metric="auc"
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs)

    }

import xgboost as xgb
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix,
f1_score,
precision_score,
recall_score,
roc_auc_score,
)

def train_xgb(X_train, y_train, X_test, y_test) -> dict:
"""
Train an XGBoost classifier and evaluate it on the held-out test set.
Class imbalance is handled via scale_pos_weight, which is computed
automatically from the training labels.

Parameters
----------
X_train, y_train : training data
X_test,  y_test  : held-out test data

Returns
-------
dict with keys: accuracy, precision, recall, f1, roc_auc,
                classification_report, confusion_matrix
"""
# Automatically balance positive / negative class weight
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"[train_xgb] Class imbalance ratio (neg/pos): {ratio:.2f}")

model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=ratio,
    eval_metric="auc",
    use_label_encoder=False,
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy":              accuracy_score(y_test, preds),
    "precision":             precision_score(y_test, preds),
    "recall":                recall_score(y_test, preds),
    "f1":                    f1_score(y_test, preds),
    "roc_auc":               roc_auc_score(y_test, probs),
    "classification_report": classification_report(
        y_test, preds, target_names=["No Virus", "WNV Present"]
    ),
    "confusion_matrix":      confusion_matrix(y_test, preds).tolist(),
}

print("\n─── Model Evaluation ───────────────────────────────────────────")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    print(f"  {k:<12}: {metrics[k]:.4f}")
print("\n" + metrics["classification_report"])

return metrics
