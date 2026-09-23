"""集中计算指标、混淆矩阵，并提取错误分类样本。"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, features: pd.DataFrame, target: pd.Series):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    # zero_division=0：模型完全没预测出正类时返回 0，而不是产生警告。
    metrics = {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
    }
    matrix = confusion_matrix(target, predictions, labels=[0, 1])

    # 保留原始行索引，便于回到输入特征定位错误样本。
    errors = features.loc[target.to_numpy() != predictions].copy()
    errors["true_label"] = target.loc[errors.index]
    errors["predicted_label"] = predictions[target.to_numpy() != predictions]
    errors["error_type"] = np.where(
        (errors["true_label"] == 1) & (errors["predicted_label"] == 0),
        "FN",
        "FP",
    )
    if metrics is None or matrix is None or errors is None:
        raise NotImplementedError("请完成当前教学填空")
    return metrics, matrix, errors, predictions
