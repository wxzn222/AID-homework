from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def explicit_to_positive(ratings: pd.DataFrame, threshold: int = 4) -> pd.DataFrame:
    """把显式评分转换为按时间排序的隐式正反馈。"""
    # FILL-01: 保留 rating >= threshold 的行，稳定排序后重置索引。
    positives = ratings.loc[ratings["rating"] >= threshold].copy()
    return positives.sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    ).reset_index(drop=True)


def temporal_leave_one_out(positives: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每位合格用户最后一次正反馈作测试，其余作训练。"""
    # FILL-02: 统计每位用户记录数，只保留至少两条正反馈的用户。
    counts = positives.groupby("user_id")["item_id"].transform("size")
    eligible = positives.loc[counts >= 2].copy()

    # FILL-03: 按时间排序，取每位用户最后一条并构造互斥训练/测试集。
    eligible = eligible.sort_values(
        ["user_id", "timestamp", "item_id"], kind="mergesort"
    )
    test_index = eligible.groupby("user_id", sort=True).tail(1).index
    test = eligible.loc[test_index].copy()
    train = eligible.drop(index=test_index).copy()
    train_pairs = set(zip(train["user_id"], train["item_id"], strict=True))
    test_pairs = set(zip(test["user_id"], test["item_id"], strict=True))
    if train_pairs & test_pairs:
        raise AssertionError("训练/测试交互发生重叠")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_interaction_matrix(
    train: pd.DataFrame, user_ids: np.ndarray, item_ids: np.ndarray
) -> tuple[csr_matrix, dict[int, int], dict[int, int]]:
    """把原始 ID 映射为连续下标并构造 float32 CSR 矩阵。"""
    # FILL-04: 建立两个映射，并用全 1 数据构造用户—物品矩阵。
    user_to_index = {int(v): i for i, v in enumerate(user_ids)}
    item_to_index = {int(v): i for i, v in enumerate(item_ids)}
    row = train["user_id"].map(user_to_index).to_numpy()
    col = train["item_id"].map(item_to_index).to_numpy()
    values = np.ones(len(train), dtype=np.float32)
    matrix = csr_matrix(
        (values, (row, col)), shape=(len(user_ids), len(item_ids)), dtype=np.float32
    )
    return matrix, user_to_index, item_to_index