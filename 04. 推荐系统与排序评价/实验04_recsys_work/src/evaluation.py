from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    """计算 Top-K 对相关物品集合的召回率。"""
    # FILL-09: 处理空相关集，再计算交集数除以相关物品数。
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    return len(set(top_k) & relevant) / len(relevant)


def ndcg_at_k(recommended: Iterable[int], relevant: set[int], k: int) -> float:
    """计算二值相关性下归一化折损累计增益。"""
    # FILL-10: 计算 DCG、理想 IDCG，并返回二者之比。
    if not relevant or k <= 0:
        return 0.0
    top_k = list(recommended)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(top_k, start=1)
        if item_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def first_hit_rank(recommended: Iterable[int], relevant: set[int], k: int) -> int | None:
    """返回第一个相关物品的 1 起始排名；未命中返回 None。"""
    # FILL-11: 枚举前 K 项，发现相关物品时立即返回排名。
    for rank, item_id in enumerate(list(recommended)[:k], start=1):
        if item_id in relevant:
            return rank
    return None