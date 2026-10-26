from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation import ndcg_at_k, recall_at_k

output_dir = Path("outputs")
recommendations = pd.read_csv(output_dir / "user_recommendations.csv")
test = pd.read_csv(output_dir / "test_positive.csv")
targets = test.set_index("user_id")["item_id"].astype(int).to_dict()

rows = []
for method, group in recommendations.groupby("method"):
    for k in (5, 10):
        recall_values = []
        ndcg_values = []
        for user_id, user_rows in group.groupby("user_id"):
            ranked = (
                user_rows.loc[user_rows["rank"] <= k]
                .sort_values("rank")["item_id"]
                .astype(int)
                .tolist()
            )
            relevant = {targets[int(user_id)]}
            recall_values.append(recall_at_k(ranked, relevant, k))
            ndcg_values.append(ndcg_at_k(ranked, relevant, k))
        rows.append({
            "method": method,
            "k": k,
            "users": len(recall_values),
            "recall_at_k": sum(recall_values) / len(recall_values),
            "ndcg_at_k": sum(ndcg_values) / len(ndcg_values),
        })

result = pd.DataFrame(rows).sort_values(["k", "method"])
print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
