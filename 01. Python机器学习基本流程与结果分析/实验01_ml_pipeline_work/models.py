"""建立两个可比较的 Pipeline，所有预处理只在训练集上拟合。"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import SEED


def build_models() -> dict[str, Pipeline]:
    """返回名称到未训练 Pipeline 的映射。"""
    logistic_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=SEED, max_iter=2000)),
        ]
    )

    tree_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=SEED, max_depth=4, min_samples_leaf=3
                ),
            ),
        ]
    )
    if logistic_pipeline is None or tree_pipeline is None:
        raise NotImplementedError("请完成当前教学填空")
    return {"logistic_regression": logistic_pipeline, "decision_tree": tree_pipeline}
