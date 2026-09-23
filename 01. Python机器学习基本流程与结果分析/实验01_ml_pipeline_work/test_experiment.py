"""实验 01 的最小回归检查。"""

from data_utils import load_dataset, make_split
from models import build_models


def test_dataset_and_stratified_split_have_expected_shapes():
    frame = load_dataset()
    x_train, x_test, y_train, y_test = make_split(frame)

    assert frame.shape == (569, 31)
    assert x_train.shape == (455, 30)
    assert x_test.shape == (114, 30)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_model_factory_returns_two_pipelines():
    models = build_models()

    assert set(models) == {"logistic_regression", "decision_tree"}
    assert all(hasattr(model, "fit") for model in models.values())


if __name__ == "__main__":
    test_dataset_and_stratified_split_have_expected_shapes()
    test_model_factory_returns_two_pipelines()
    print("test_experiment: 2 passed")
