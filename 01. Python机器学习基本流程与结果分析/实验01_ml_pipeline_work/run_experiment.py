"""阶段二至五：划分、训练、评价、保存产物。"""

import json
import platform
import time

import joblib
import sklearn

from config import MODEL_DIR, OUTPUT_DIR
from plots import save_confusion_plot, save_metric_plot
from training import train_models


def main() -> None:
    started_at = time.perf_counter()
    metrics_frame, matrices, fitted_models = train_models()
    save_confusion_plot(matrices)
    save_metric_plot(metrics_frame)

    best_name = str(metrics_frame.iloc[0]["model"])
    model_path = MODEL_DIR / "best_pipeline.joblib"
    joblib.dump(fitted_models[best_name], model_path)
    elapsed = time.perf_counter() - started_at
    summary = {
        "best_model": best_name,
        "elapsed_seconds": round(elapsed, 3),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"best_model={best_name}")
    print(f"saved_model={model_path.relative_to(model_path.parent.parent)}")
    print(f"elapsed_seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
