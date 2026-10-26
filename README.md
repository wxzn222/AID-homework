# AID-homework

《人工智能开发技术（Python）》课程实验作业仓库。

| 项目 | 内容 |
| :-- | :-- |
| 姓名 | 何嘉欣 |
| 学号 | 2410311411 |
| 班级 | 24计科4班 |
| 专业 | 计算机科学与技术 |
| 运行基线 | Windows、Python 3.12、CPU、约 16 GB 内存 |

## 目录结构

~~~text
AID-homework/
└─ 01. Python机器学习基本流程与结果分析/
   ├─ 实验01_ml_pipeline_work/          实验骨架的个人工作副本（已完成全部 FILL）
   │  ├─ config.py                      随机种子、测试比例、路径与类别名
   │  ├─ data_utils.py                  加载数据、标签反转、分层划分
   │  ├─ inspect_data.py                数据质量检查与描述统计
   │  ├─ models.py                      逻辑回归与决策树两个 Pipeline
   │  ├─ evaluation.py                  指标、混淆矩阵、错误样本
   │  ├─ plots.py                       无界面绘图
   │  ├─ training.py                    训练两个模型并汇总指标
   │  ├─ run_experiment.py              训练主脚本，按 F1 保存最佳模型
   │  ├─ verify_saved_model.py          新进程重载并比对预测
   │  ├─ test_experiment.py             最小回归检查
   │  ├─ student_identity.py            学号登记与阶段身份显示
   │  ├─ requirements.txt               固定依赖版本
   │  ├─ outputs/                       12 个实验产物
   │  └─ models/best_pipeline.joblib    最佳模型（完整 Pipeline）
   └─ 《人工智能开发技术》实验01报告_何嘉欣.docx
~~~

## 实验 01 运行方式

~~~powershell
Set-Location "01. Python机器学习基本流程与结果分析\实验01_ml_pipeline_work"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe inspect_data.py
.\.venv\Scripts\python.exe run_experiment.py
.\.venv\Scripts\python.exe verify_saved_model.py
~~~

## 实验 01 关键结果

- 数据：569 条样本、30 个特征；`class_0` 357 条、`class_1` 212 条。
- 划分：`random_state=42`、`test_size=0.20`、`stratify=target`；训练 455 条、测试 114 条。
- 逻辑回归：Accuracy 0.9649、Precision 0.9750、Recall 0.9286、F1 0.9512。
- 决策树：Accuracy 0.9035、Precision 0.9189、Recall 0.8095、F1 0.8608。
- 最佳模型：`logistic_regression`，混淆矩阵 `[[71, 1], [3, 39]]`。
- 重载验证：`loaded_model=Pipeline`、`sample_count=114`、`same_predictions=True`。
- 模型 SHA256：`FCB4FA0C4A9E168289BD87E8D496126997185D8346D2D2650CD56728DC19E9E3`

## 说明

- `outputs/` 与 `models/` 是本人运行产生的实验产物，按评分标准一并提交。
- 不提交 `.venv`、Python 缓存与临时文件。
- 类别使用中性标记 `class_0` / `class_1`，不做行业语义假设。
## 实验 02～05

| 实验 | 工作目录 | 主要内容 | 关键结果 |
| :-- | :-- | :-- | :-- |
| 02 | `02. PyTorch神经网络训练与模型管理/实验02_pytorch_mlp_work` | PyTorch MLP 训练、验证、测试与模型保存/重载 | 测试集 Accuracy 0.8525，Macro-F1 0.8550 |
| 03 | `03. CNN迁移学习图像分类/实验03_cnn_transfer_work` | ResNet18 迁移学习，冻结骨干与部分微调对比 | 冻结 Accuracy 0.784；部分微调 Accuracy 0.834 |
| 04 | `04. 推荐系统与排序评价/实验04_recsys_work` | MovieLens 100K、ItemCF 与流行度基线、Top-K 排序评价 | ItemCF Recall@10 0.085987、NDCG@10 0.044131 |
| 05 | `05. Transformer与大语言模型推理基础/实验05_transformer_llm_work` | Qwen2.5-0.5B CPU 推理、token、解码、上下文裁剪与结构化校验 | Pydantic 校验 PASS；在线 API 因缺少配置为 SKIP |

每个实验目录均包含个人工作副本和对应 Word/PDF 报告。`outputs/` 中保留本人运行产生的指标、图表和模型文件；数据集、`.venv`、`hf_cache` 与 Python 缓存不提交。