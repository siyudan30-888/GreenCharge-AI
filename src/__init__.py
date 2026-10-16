"""GreenCharge-AI 核心计算包。

分层约定（**这是本项目最重要的一条结构规则**）：

    app.py          只做展示。**不做任何计算。**
    scripts/        命令行入口，复用 src 的函数，不重复实现。
    src/schema.py   数据的唯一入口：载入 / 校验 / 派生
    src/synthetic.py 模拟数据生成器（唯一的假设来源）
    src/metrics.py  电力负荷指标与碳核算
    src/features.py 特征工程（防泄漏）
    src/models.py   模型训练 / 评估 / 预测
    src/strategy.py 有序充电策略与假设面板

屏幕上出现的每一个数字，都由本包里的某个具名函数产出，
于是「这个数是怎么来的」永远有一个明确答案：
「看 `src/metrics.py` 的 `load_factor`」。

这条规则就是全部的答辩防线。
"""

__version__ = "2.0.0"

__all__ = [
    "features",
    "metrics",
    "models",
    "schema",
    "strategy",
    "synthetic",
]
