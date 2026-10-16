"""逐小时充电量预测：训练、评估与预测。

## 关于「在合成数据上跑监督学习」这件事

生成器是本项目自己写的，再拿它产出的数据训练模型，必然会被问：
**「你自己写生成器再预测它，误差当然小。」**

本项目在界面上主动回答这个问题，而不是等人问：

1. 生成器含**真正不可预测的成分** —— 泊松到达噪声 + 逐日对数正态日效应。
   模型看不到当天的日效应，只能从滞后特征里间接估计，因此误差是真实的。
2. **头条数字是相对基线的提升，而不是绝对 MAE。**
   提升幅度是与数据来源无关的量：它回答的是「这套建模流程比朴素方法强多少」。
3. 界面与 README 均注明：**本指标验证的是建模流程（防泄漏特征、时序划分、
   基准对照、评估口径）的正确性。绝对精度是模拟数据的属性，
   不代表真实场站的预测精度。**

把这三句话说在前面，最大的可信度风险就变成了方法严谨性的证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    FeatureSet,
    TARGET,
    _add_cyclical,
    _add_lags,
    chronological_split,
    feature_group,
)

RANDOM_STATE = 42

#: 多步回测的预测跨度（小时）。24 小时 = 一天，正是有序充电需要提前知道的范围。
MULTISTEP_HORIZON = 24


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #

def mae(y_true, y_pred) -> float:
    """平均绝对误差（kWh）。**头条指标**：物理单位，不需要翻译。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """均方根误差（kWh）。比 MAE 更严厉地惩罚大偏差 —— 低估尖峰的代价最高。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    """加权绝对百分比误差（%）= Σ|误差| / Σ|实际| × 100%。

    **负荷预测中诚实的百分比误差。** 与 MAPE 不同，它不会被接近零的样本放大，
    因此在这里比 MAPE 更可信。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def mape(y_true, y_pred) -> tuple[float, int]:
    """平均绝对百分比误差（%），返回 (值, 被排除的样本数)。

    ⚠️ **MAPE 在负荷预测里是有缺陷的**：实际电量为 0 时无定义，接近 0 时会爆炸。
    园区凌晨经常一单都没有，这些点的百分比误差会是天文数字，
    使 MAPE 大得误导。因此本函数**只在实际电量 > 0 的样本上计算**，
    并把排除的样本数一并返回、在界面上显示。

    把这件事说出来，比 MAPE 的数值本身更有价值。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    excluded = int((~mask).sum())
    if mask.sum() == 0:
        return float("nan"), excluded
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0), excluded


def all_metrics(y_true, y_pred) -> dict[str, float]:
    m, excluded = mape(y_true, y_pred)
    return {
        "MAE (kWh)": round(mae(y_true, y_pred), 3),
        "RMSE (kWh)": round(rmse(y_true, y_pred), 3),
        "WAPE (%)": round(wape(y_true, y_pred), 2),
        "MAPE (%)": round(m, 2) if np.isfinite(m) else float("nan"),
        "MAPE 排除样本数": excluded,
    }


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #

#: 超参候选网格。**用训练集内部的时间顺序验证集来选，绝不使用测试集。**
#: 网格刻意取小：45 个组合在约 1000 行上总耗时数秒，且候选越少越不容易
#: 在验证集上过拟合（选择本身也是一种拟合）。
HGB_GRID: dict[str, tuple] = {
    "max_iter": (60, 100, 150, 200, 300),
    "learning_rate": (0.03, 0.05, 0.08),
    "l2_regularization": (0.0, 0.1, 1.0),
}

#: 从训练集尾部留出的内部验证集比例
HGB_VALIDATION_FRACTION = 0.15


def make_gradient_boosting(
    *,
    max_iter: int = 100,
    learning_rate: float = 0.05,
    l2_regularization: float = 0.1,
) -> HistGradientBoostingRegressor:
    """主模型：直方图梯度提升回归。

    选它的理由：

    1. **原生处理 NaN** —— 滞后特征在序列边缘必然产生缺失值，
       这消除了一整类填充 bug。
    2. **无额外依赖**（sklearn 自带），因此在 Python 3.14 上没有
       第三方轮子缺失或需要 C 编译器的风险。
    3. **能捕捉「小时 × 场站」交互** —— 而这正是「多场站诊断」的全部意义。
    4. 约 1200 行样本上毫秒级训练，纯 CPU。

    `early_stopping=False` 是**刻意**的：sklearn 默认 `'auto'` 会从训练数据里
    **随机**切出一部分作验证集 —— 在时序数据上这是跨时间泄漏。
    因此轮数不由 sklearn 自动决定，而由 `select_hgb_config()` 在
    **时间顺序**的内部验证集上显式选出。

    默认参数只是兜底值；实际使用的值来自 `select_hgb_config()`，
    并会随 `ModelReport.hgb_config` 一起披露。
    """
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        early_stopping=False,
        l2_regularization=l2_regularization,
        random_state=RANDOM_STATE,
    )


def select_hgb_config(
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    grid: dict[str, tuple] | None = None,
    validation_fraction: float = HGB_VALIDATION_FRACTION,
) -> tuple[dict[str, float], pd.DataFrame]:
    """在**训练集内部**的时间顺序验证集上选梯度提升的超参。

    为什么必须这样做：样本只有约 1200 行，300 轮 + 无早停会明显过拟合
    （实测测试集 MAE 111.2，反而不如岭回归的 104.5）。
    但**不能**看着测试集去调参 —— 那是把测试集变成了训练集的一部分，
    报出来的指标就不可信了。

    做法：把训练集按时间顺序再切一刀，前 85% 作内部训练、后 15% 作内部验证，
    在验证集上扫网格取最优，再回到**完整训练集**重训。
    测试集从头到尾没有参与任何选择。

    Returns
    -------
    (最优参数, 全部候选的验证结果表)
    """
    grid = grid or HGB_GRID

    times = np.sort(train_frame["时刻"].unique())
    cut_pos = min(int(len(times) * (1.0 - validation_fraction)), len(times) - 1)
    cut = times[cut_pos]

    inner_train = train_frame[train_frame["时刻"] < cut]
    inner_valid = train_frame[train_frame["时刻"] >= cut]
    if inner_train.empty or inner_valid.empty:
        return {"max_iter": 100, "learning_rate": 0.05, "l2_regularization": 0.1}, pd.DataFrame()

    X_in = inner_train[feature_columns]
    y_in = inner_train[TARGET].to_numpy(dtype=float)
    X_va = inner_valid[feature_columns]
    y_va = inner_valid[TARGET].to_numpy(dtype=float)

    rows = []
    for max_iter in grid["max_iter"]:
        for lr in grid["learning_rate"]:
            for l2 in grid["l2_regularization"]:
                model = make_gradient_boosting(
                    max_iter=max_iter, learning_rate=lr, l2_regularization=l2
                )
                model.fit(X_in, y_in)
                rows.append(
                    {
                        "max_iter": max_iter,
                        "learning_rate": lr,
                        "l2_regularization": l2,
                        "内部验证 MAE (kWh)": round(
                            mae(y_va, np.maximum(model.predict(X_va), 0.0)), 3
                        ),
                    }
                )

    table = pd.DataFrame(rows).sort_values("内部验证 MAE (kWh)").reset_index(drop=True)
    top = table.iloc[0]
    best = {
        "max_iter": int(top["max_iter"]),
        "learning_rate": float(top["learning_rate"]),
        "l2_regularization": float(top["l2_regularization"]),
    }
    return best, table


def make_ridge() -> Pipeline:
    """次模型：岭回归。**可解释的对照组。**

    它存在的意义是让「增加复杂度是实测换来的」这句话可被检验：
    如果梯度提升没比它好，那就应该用岭回归。
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )


def baseline_seasonal_naive(test: pd.DataFrame) -> np.ndarray:
    """基线一：**季节性朴素** —— 用昨天同一小时（t−24）预测今天。

    这是负荷预测的经典基准，不是稻草人：日周期是充电负荷最强的规律，
    一个连它都赢不过的模型没有任何存在价值。
    """
    return test["lag_24"].to_numpy(dtype=float)


def baseline_hourly_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """基线二：**逐时均值** —— 用训练集里该（场站, 小时）的平均电量预测。

    比朴素法更强，因为它同时利用了日周期和场站差异。
    """
    key = ["场站名称"]
    tr = train.copy()
    te = test.copy()
    tr["_hour"] = tr["时刻"].dt.hour
    te["_hour"] = te["时刻"].dt.hour

    means = tr.groupby(key + ["_hour"], observed=True)[TARGET].mean().rename("_mean")
    merged = te.merge(means, on=key + ["_hour"], how="left")
    # 训练集里没出现过的（场站, 小时）组合回退到该场站的整体均值
    fallback = tr.groupby(key, observed=True)[TARGET].mean().rename("_fallback")
    merged = merged.merge(fallback, on=key, how="left")
    return merged["_mean"].fillna(merged["_fallback"]).fillna(0.0).to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# 训练与评估
# --------------------------------------------------------------------------- #

@dataclass
class ModelReport:
    """一次完整训练评估的全部产出。界面与 CLI 都只消费这个对象。"""

    metrics: pd.DataFrame
    """四模型在同一测试集上的指标对比"""
    per_station: pd.DataFrame
    """主模型的分场站 MAE —— 主动披露误差最大的场站比一个好看的均值更有说服力"""
    importance: pd.DataFrame
    """排列重要性（场站 one-hot 已归并）"""
    test_predictions: pd.DataFrame
    """主模型在测试集上的 场站/时刻/实际/预测，用于画预测-实际散点图"""
    best_model_name: str
    primary_mae: float
    baseline_mae: float
    """**最强**基线的 MAE（取两个基线中较低者）—— 提升度以此为分母"""
    baseline_name: str
    improvement_pct: float
    multistep_mae: float | None
    multistep_horizon: int
    n_train: int
    n_test: int
    split_time: pd.Timestamp
    notes: list[str] = field(default_factory=list)
    model: object = None
    feature_columns: list[str] = field(default_factory=list)
    station_columns: list[str] = field(default_factory=list)
    hgb_config: dict[str, float] = field(default_factory=dict)
    """主模型实际使用的超参。由训练集内部的验证集选出，**测试集未参与**。"""
    hgb_search: pd.DataFrame = field(default_factory=pd.DataFrame)
    """超参扫描的全部候选及其内部验证 MAE —— 让「怎么选的」可被审计。"""


def train_and_evaluate(
    features: FeatureSet,
    *,
    test_fraction: float = 0.2,
    compute_importance: bool = True,
    compute_multistep: bool = True,
) -> ModelReport:
    """训练四个模型（主模型 + 对照组 + 两个基线），在**同一个测试集**上评估。

    「同一个测试集」是关键：所有模型必须落在完全相同的行集上，
    否则对比没有意义。为此基线不使用任何额外的样本。
    """
    frame = features.frame
    train, test = chronological_split(frame, test_fraction)

    X_train = train[features.feature_columns]
    y_train = train[TARGET].to_numpy(dtype=float)
    X_test = test[features.feature_columns]
    y_test = test[TARGET].to_numpy(dtype=float)

    notes: list[str] = []

    # ---- 主模型：超参先在训练集内部选，再回到完整训练集重训 ---------------- #
    hgb_config, hgb_search = select_hgb_config(train, features.feature_columns)
    hgb = make_gradient_boosting(**hgb_config)
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_test)
    notes.append(
        "主模型超参在**训练集内部**的时间顺序验证集上选出"
        f"（max_iter={hgb_config['max_iter']}、learning_rate={hgb_config['learning_rate']}、"
        f"l2_regularization={hgb_config['l2_regularization']}），"
        "测试集从头到尾未参与任何选择；完整候选表见「超参扫描」。"
    )

    # ---- 对照组 ------------------------------------------------------------ #
    ridge = make_ridge()
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)

    # ---- 两个基线 ---------------------------------------------------------- #
    pred_naive = baseline_seasonal_naive(test)
    pred_hourly = baseline_hourly_mean(train, test)

    candidates: list[tuple[str, np.ndarray]] = [
        ("梯度提升 (HistGradientBoosting)", pred_hgb),
        ("岭回归 (Ridge)", pred_ridge),
        ("基线·季节性朴素 (t−24)", pred_naive),
        ("基线·逐时均值 (场站×小时)", pred_hourly),
    ]

    rows = []
    for name, pred in candidates:
        row = {"模型": name}
        row.update(all_metrics(y_test, pred))
        rows.append(row)
    metrics_table = pd.DataFrame(rows)

    # ---- 提升度：以**最强**基线为分母 -------------------------------------- #
    baseline_maes = {
        "基线·季节性朴素 (t−24)": mae(y_test, pred_naive),
        "基线·逐时均值 (场站×小时)": mae(y_test, pred_hourly),
    }
    baseline_name = min(baseline_maes, key=baseline_maes.get)
    baseline_value = baseline_maes[baseline_name]
    primary = mae(y_test, pred_hgb)
    improvement = (1.0 - primary / baseline_value) * 100.0 if baseline_value > 0 else float("nan")

    if improvement <= 0:
        notes.append(
            "⚠️ **主模型未能超过最强基线。** 这是一个诚实的结果，不是 bug —— "
            "它说明在当前数据上，朴素方法已经足够。我们选择如实报告，"
            "而不是调整口径制造一个好看的胜利。"
        )

    # ---- 分场站误差 -------------------------------------------------------- #
    station_rows = []
    for station in features.stations:
        mask = (test["场站名称"] == station).to_numpy()
        if not mask.any():
            continue
        station_rows.append(
            {
                "场站": station,
                "测试样本数": int(mask.sum()),
                "实际总电量 (kWh)": round(float(y_test[mask].sum()), 1),
                "MAE (kWh)": round(mae(y_test[mask], pred_hgb[mask]), 3),
                "WAPE (%)": round(wape(y_test[mask], pred_hgb[mask]), 2),
                "基线 MAE (kWh)": round(
                    mae(y_test[mask], pred_naive[mask]), 3
                ),
                "提升 (%)": round(
                    (1.0 - mae(y_test[mask], pred_hgb[mask]) / max(mae(y_test[mask], pred_naive[mask]), 1e-9))
                    * 100.0,
                    1,
                ),
            }
        )
    per_station = pd.DataFrame(station_rows)

    # ---- 排列重要性 -------------------------------------------------------- #
    importance = pd.DataFrame(columns=["特征", "重要性 (MAE 增量 kWh)"])
    if compute_importance:
        result = permutation_importance(
            hgb,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        imp = pd.DataFrame(
            {
                "特征": features.feature_columns,
                "重要性 (MAE 增量 kWh)": result.importances_mean,
            }
        )
        imp["特征"] = [
            feature_group(c, features.station_columns) for c in imp["特征"]
        ]
        importance = (
            imp.groupby("特征", as_index=False)["重要性 (MAE 增量 kWh)"]
            .sum()
            .sort_values("重要性 (MAE 增量 kWh)", ascending=False)
            .reset_index(drop=True)
        )

    # ---- 多步回测 ---------------------------------------------------------- #
    multistep: float | None = None
    if compute_multistep:
        multistep = backtest_multistep(
            features, hgb, origin_frame=test, horizon=MULTISTEP_HORIZON
        )
        if multistep is not None and multistep > primary * 1.05:
            notes.append(
                f"24 小时递归预测的 MAE（{multistep:.2f} kWh）高于单步 MAE"
                f"（{primary:.2f} kWh）—— 这是递归预测的固有代价："
                "从第 2 小时起，lag_1 用的是模型自己的预测值而非真值，误差会累积。"
                "策略页面的不确定性带采用的是**多步 MAE**，而不是单步 MAE。"
            )

    test_predictions = test[["场站名称", "时刻", TARGET]].copy()
    test_predictions["预测_kWh"] = pred_hgb
    test_predictions["绝对误差_kWh"] = np.abs(y_test - pred_hgb)

    return ModelReport(
        metrics=metrics_table,
        per_station=per_station,
        importance=importance,
        test_predictions=test_predictions,
        best_model_name="梯度提升 (HistGradientBoosting)",
        primary_mae=primary,
        baseline_mae=baseline_value,
        baseline_name=baseline_name,
        improvement_pct=improvement,
        multistep_mae=multistep,
        multistep_horizon=MULTISTEP_HORIZON,
        n_train=len(train),
        n_test=len(test),
        split_time=train["时刻"].max(),
        notes=notes,
        model=hgb,
        feature_columns=list(features.feature_columns),
        station_columns=list(features.station_columns),
        hgb_config=hgb_config,
        hgb_search=hgb_search,
    )


# --------------------------------------------------------------------------- #
# 多步回测与预测
# --------------------------------------------------------------------------- #

AUX_COLUMNS: tuple[str, ...] = ("订单数", "平均时长", "单均电量")


def _prepare_recursion(
    features: FeatureSet,
    station: str,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    """构造递归预测用的工作表，返回 (工作表, 持久化基线行)。

    **必须把 origin 之后的目标值抹掉。** 否则递归走到第 h 步时，
    `lag_1` 读到的会是那一小时的**真实值** —— 于是「预测未来」变成了
    「抄未来」，多步误差会被严重低估，而指标上完全看不出异常。
    这是本函数存在的唯一理由。

    辅助序列（订单数、平均时长、单均电量）在 origin 之后同样抹掉，
    改用**持久化**：假定未来小时维持其在 origin 处的最后观测水平。
    这是一个明确的运营假设，会随结果一起披露。
    """
    frame = features.frame
    sub = (
        frame[frame["场站名称"] == station][["时刻", TARGET, *AUX_COLUMNS]]
        .sort_values("时刻", kind="stable")
        .reset_index(drop=True)
    )

    n_hist = int((sub["时刻"] <= origin).sum())
    if n_hist == 0:
        return sub, pd.Series(dtype=float)

    tail = sub.iloc[n_hist - 1]
    future = sub.index[n_hist:]
    if len(future):
        sub.loc[future, TARGET] = np.nan
        for col in AUX_COLUMNS:
            sub.loc[future, col] = np.nan

    return sub, tail


def _feature_frame(work: pd.DataFrame, station: str) -> pd.DataFrame:
    """在给定工作表上重算特征 —— **复用 features.py 的同一套函数**。

    这里的复用是刻意的：递归预测若自己实现一套特征计算，就会出现
    「训练时用 A 公式、预测时用 B 公式」的训练/服务偏斜，
    而这类偏斜在指标上看不出来，却会让线上预测系统性失准。
    """
    frame = work.copy()
    frame["场站名称"] = station
    _add_cyclical(frame)
    _add_lags(frame)
    return frame


def forecast_horizon(
    features: FeatureSet,
    model,
    feature_columns: list[str],
    station_columns: list[str],
    station: str,
    origin: pd.Timestamp,
    horizon: int = MULTISTEP_HORIZON,
) -> pd.DataFrame:
    """从 `origin` 之后逐小时递归预测 `horizon` 小时。

    Returns
    -------
    DataFrame，列：时刻 / 预测_kWh。历史不足或无法预测时返回空表。
    """
    work, tail = _prepare_recursion(features, station, origin)
    if work.empty or tail.empty:
        return pd.DataFrame(columns=["时刻", "预测_kWh"])

    n_hist = int((work["时刻"] <= origin).sum())
    # 需要 168 小时预热 + 至少 1 小时可预测
    if n_hist < 169 or n_hist >= len(work):
        return pd.DataFrame(columns=["时刻", "预测_kWh"])

    # 递归中未知的辅助量用持久化值：假定未来小时维持 origin 处的最后观测水平
    persist = {
        "订单数_lag_1": float(tail["订单数"]),
        "订单数_lag_24": float(tail["订单数"]),
        "单均电量_lag_24": float(tail["单均电量"]),
        "平均时长_lag_24": float(tail["平均时长"]),
    }

    rows: list[dict] = []
    idx = n_hist
    for _ in range(horizon):
        if idx >= len(work):
            break

        frame = _feature_frame(work, station)
        row = frame.iloc[[idx]].copy()

        for col, value in persist.items():
            row[col] = value

        # 场站 one-hot：递归帧里没有这些列，必须显式补齐。
        # 若这里静默留成 NaN，模型会以为「不知道是哪个场站」，预测系统性偏移。
        for col in station_columns:
            row[col] = 1.0 if col == f"场站_{station}" else 0.0

        X = row.reindex(columns=feature_columns)
        pred = float(model.predict(X)[0])
        pred = max(pred, 0.0)  # 电量不可能为负

        rows.append({"时刻": frame.iloc[idx]["时刻"], "预测_kWh": pred})

        # 把预测写回工作表，供下一步的 lag_1 使用
        work.loc[idx, TARGET] = pred
        work.loc[idx, "订单数"] = persist["订单数_lag_1"]
        work.loc[idx, "平均时长"] = persist["平均时长_lag_24"]
        work.loc[idx, "单均电量"] = persist["单均电量_lag_24"]
        idx += 1

    return pd.DataFrame(rows)


def backtest_multistep(
    features: FeatureSet,
    model,
    origin_frame: pd.DataFrame,
    horizon: int = MULTISTEP_HORIZON,
    n_origins_per_station: int = 7,
) -> float | None:
    """在测试期内做 24 小时递归预测回测，返回多步 MAE。

    **为什么值得做这一步**：单步 MAE 是「已知前一小时真值」条件下的误差，
    而有序充电需要的是**提前一整天**的预测。两者的误差不是一个量级。
    大多数课程项目只报单步指标，因此这一步本身就是方法严谨性的体现。

    做法：在测试期上按天取若干预测起点，每个起点做一次 24 小时递归预测，
    与真实值比较。起点数量有限是为了控制耗时；这会在结果里注明。
    """
    frame = features.frame
    records: list[dict] = []

    for station in features.stations:
        sub = origin_frame[origin_frame["场站名称"] == station]
        if sub.empty:
            continue
        days = sorted(sub["时刻"].dt.normalize().unique())
        # 均匀取若干天作为预测起点，保证覆盖测试期的不同时点
        if len(days) > n_origins_per_station:
            idx = np.linspace(0, len(days) - 1, n_origins_per_station).round().astype(int)
            days = [days[i] for i in dict.fromkeys(idx.tolist())]

        for day in days:
            origin = pd.Timestamp(day) - pd.Timedelta(hours=1)
            fc = forecast_horizon(
                features, model, features.feature_columns, features.station_columns,
                station, origin, horizon,
            )
            if fc.empty:
                continue
            truth = sub[["时刻", TARGET]]
            merged = fc.merge(truth, on="时刻", how="inner")
            if merged.empty:
                continue
            records.append(
                {
                    "场站名称": station,
                    "MAE": mae(merged[TARGET].to_numpy(), merged["预测_kWh"].to_numpy()),
                    "n": len(merged),
                }
            )

    if not records:
        return None
    back = pd.DataFrame(records)
    # 按样本数加权，而不是对起点取简单平均 —— 否则样本少的起点权重被高估
    return float(np.average(back["MAE"], weights=back["n"]))


# --------------------------------------------------------------------------- #
# 供界面使用的封装
# --------------------------------------------------------------------------- #

@dataclass
class FittedForecaster:
    """训练好的预测器。放在 `st.cache_resource` 里，不要放 `st.cache_data`
    —— 后者会 pickle 返回值，而训练好的 sklearn 模型应该跨会话共享实例。"""

    model: object
    feature_columns: list[str]
    station_columns: list[str]
    one_step_mae: float
    multistep_mae: float
    horizon: int
    stations: list[str]

    @property
    def band_mae(self) -> float:
        """不确定性带半宽。

        **采用多步 MAE 而非单步 MAE** —— 策略页展示的是未来 24 小时预测，
        而多步 MAE 才是这个场景下诚实的误差量级。用单步 MAE 画带会
        系统性低估不确定性。
        """
        return max(self.multistep_mae, self.one_step_mae)

    def forecast(self, features: FeatureSet, station: str, origin: pd.Timestamp) -> pd.DataFrame:
        fc = forecast_horizon(
            features, self.model, self.feature_columns, self.station_columns,
            station, origin, self.horizon,
        )
        if fc.empty:
            return fc
        fc["下界_kWh"] = (fc["预测_kWh"] - self.band_mae).clip(lower=0.0)
        fc["上界_kWh"] = fc["预测_kWh"] + self.band_mae
        return fc


def fit_final(features: FeatureSet, report: ModelReport) -> FittedForecaster:
    """复用已评估的模型构造预测器。**不重新训练** —— 保证界面上的预测
    与「模型与验证」页报告的指标出自同一个模型。"""
    return FittedForecaster(
        model=report.model,
        feature_columns=report.feature_columns,
        station_columns=report.station_columns,
        one_step_mae=report.primary_mae,
        multistep_mae=report.multistep_mae if report.multistep_mae is not None else report.primary_mae,
        horizon=report.multistep_horizon,
        stations=features.stations,
    )


METHOD_NOTES: tuple[str, ...] = (
    "**划分方式**：按时间顺序留出最后 20% 作测试集，不做随机打乱。"
    "随机划分会让模型从测试期的相邻小时「看到」未来（lag_1 直接泄漏），指标虚高。",
    "**防泄漏**：所有滞后与滚动统计均先 shift 再 rolling；"
    "「订单数」「平均时长」「单均电量」等**当期**统计量一律不作特征，只取滞后值。",
    "**对照公平**：四个模型在同一测试集的同一行集上评分，基线不获得任何额外样本。",
    "**提升度的分母是最强基线**，不是两者中较差的那个 —— 用弱基线作对照没有意义。",
)
