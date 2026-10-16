"""特征工程：把订单明细聚合成「场站 × 小时」样本，并构造防泄漏特征。

预测任务：**给定第 t 小时之前的全部信息，预测该场站的第 t 小时充电量**。
这是「有序充电」的使能能力 —— 无法预见的负荷无法调度。

## 防泄漏是这一层的全部要点

时序预测里最常见、也最隐蔽的造假，是让第 t 小时的特征里混进 t 或 t 之后的信息。
本模块用两条规则杜绝它：

**规则一：所有滞后与滚动统计都先 `shift` 再 `rolling`。**
    `lag_1` 是 t−1 小时的电量，不是 t 小时的电量。
    滚动窗口在**已平移的序列**上计算，因此窗口内的最大值是 t−1。

**规则二：当期聚合量一律不作特征。**
    `订单数`、`平均时长`、`单均电量` 都是**当期**（第 t 小时）统计量 ——
    它们要到该小时结束才知道。把它们当特征，等于把答案的一部分喂给模型。
    因此这三个量在本模块里只以**滞后值**的形式出现（如 `订单数_lag_1`）。

做错这两条会让指标显著虚高，而且从结果上看不出任何异常。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: 预测目标
TARGET = "电量_kWh"

#: 周期编码列。**为什么不用整数编码**：整数编码会让 23 点与 0 点看起来距离最远，
#: 而实际它们相邻；正弦/余弦编码把「一天」映射成圆，使相邻小时在特征空间里也相邻。
CYCLICAL_FEATURES: tuple[str, ...] = ("hour_sin", "hour_cos", "dow_sin", "dow_cos")

#: 非场站的数值特征
NUMERIC_FEATURES: tuple[str, ...] = (
    "is_weekend",
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_168",
    "roll_mean_3", "roll_mean_24", "roll_std_24",
    "订单数_lag_1", "订单数_lag_24",
    "单均电量_lag_24",
    "平均时长_lag_24",
)

#: 计算滚动统计与滞后时，最少需要的前置小时数（lag_168 是其中最长的）
WARMUP_HOURS = 168


@dataclass
class FeatureSet:
    """构造好的特征矩阵。"""

    frame: pd.DataFrame
    """含 场站名称 / 时刻 / 目标列 / 全部特征列"""
    feature_columns: list[str]
    station_columns: list[str]
    """one-hot 场站列。排列重要性分析时会把它们合并回「场站」一项。"""
    stations: list[str]
    n_dropped_warmup: int
    """因预热不足而丢弃的行数。必须披露 —— 否则「用了多少数据」说不清。"""


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #

def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """把订单明细聚合成「场站 × 时刻」的逐小时样本。

    **关键：必须补全缺失的小时。** 若某场站凌晨 3 点一单都没有，直接 groupby
    会让这一小时从表里消失，于是 `lag_1` 会悄悄跳回两小时之前 ——
    时间轴错位，滞后特征全部失去意义。这里显式重索引到完整小时网格，
    缺失小时的订单数补 0、电量补 0，而平均时长/单均电量保持缺失（无订单则无均值）。
    """
    if df.empty:
        raise ValueError("数据为空，无法聚合。")

    df = df.copy()
    df["时刻"] = df["充电开始时间"].dt.floor("h")

    aggregations: dict[str, tuple[str, str]] = {
        "电量_kWh": (TARGET, "sum"),
        "订单数": ("订单ID", "size") if "订单ID" in df.columns else ("充电量_kWh", "size"),
    }
    if "充电时长_分钟" in df.columns:
        aggregations["平均时长"] = ("充电时长_分钟", "mean")
    aggregations["单均电量"] = ("充电量_kWh", "mean")

    agg = df.groupby(["场站名称", "时刻"], observed=True).agg(**aggregations).reset_index()

    # 重命名（agg 的 kwargs 已经命名，这里只补上可能缺失的列）
    if "平均时长" not in agg.columns:
        agg["平均时长"] = np.nan

    # ---- 补全完整小时网格 -------------------------------------------------- #
    stations = sorted(df["场站名称"].unique().tolist())
    hours = pd.date_range(agg["时刻"].min(), agg["时刻"].max(), freq="h")
    full = pd.MultiIndex.from_product([stations, hours], names=["场站名称", "时刻"])

    agg = (
        agg.set_index(["场站名称", "时刻"])
        .reindex(full)
        .reset_index()
    )
    # 无订单的小时：电量与订单数为真实的 0（确实没充电），
    # 平均时长与单均电量保持缺失（没有订单就没有「平均」可言）
    agg["电量_kWh"] = agg["电量_kWh"].fillna(0.0)
    agg["订单数"] = agg["订单数"].fillna(0).astype("int64")
    agg = agg.sort_values(["场站名称", "时刻"], kind="stable").reset_index(drop=True)

    return agg


# --------------------------------------------------------------------------- #
# 特征
# --------------------------------------------------------------------------- #

def _add_cyclical(agg: pd.DataFrame) -> None:
    hour = agg["时刻"].dt.hour.to_numpy(dtype=float)
    dow = agg["时刻"].dt.dayofweek.to_numpy(dtype=float)
    agg["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    agg["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    agg["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    agg["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    agg["is_weekend"] = (dow >= 5).astype(float)


def _add_lags(agg: pd.DataFrame) -> None:
    """滞后与滚动统计。**一律先 shift 再 rolling** —— 见模块 docstring 规则一。"""
    grp = agg.groupby("场站名称", observed=True)

    for lag in (1, 2, 3, 24, 168):
        agg[f"lag_{lag}"] = grp[TARGET].shift(lag)

    # 滚动窗口作用在 lag_1 上：窗口 [t-1, t-2, t-3]，窗口内不含第 t 小时
    for window, name in ((3, "roll_mean_3"), (24, "roll_mean_24")):
        agg[name] = grp["lag_1"].transform(
            lambda s, w=window: s.rolling(w, min_periods=max(2, w // 2)).mean()
        )
    agg["roll_std_24"] = grp["lag_1"].transform(
        lambda s: s.rolling(24, min_periods=12).std()
    )

    # 当期聚合量的**滞后值**（当期值绝不入模，见 docstring 规则二）
    agg["订单数_lag_1"] = grp["订单数"].shift(1)
    agg["订单数_lag_24"] = grp["订单数"].shift(24)
    agg["单均电量_lag_24"] = grp["单均电量"].shift(24)
    agg["平均时长_lag_24"] = grp["平均时长"].shift(24)


def build_features(df: pd.DataFrame) -> FeatureSet:
    """从订单明细直接构造特征矩阵。

    Returns
    -------
    FeatureSet，其中 `frame` 已丢弃预热期（`lag_168` 为缺失的行）。
    """
    agg = aggregate_hourly(df)

    if len(agg) <= WARMUP_HOURS:
        raise ValueError(
            f"数据只有 {len(agg)} 个「场站 × 小时」样本，不足以完成 "
            f"{WARMUP_HOURS} 小时预热。请提供更长时间跨度的数据"
            f"（建议至少 28 天）。"
        )

    _add_cyclical(agg)
    _add_lags(agg)

    before = len(agg)
    frame = agg[agg["lag_168"].notna()].reset_index(drop=True)
    dropped = before - len(frame)

    if frame.empty:
        raise ValueError(
            "预热后没有剩余样本。请提供更长时间跨度的数据（建议至少 28 天）。"
        )

    # 场站 one-hot。场站是无序类别，**不能用序数编码** ——
    # 序数编码会暗示「C 站比 A 站大」，而这没有任何意义。
    station_dummies = pd.get_dummies(frame["场站名称"], prefix="场站", dtype=float)
    station_columns = list(station_dummies.columns)
    frame = pd.concat([frame, station_dummies], axis=1)

    feature_columns = list(CYCLICAL_FEATURES) + list(NUMERIC_FEATURES) + station_columns

    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise AssertionError(f"特征列缺失，实现有误：{missing}")

    return FeatureSet(
        frame=frame,
        feature_columns=feature_columns,
        station_columns=station_columns,
        stations=sorted(frame["场站名称"].unique().tolist()),
        n_dropped_warmup=dropped,
    )


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """**按时间顺序**留出最后一段作测试集。

    绝不使用 `train_test_split(shuffle=True)`：随机划分会让时序模型从测试期的
    相邻小时「看到」未来（`lag_1` 直接泄漏），指标虚高；按时间顺序划分才是
    真实部署条件 —— 上线时你永远只有过去的数据。

    注意本函数按**全局时间点**切分，而不是每个场站各切 80%：
    三个场站的测试期必须落在同一段日历时间上，否则「测试集」对每个场站
    意味着不同的季节/周次，分场站指标就不可比了。
    """
    # 刻意不用 Series.quantile()：它在 datetime64 上的行为依赖后端实现，
    # 而「排序后按位置取分位点」是等价且确定的做法。
    times = np.sort(frame["时刻"].unique())
    if len(times) < 2:
        raise ValueError("数据的时刻不足两个，无法按时间划分。")
    cut_pos = min(int(len(times) * (1.0 - test_fraction)), len(times) - 1)
    cut = times[cut_pos]

    train = frame[frame["时刻"] < cut].reset_index(drop=True)
    test = frame[frame["时刻"] >= cut].reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError(
            f"按时间划分后训练集 {len(train)} 行、测试集 {len(test)} 行，"
            "数据量不足以完成划分。"
        )
    return train, test


def feature_group(column: str, station_columns: list[str]) -> str:
    """把 one-hot 场站列归并回「场站」，使排列重要性图不出现 3 条重复的柱子。"""
    if column in station_columns:
        return "场站"
    return column
