"""电力负荷指标与碳核算。

本模块的每一个数字都有具名函数、明确公式和单位 —— 这是整个项目的答辩防线：
屏幕上出现的任何数值，都能被追溯到这里的某一行。

## 负荷曲线的构造（重要）

订单表里只有「起始时刻 + 时长 + 电量」。直接按**起始小时**汇总会得到一个
与实际不符的曲线：园区 3.5 小时的订单会把全部电量压在起始那一小时，
而它实际上横跨了后面几个小时。

因此 `hourly_load()` 把每单电量**按其持续时间摊到所跨越的每个小时**上，
再按「场站 × 时刻」汇总。因为桶宽恰好是 1 小时，桶内 kWh 数值上就等于
该小时的平均功率 kW —— 这个等价关系让后续指标可以直接引用功率单位。

若数据缺少 `充电时长_分钟`，则退化为按起始小时归属，并在结果里
以 `spread=False` 标记出来，由界面明示。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import DEFAULT_TOU, FLAT, PEAK, VALLEY, TouConfig

# --------------------------------------------------------------------------- #
# 负荷曲线
# --------------------------------------------------------------------------- #

@dataclass
class LoadCurve:
    """逐小时的场站负荷。单位：kWh（数值上等于该小时的平均功率 kW）。"""

    frame: pd.DataFrame
    """列：场站名称 / 时刻 / 电量_kWh / 功率_kW"""
    spread: bool
    """True 表示电量已按时长摊分；False 表示数据缺时长，按起始小时归属"""

    caveat: str


def hourly_load(df: pd.DataFrame) -> LoadCurve:
    """把订单明细转成逐小时负荷曲线，单位 kWh（= 该小时平均功率 kW）。

    摊分算法：一单从起始小时的第 `offset` 分钟开始、持续 `dur` 分钟，
    它对第 k 个小时的贡献是它与该小时的**重叠时长**占 `dur` 的比例。

        overlap_k = clip(offset + dur − 60k, 0, 60) − clip(offset − 60k, 0, 60)

    这是把「订单级数据」转成「负荷曲线」的标准做法，也是本模块唯一有技巧的部分。
    """
    if df.empty:
        raise ValueError("数据为空，无法计算负荷曲线。")

    codes, stations = pd.factorize(df["场站名称"])
    energy = df["充电量_kWh"].to_numpy(dtype=float)
    start = df["充电开始时间"].to_numpy()

    if "充电时长_分钟" in df.columns and df["充电时长_分钟"].notna().all():
        duration = df["充电时长_分钟"].to_numpy(dtype=float)
        spread = True
        caveat = ""
    else:
        # 缺时长：退化为「整单计入起始小时」。这会高估峰、低估相邻小时，必须明示。
        duration = np.full(len(df), 1.0)
        spread = False
        caveat = (
            "数据缺少「充电时长_分钟」，负荷按**起始小时**归属而非按时长摊分，"
            "因此曲线比实际更尖、相邻小时偏低。"
        )

    # 每条订单落在哪个整点，以及从该整点起算的分钟偏移
    hour_start = start.astype("datetime64[h]")
    offset_min = (start - hour_start) / np.timedelta64(1, "m")
    n_hours = np.maximum(np.ceil((offset_min + duration) / 60.0), 1.0).astype(int)
    max_hours = int(n_hours.max())

    base_index = hour_start.astype("datetime64[h]").astype("int64")

    chunks_station: list[np.ndarray] = []
    chunks_hour: list[np.ndarray] = []
    chunks_energy: list[np.ndarray] = []

    for k in range(max_hours):
        active = n_hours > k
        if not active.any():
            continue
        lo = np.clip(offset_min + duration - 60.0 * k, 0.0, 60.0)
        hi = np.clip(offset_min - 60.0 * k, 0.0, 60.0)
        overlap = np.clip(lo - hi, 0.0, None)

        keep = active & (overlap > 0)
        if not keep.any():
            continue

        chunks_station.append(codes[keep])
        chunks_hour.append(base_index[keep] + k)
        chunks_energy.append(energy[keep] * overlap[keep] / duration[keep])

    if not chunks_station:
        raise ValueError("无法构造负荷曲线：所有订单的持续时长均为零。")

    long = pd.DataFrame(
        {
            "场站编码": np.concatenate(chunks_station),
            "小时序号": np.concatenate(chunks_hour),
            "电量_kWh": np.concatenate(chunks_energy),
        }
    )

    grouped = long.groupby(["场站编码", "小时序号"], as_index=False)["电量_kWh"].sum()
    grouped["场站名称"] = np.asarray(stations)[grouped["场站编码"].to_numpy()]
    # 桶宽恰好 1 小时，故 kWh/1h 的数值就是平均功率 kW
    grouped["功率_kW"] = grouped["电量_kWh"]
    # 「1970-01-01 起算的小时数」还原为时间戳。
    # 刻意走 to_datetime(unit="s") 而不是拼 datetime64[h]：
    # pandas 只支持 datetime64 的 s/ms/us/ns 四种单位，
    # 直接塞一个 datetime64[h] 数组进去会得到不可预期的 dtype。
    grouped["时刻"] = pd.to_datetime(
        grouped["小时序号"].to_numpy(dtype="int64") * 3600, unit="s"
    )
    grouped = grouped[["场站名称", "时刻", "电量_kWh", "功率_kW"]].sort_values(
        ["场站名称", "时刻"], kind="stable"
    ).reset_index(drop=True)

    return LoadCurve(frame=grouped, spread=spread, caveat=caveat)


def daily_profile(curve: LoadCurve) -> pd.DataFrame:
    """把逐小时负荷折叠成「场站 × 小时(0–23)」的平均日曲线。

    这是「多场站负荷特征分析」的骨干：三条形状不同的曲线，一眼可辨。
    """
    frame = curve.frame.copy()
    frame["小时"] = frame["时刻"].dt.hour
    out = (
        frame.groupby(["场站名称", "小时"], as_index=False)["功率_kW"]
        .mean()
        .rename(columns={"功率_kW": "平均功率_kW"})
    )
    return out.sort_values(["场站名称", "小时"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 单场站负荷指标
# --------------------------------------------------------------------------- #

@dataclass
class StationLoadMetrics:
    """单个场站的负荷指标。**每个字段都带单位或明确说明。**"""

    station: str
    n_sessions: int
    total_kwh: float
    peak_kw: float
    valley_kw: float
    mean_kw: float

    n_days: float
    """统计覆盖的自然日天数。**所有「日均」换算都必须经过它。**

    谷时可用裕度与谷电利用率的分母是「额定容量 × 谷时小时数」，即**单日**
    可充电量；分子若直接用期间累计电量，结果会高估 n_days 倍（本项目 28 天
    即 28 倍）。这是一个不会报错、只会给出荒谬数字的错误，故显式记录在这里。"""

    load_factor: float
    """: 负荷率 = 平均功率 / 峰值功率。0–1，越高表示曲线越平坦、资产利用越充分。
    ⚠️ 本处的峰值取**实测小时峰值**，不是合同容量。两者会给出不同的负荷率。"""

    peak_valley_diff_kw: float
    """: 峰谷差 = 实测峰值 − 实测谷值"""

    peak_valley_ratio_pct: float
    """: 峰谷差率 = 峰谷差 / 实测峰值 × 100%。
    **另一版本以额定容量为分母，本项目明确采用实测峰值为分母。**"""

    volatility: float
    """: 负荷波动率 = 标准差 / 均值（变异系数）。无量纲，越大越尖。"""

    peak_energy_pct: float
    """: 峰时电量占比 = 峰时电量 / 总电量 × 100%"""

    valley_energy_pct: float
    """: 谷时电量占比 = 谷时电量 / 总电量 × 100%"""

    valley_utilization_pct: float | None
    """: 谷电利用率(b) = 谷时电量 / (额定容量 × 谷时小时数) × 100%。
    需要场站额定容量；未知时为 None。"""

    charger_utilization_pct: float | None
    """: 桩利用率 = Σ充电时长 / (桩数 × 统计时长) × 100%。
    需要桩数（由容量与单桩功率推出）与「充电时长_分钟」；未知时为 None。"""


def _station_energy_by_period(df: pd.DataFrame, tou: TouConfig) -> dict[str, float]:
    total = float(df["充电量_kWh"].sum())
    out = {"total": total, PEAK: 0.0, FLAT: 0.0, VALLEY: 0.0}
    if total <= 0:
        return out
    by_period = df.groupby("时段类型", observed=True)["充电量_kWh"].sum()
    for key in (PEAK, FLAT, VALLEY):
        if key in by_period.index:
            out[key] = float(by_period.loc[key])
    return out


def _per_station(value: float | dict[str, float] | None, station: str) -> float | None:
    """把「标量或按场站的字典」解析成该场站的取值。

    三个场站的额定容量本来就不同（商场 24 桩 × 60 kW、园区 60 桩 × 7 kW、
    枢纽 16 桩 × 120 kW），只接受一个标量会强迫其中两个用错值。
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(station)
    return float(value)


def station_load_metrics(
    df: pd.DataFrame,
    curve: LoadCurve,
    tou: TouConfig = DEFAULT_TOU,
    *,
    capacity_kw: float | dict[str, float] | None = None,
    charger_power_kw: float | dict[str, float] | None = None,
) -> list[StationLoadMetrics]:
    """计算每个场站的负荷指标。

    Parameters
    ----------
    capacity_kw:
        场站额定容量（kW）。可以是标量，也可以是 `{场站名: 容量}` 的字典。
        为 None 时，依赖容量的两个指标返回 None，
        界面应显示「需要额定容量」而**不是**用某个默认值冒充。
    charger_power_kw:
        单桩额定功率（kW），同样支持标量或字典。用于由容量反推桩数，
        进而计算桩利用率。
    """
    results: list[StationLoadMetrics] = []

    for station, sub in df.groupby("场站名称", observed=True):
        s_curve = curve.frame[curve.frame["场站名称"] == station]
        if s_curve.empty:
            continue

        st_capacity = _per_station(capacity_kw, str(station))
        st_charger_power = _per_station(charger_power_kw, str(station))

        power = s_curve["功率_kW"].to_numpy(dtype=float)
        peak_kw = float(power.max())
        valley_kw = float(power.min())
        mean_kw = float(power.mean())

        load_factor = mean_kw / peak_kw if peak_kw > 0 else float("nan")
        diff = peak_kw - valley_kw
        ratio = diff / peak_kw * 100.0 if peak_kw > 0 else float("nan")
        volatility = float(power.std(ddof=0) / mean_kw) if mean_kw > 0 else float("nan")

        energy = _station_energy_by_period(sub, tou)
        total = energy["total"]
        peak_pct = energy[PEAK] / total * 100.0 if total > 0 else float("nan")
        valley_pct = energy[VALLEY] / total * 100.0 if total > 0 else float("nan")

        # 统计覆盖的自然日天数：按**去重日期**数，而不是跨度/24，
        # 这样数据中间有缺口时不会把分母算大。
        n_days = float(sub["日期"].nunique()) if "日期" in sub.columns else 1.0
        n_days = max(n_days, 1.0)

        valley_util: float | None = None
        if st_capacity and st_capacity > 0 and tou.valley_hours > 0 and n_days > 0:
            # 分子必须折算成**日均**谷时电量，才能与「单日容量」相比。
            valley_util = (
                energy[VALLEY] / n_days / (st_capacity * tou.valley_hours) * 100.0
            )

        charger_util: float | None = None
        if (
            st_capacity
            and st_capacity > 0
            and st_charger_power
            and st_charger_power > 0
            and "充电时长_分钟" in sub.columns
            and sub["充电时长_分钟"].notna().any()
        ):
            n_chargers = st_capacity / st_charger_power
            span_hours = _observed_span_hours(sub)
            if span_hours > 0 and n_chargers > 0:
                busy_hours = float(sub["充电时长_分钟"].sum()) / 60.0
                charger_util = busy_hours / (n_chargers * span_hours) * 100.0

        results.append(
            StationLoadMetrics(
                station=str(station),
                n_sessions=int(len(sub)),
                total_kwh=total,
                peak_kw=peak_kw,
                valley_kw=valley_kw,
                mean_kw=mean_kw,
                n_days=n_days,
                load_factor=load_factor,
                peak_valley_diff_kw=diff,
                peak_valley_ratio_pct=ratio,
                volatility=volatility,
                peak_energy_pct=peak_pct,
                valley_energy_pct=valley_pct,
                valley_utilization_pct=valley_util,
                charger_utilization_pct=charger_util,
            )
        )

    return sorted(results, key=lambda m: m.station)


def _observed_span_hours(df: pd.DataFrame) -> float:
    """数据覆盖的时长（小时）。用于把「总占用时长」折算成利用率分母。"""
    span = df["充电开始时间"].max() - df["充电开始时间"].min()
    return max(span.total_seconds() / 3600.0, 1.0)


# --------------------------------------------------------------------------- #
# 站间关系
# --------------------------------------------------------------------------- #

def inter_station_correlation(curve: LoadCurve) -> pd.DataFrame:
    """各场站逐小时功率曲线的 Pearson 相关系数矩阵。

    直接支撑「多场站对比」：相关性低说明三个场站不能互相替代，
    必须分别建模；相关性高则说明存在共同的区域性驱动因素。
    """
    wide = curve.frame.pivot_table(
        index="时刻", columns="场站名称", values="功率_kW", aggfunc="sum"
    ).fillna(0.0)
    return wide.corr()


# --------------------------------------------------------------------------- #
# 碳核算
# --------------------------------------------------------------------------- #

#: 电网平均 CO₂ 排放因子默认值（kg CO₂/kWh）。
#:
#: ⚠️ **这是量级参考值，不是对现行官方数值的断言。**
#: 该因子由生态环境部逐年发布（全国平均值与省级值分别公布），
#: 取值必须以上述最新官方发布为准，并在任何对外材料中注明所引用的年份与文件名。
GRID_EF_DEFAULT = 0.55

GRID_EF_NOTE = (
    "**量级参考值，须以生态环境部最新发布的电力二氧化碳排放因子为准。** "
    "该因子逐年更新、且分全国平均值与省级值，本项目不宣称此默认值即为现行值。"
)

#: 燃油乘用车基准油耗（L/100km）
ICEV_L_PER_100KM_DEFAULT = 8.0

#: 汽油燃烧的 CO₂ 排放因子（kg CO₂/L）
GASOLINE_KG_CO2_PER_L = 2.31

#: 纯电动车平均电耗（kWh/100km）
EV_KWH_PER_100KM_DEFAULT = 15.0


@dataclass(frozen=True)
class CarbonParams:
    """碳核算参数。**五个物理量各自独立命名、独立溯源** ——

    原实现用单个滑块同时承担了「电网排放因子」这一件事，
    但转移类主张需要的是**边际**因子，与平均因子是不同的物理量。
    混用是碳核算里最常见的错误，因此这里显式拆开。
    """

    grid_ef_avg: float = GRID_EF_DEFAULT
    """电网平均排放因子（kg CO₂/kWh）。用于**存量排放**（充电本身产生多少排放）。"""

    icev_l_per_100km: float = ICEV_L_PER_100KM_DEFAULT
    """燃油车基准油耗（L/100km）。"""

    ev_kwh_per_100km: float = EV_KWH_PER_100KM_DEFAULT
    """电动车平均电耗（kWh/100km）。"""

    marginal_ef_peak: float = 0.75
    """峰时**边际**排放因子（kg CO₂/kWh）。**显式假设**：峰时边际机组为燃气/煤电调峰。
    边际因子难以直接获取，此值为假设量级，不是实测或官方发布值。"""

    marginal_ef_valley: float = 0.35
    """谷时**边际**排放因子（kg CO₂/kWh）。**显式假设**：谷时边际机组为基荷（核电/大水电/高效煤电）。"""

    @property
    def icev_kg_per_km(self) -> float:
        return self.icev_l_per_100km / 100.0 * GASOLINE_KG_CO2_PER_L

    @property
    def ev_kg_per_km(self) -> float:
        return self.ev_kwh_per_100km / 100.0 * self.grid_ef_avg

    def rows(self) -> list[tuple[str, str, str, str]]:
        """渲染到假设面板：(参数, 取值, 单位, 来源/理由)"""
        return [
            ("电网平均排放因子", f"{self.grid_ef_avg:.3f}", "kg CO₂/kWh", GRID_EF_NOTE),
            ("燃油车基准油耗", f"{self.icev_l_per_100km:.1f}", "L/100km",
             "乘用车平均水平的量级参考；由它推导燃油车每公里排放。"),
            ("汽油 CO₂ 因子", f"{GASOLINE_KG_CO2_PER_L:.2f}", "kg CO₂/L",
             "汽油燃烧的化学计量排放因子（IPCC 常用值）。"),
            ("电动车电耗", f"{self.ev_kwh_per_100km:.1f}", "kWh/100km",
             "纯电动乘用车平均水平的量级参考。"),
            ("峰时边际排放因子", f"{self.marginal_ef_peak:.2f}", "kg CO₂/kWh",
             "**假设值**：峰时边际机组为调峰火电。边际因子难以获取，此值不代表官方发布数据。"),
            ("谷时边际排放因子", f"{self.marginal_ef_valley:.2f}", "kg CO₂/kWh",
             "**假设值**：谷时边际机组为基荷。边际因子难以获取，此值不代表官方发布数据。"),
        ]


@dataclass
class CarbonResult:
    """一次碳核算的全部结果。**注意每一项的名称都精确区分了「排放」与「减排」。**"""

    energy_kwh: float
    emission_kg: float
    """: 充电**产生**的排放 = 电量 × 电网平均因子。这是**排放**，不是减排。"""

    ev_km: float
    """: 由电量折算的等效行驶里程 = 电量 / 电耗"""

    icev_emission_kg: float
    """: 同样里程下燃油车会产生的排放"""

    avoided_kg: float
    """: 相对燃油车的**减排量** = 燃油车排放 − 电动车排放。
    **这才是「碳减排」成立的地方** —— 它相对「被替代的燃油车」这一基准线。"""

    shifted_kwh: float
    """: 从峰时段转移到谷时段的电量"""

    shift_saving_kg: float
    """: 转移带来的减排 = 转移电量 × (峰时边际因子 − 谷时边际因子)。
    **必须用边际因子**：把 1 kWh 从 19 点移到 3 点，改变的是边际机组的出力。"""


def carbon_accounting(
    energy_kwh: float,
    params: CarbonParams,
    shifted_kwh: float = 0.0,
) -> CarbonResult:
    """碳核算。

    与原实现的三点关键差异（见 README「碳核算口径」）：

    1. **区分排放与减排。** `电量 × 因子` 是充电**产生**的排放。
       称其为减排只有在相对燃油车基准线时才成立，且必须写明基准线是什么。
    2. **转移类主张用边际因子。** 平均因子适合算存量排放；削峰填谷改变的是
       边际机组出力，必须用边际因子。
    3. **不宣称 CCER。** 本结果为基于公开因子的估算，未经核证，不可用于碳交易。
    """
    energy_kwh = float(energy_kwh)
    emission = energy_kwh * params.grid_ef_avg

    ev_km = energy_kwh / (params.ev_kwh_per_100km / 100.0) if params.ev_kwh_per_100km > 0 else 0.0
    icev_emission = ev_km * params.icev_kg_per_km
    avoided = icev_emission - emission

    shift_saving = float(shifted_kwh) * (params.marginal_ef_peak - params.marginal_ef_valley)

    return CarbonResult(
        energy_kwh=energy_kwh,
        emission_kg=emission,
        ev_km=ev_km,
        icev_emission_kg=icev_emission,
        avoided_kg=avoided,
        shifted_kwh=float(shifted_kwh),
        shift_saving_kg=shift_saving,
    )


CARBON_DISCLAIMER = (
    "本平台的碳核算为**基于公开因子的估算，非经核证减排量（CCER）**，"
    "不可用于碳交易或履约。因子取值与来源见「假设与参数」面板。"
)


# --------------------------------------------------------------------------- #
# 汇总：供界面一次性取用
# --------------------------------------------------------------------------- #

@dataclass
class LoadAnalysis:
    """一次完整的负荷分析结果。界面只消费这个对象，不做任何二次计算。"""

    curve: LoadCurve
    profile: pd.DataFrame
    metrics: list[StationLoadMetrics] = field(default_factory=list)
    correlation: pd.DataFrame | None = None


def analyze_load(
    df: pd.DataFrame,
    tou: TouConfig = DEFAULT_TOU,
    *,
    capacity_kw: float | dict[str, float] | None = None,
    charger_power_kw: float | dict[str, float] | None = None,
) -> LoadAnalysis:
    curve = hourly_load(df)
    return LoadAnalysis(
        curve=curve,
        profile=daily_profile(curve),
        metrics=station_load_metrics(
            df, curve, tou, capacity_kw=capacity_kw, charger_power_kw=charger_power_kw
        ),
        correlation=inter_station_correlation(curve),
    )


# --------------------------------------------------------------------------- #
# 日负荷形态聚类（无监督）
# --------------------------------------------------------------------------- #

@dataclass
class ClusterResult:
    """日负荷形态聚类结果。"""

    k: int
    silhouette: float
    k_scores: pd.DataFrame
    """k 从 2 到 5 的轮廓系数扫描结果 —— **k 是扫出来的，不是因为有 3 个场站就定 3**"""
    labels: pd.DataFrame
    """每行一个「场站 × 日」，含所属簇与簇内占比"""
    profiles: pd.DataFrame
    """各簇的归一化中心形态（24 维），用于画图解释「这簇长什么样」"""


def daily_shape_matrix(curve: LoadCurve) -> tuple[np.ndarray, pd.DataFrame]:
    """构造每个「场站 × 日」的 24 维**归一化**日形态向量。

    **归一化是关键**：把每天的曲线除以它自己的均值，从而**剥离水平、保留形状**。
    否则聚类只会按「哪个场站电量多」来分，得到的簇等价于场站本身，毫无信息量。
    归一化之后，同一个场站的不同日子会按「今天像什么形态」聚到一起。
    """
    frame = curve.frame.copy()
    frame["日期"] = frame["时刻"].dt.normalize()
    frame["小时"] = frame["时刻"].dt.hour

    wide = frame.pivot_table(
        index=["场站名称", "日期"], columns="小时", values="功率_kW", aggfunc="sum"
    )
    wide = wide.reindex(columns=range(24), fill_value=0.0).fillna(0.0)

    scale = wide.mean(axis=1)
    # 全天零负荷的日子没有「形状」可言，剔除而不是拿 0 去凑
    valid = scale > 0
    normalized = wide.loc[valid].div(scale[valid], axis=0)

    index = normalized.index.to_frame(index=False)
    return normalized.to_numpy(dtype=float), index.reset_index(drop=True)


def cluster_daily_shapes(
    curve: LoadCurve,
    k_range: tuple[int, ...] = (2, 3, 4, 5),
    *,
    min_cluster_size: int | None = None,
    tie_tolerance: float = 0.01,
) -> ClusterResult | None:
    """对归一化日形态做 k-means，并用**轮廓系数**选择 k。

    价值：揭示单个场站内部存在**多种运行形态**（例如枢纽有「工作日通勤」
    与「周末休闲」两种日型）。这直接指导有序充电 —— 它告诉你能转移的是**哪些日子**，
    而不只是「总共能转移多少」。同时也让原本空洞的「多场站负荷特征分析」第一次有内容。

    **诚实要求一**：k 由轮廓系数扫描决定，绝不因为「有三个场站」就断言 k=3。

    **诚实要求二（容易漏掉的一条）**：必须识别**由离群点撑起来的解**。
    轮廓系数偏好「一个孤立点 + 其余全部」这种划分 —— 把最难解释的那个点
    单独隔离开，分数反而很高。但那不是一种运行形态，只是几天异常数据。
    因此这里要求至少 `min_cluster_size` 个样本才算一个「实质形态」
    （默认 max(3, 5% 样本量)），并区分两种情况：

    - 实质形态数 ≥ 2：合法解，其中过小的簇在结果里**标记为「离群日」**，
      不再冒充一种形态。轮廓系数在这些解之间比较。
    - 实质形态数 < 2：说明数据里没有可复现的日型，该 k 不参与比较。

    每个 k 的**最小簇规模与实质形态数**都会随结果一并返回，
    让扫描过程可审计，而不是只给一个结论。轮廓系数相差不足 `tie_tolerance`
    时取更小的 k —— 避免把同一形态过度切分。
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    matrix, index = daily_shape_matrix(curve)
    n_samples = matrix.shape[0]

    feasible = [k for k in k_range if 1 < k < n_samples]
    if len(feasible) < 2:
        return None

    if min_cluster_size is None:
        min_cluster_size = max(3, int(round(0.05 * n_samples)))

    scores = []
    for k in feasible:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        sizes = np.bincount(labels, minlength=k)
        n_substantive = int((sizes >= min_cluster_size).sum())
        scores.append(
            {
                "k": k,
                "轮廓系数": float(silhouette_score(matrix, labels)),
                "最小簇规模": int(sizes.min()),
                "实质形态数": n_substantive,
                "可用": "是" if n_substantive >= 2 else "否（不足以构成两种形态）",
            }
        )
    if not scores:
        return None

    k_scores = pd.DataFrame(scores)
    usable = k_scores[k_scores["可用"] == "是"]
    if usable.empty:
        return None

    # 平手（差值在容差内）时取更小的 k，避免过度切分
    top_score = float(usable["轮廓系数"].max())
    near_tie = usable[usable["轮廓系数"] >= top_score - tie_tolerance]
    best_k = int(near_tie["k"].min())

    final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = final.fit_predict(matrix)
    sizes = np.bincount(labels, minlength=best_k)

    # 大到足以称为「形态」的簇依次编号；过小的簇统一标为「离群日」
    substantive = [i for i in range(best_k) if sizes[i] >= min_cluster_size]
    name_of = {c: f"形态{chr(65 + j)}" for j, c in enumerate(substantive)}
    for i in range(best_k):
        name_of.setdefault(i, "离群日")

    labels_frame = index.copy()
    labels_frame["簇"] = [name_of[int(i)] for i in labels]
    counts = labels_frame["簇"].value_counts(normalize=True)
    labels_frame["簇内占比"] = labels_frame["簇"].map(lambda c: round(float(counts[c]), 3))

    profiles = pd.DataFrame(
        final.cluster_centers_, columns=[f"{h}时" for h in range(24)]
    )
    profiles.insert(0, "簇", [name_of[i] for i in range(best_k)])

    return ClusterResult(
        k=best_k,
        silhouette=float(k_scores.loc[k_scores["k"] == best_k, "轮廓系数"].iloc[0]),
        k_scores=k_scores,
        labels=labels_frame,
        profiles=profiles,
    )


def station_shape_labels(clusters: ClusterResult) -> pd.DataFrame:
    """每个场站出现过的日形态及其占比。用来回答「这个场站有哪些典型日型」。"""
    grouped = (
        clusters.labels.groupby(["场站名称", "簇"], observed=True)
        .size()
        .rename("天数")
        .reset_index()
    )
    totals = grouped.groupby("场站名称", observed=True)["天数"].transform("sum")
    grouped["占比"] = (grouped["天数"] / totals).round(3)
    return grouped.sort_values(["场站名称", "天数"], ascending=[True, False]).reset_index(drop=True)


def metrics_table(metrics: list[StationLoadMetrics]) -> pd.DataFrame:
    """把指标渲染成界面表格。None 显示为「—（需额定容量）」，绝不用 0 冒充。"""
    rows = []
    for m in metrics:
        rows.append(
            {
                "场站": m.station,
                "订单数": m.n_sessions,
                "总电量 (kWh)": round(m.total_kwh, 1),
                "峰值功率 (kW)": round(m.peak_kw, 1),
                "谷值功率 (kW)": round(m.valley_kw, 1),
                "平均功率 (kW)": round(m.mean_kw, 1),
                "负荷率": round(m.load_factor, 3),
                "峰谷差 (kW)": round(m.peak_valley_diff_kw, 1),
                "峰谷差率 (%)": round(m.peak_valley_ratio_pct, 1),
                "负荷波动率": round(m.volatility, 3),
                "峰时电量占比 (%)": round(m.peak_energy_pct, 1),
                "谷时电量占比 (%)": round(m.valley_energy_pct, 1),
                "谷电利用率b (%)": _fmt_opt(m.valley_utilization_pct),
                "桩利用率 (%)": _fmt_opt(m.charger_utilization_pct),
            }
        )
    return pd.DataFrame(rows)


def _fmt_opt(value: float | None) -> str | float:
    return "—" if value is None else round(value, 1)
