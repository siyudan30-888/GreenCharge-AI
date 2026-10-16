"""有序充电策略：从模型预测到可执行建议。

## 这一层在修什么

原实现里写着「预计可转移 **{st_peak * 0.25:.1f}%** 的高峰负荷」——
那个 `0.25` 是凭空写下的常数，没有任何来源。一个编造的数字足以毁掉
整个项目的可信度，因为评委只需要问一句「这个 0.25 是怎么来的」。

本模块用下式替代它：

    可转移电量 = min( E_峰 × θ ,  H_谷 )

- `E_峰` —— 峰时段总电量，**来自数据**；
- `θ` —— 可延展比例，**用户滑块，显式标注为假设**，
  含义是「峰时段内停驻时长足以延后至谷时段充电的订单电量占比」；
- `H_谷` —— 谷时可用裕度 = 额定容量 × 谷时小时数 − 谷时实际电量，**由数据导出**。

这个式子**不可能**是任意的：它是「数据导出的物理容量」与「已披露的可调假设」
取小。而且它会产生真实的运营发现 —— 当 `H_谷` 成为约束时，界面会明确说出
「受谷时可用裕度限制」，这比一个魔法常数有意思得多。

## 不做什么

**不建模用户响应行为。** 我们不会声称「1.5 倍积分会使 30% 的用户转移充电」——
那需要真实的用户行为数据，而我们没有。本模块提供的是**给定转移比例下的
敏感性分析**：每一行都是显式公式，把「不知道的部分」作为自变量交给用户。
承认自己不知道什么的敏感性表，远比一个自信的百分比可信。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import CarbonParams, StationLoadMetrics, carbon_accounting
from .schema import DEFAULT_TOU, PEAK, VALLEY, TouConfig


# --------------------------------------------------------------------------- #
# 假设：全部可调项集中一处，既驱动计算也驱动界面渲染
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Assumptions:
    """本页全部可调假设。

    **同一个对象既驱动计算、又驱动「假设与参数」面板的渲染**，
    因此两者不可能漂移 —— 面板上显示的永远是实际参与计算的值。
    """

    theta: float = 0.30
    """可延展比例 θ：峰时段内停驻时长足以延后至谷时段充电的订单电量占比。
    **这是可调假设，不是实测值。** 默认 0.30 是一个保守起点。"""

    service_fee_base: float = 0.40
    """基础充电服务费（元/kWh）。**设计选择**，不是观测值。"""

    service_fee_peak_uplift: float = 0.15
    """峰时服务费上浮（元/kWh）。**设计选择** —— 这是策略建议，不是预测。"""

    points_per_kwh: float = 1.0
    """每 kWh 充电发放的基础碳积分（积分/kWh）。**设计选择**。"""

    valley_points_multiplier: float = 1.5
    """谷时段积分倍数。**设计选择** —— 作为可调项呈现，而非作为预测。"""

    points_value_yuan: float = 0.10
    """每积分可抵扣价值（元/积分）。**设计选择**，取值决定激励强度。"""

    demand_charge_yuan_per_kw_month: float = 40.0
    """需量电价（元/kW/月），两部制电价的「基本电价」部分。
    **量级参考，须按当地现行电价文件核对。**"""

    assume_peak_sets_monthly_demand: bool = True
    """是否假定当日峰值即为当月最大需量。

    **这是一个强假设，必须明示**：需量电费按月最大需量计收，
    降低某一天的峰值只有在「该日峰值就是当月最大值」时才真正省钱。"""

    def rows(self) -> list[tuple[str, str, str, str]]:
        """渲染成假设面板：(参数, 取值, 单位, 来源/理由)。"""
        return [
            ("可延展比例 θ", f"{self.theta:.2f}", "无量纲",
             "**可调假设，非实测值**：峰时段内停驻时长足以延后至谷时段充电的订单电量占比。"),
            ("基础充电服务费", f"{self.service_fee_base:.2f}", "元/kWh", "**设计选择**。"),
            ("峰时服务费上浮", f"{self.service_fee_peak_uplift:.2f}", "元/kWh",
             "**设计选择** —— 这是策略建议，不是对用户行为的预测。"),
            ("每 kWh 基础积分", f"{self.points_per_kwh:.2f}", "积分/kWh", "**设计选择**。"),
            ("谷时积分倍数", f"{self.valley_points_multiplier:.2f}", "倍",
             "**设计选择** —— 作为可调项呈现，而非作为预测。"),
            ("积分抵扣价值", f"{self.points_value_yuan:.2f}", "元/积分", "**设计选择**，决定激励强度。"),
            ("需量电价", f"{self.demand_charge_yuan_per_kw_month:.0f}", "元/kW/月",
             "两部制电价的「基本电价」部分。**量级参考，须按当地现行电价文件核对。**"),
            ("假定当日峰值即当月最大需量",
             "是" if self.assume_peak_sets_monthly_demand else "否", "—",
             "**强假设**：需量电费按月最大需量计收，降低单日峰值只有在"
             "「该日峰值就是当月最大值」时才真正省钱。"),
        ]


#: 敏感性表明确不含行为模型，这句话必须随表一起出现
SENSITIVITY_DISCLAIMER = (
    "本表为**给定转移比例下的敏感性分析，不含用户响应的行为模型**。"
    "「需要多大激励才能让用户真的转移这么多」需要真实用户行为数据才能回答，"
    "本平台没有该数据，因此不给出此类预测。"
)


# --------------------------------------------------------------------------- #
# 可转移电量
# --------------------------------------------------------------------------- #

@dataclass
class ShiftPlan:
    """一个场站的削峰填谷方案。每个数字要么来自数据，要么来自已披露的假设。"""

    station: str
    peak_energy_kwh: float
    """峰时**日均**电量（kWh/日）。"""
    valley_energy_kwh: float
    """谷时**日均**电量（kWh/日）。"""
    valley_margin_kwh: float | None
    """谷时可用裕度 = 额定容量 × 谷时小时数 − 谷时日均电量（kWh/日）。
    未知容量时为 None。"""
    theta: float
    candidate_kwh: float
    """按 θ 计算的可延展电量 = E_峰(日均) × θ（kWh/日）"""
    shiftable_kwh: float
    """最终可转移电量 = min(candidate, 谷时裕度)（kWh/日）"""
    binding_constraint: str
    """「θ」表示假设是限制因素；「谷时可用裕度」表示物理容量是限制因素。"""
    peak_reduction_kw: float
    """削峰量 = 可转移电量(kWh/日) / 峰时小时数 —— 量纲为 kW，因为分母是「小时/日」。"""
    peak_window_hours: int
    valley_window_hours: int


def valley_margin(
    capacity_kw: float | None,
    tou: TouConfig,
    valley_energy_daily_kwh: float,
) -> float | None:
    """谷时可用裕度 H_谷 = 额定容量 × 谷时小时数 − 谷时**日均**实际电量。

    **这是本模块取代那个凭空常数 0.25 的关键。** 它是物理容量，由数据导出，
    不是猜的。

    ⚠️ **量纲警告：入参必须是日均电量，不是期间累计电量。**
    `capacity_kw × valley_hours` 是**单日**可充电量（kWh/日）。若拿 28 天的
    累计谷时电量去减，结果会荒谬地偏大或偏小，而且不会报错 —— 只会让
    敏感性表每一行数字都相同。调用方须先除以天数。

    简化说明：这里假定谷时各小时同等可用。若谷时本身已接近满负荷
    （例如枢纽整夜都有营运车辆充电），实际裕度会小于此值 ——
    界面上会把这个约束明确说出来。
    """
    if capacity_kw is None or capacity_kw <= 0:
        return None
    return max(capacity_kw * tou.valley_hours - valley_energy_daily_kwh, 0.0)


def plan_shift(
    station: str,
    metrics: StationLoadMetrics,
    assumptions: Assumptions,
    tou: TouConfig = DEFAULT_TOU,
    *,
    capacity_kw: float | None = None,
    peak_energy_kwh: float | None = None,
    valley_energy_kwh: float | None = None,
) -> ShiftPlan:
    """计算某场站的削峰填谷方案。

    **本函数内部一律按「日」计算。** 传入的峰谷电量是期间累计值，先除以
    `metrics.n_days` 折算成日均，再与单日容量比较；否则量纲不一致。
    返回的 ShiftPlan 中所有电量字段均为**日均值（kWh/日）**。
    """
    n_days = max(float(metrics.n_days), 1.0)

    peak_e = (
        peak_energy_kwh
        if peak_energy_kwh is not None
        else metrics.total_kwh * metrics.peak_energy_pct / 100.0
    ) / n_days
    valley_e = (
        valley_energy_kwh
        if valley_energy_kwh is not None
        else metrics.total_kwh * metrics.valley_energy_pct / 100.0
    ) / n_days

    margin = valley_margin(capacity_kw, tou, valley_e)
    candidate = peak_e * assumptions.theta

    if margin is None:
        shiftable = candidate
        binding = "θ（未提供额定容量，无法判断谷时裕度）"
    elif margin < candidate:
        shiftable = margin
        binding = "谷时可用裕度"
    else:
        shiftable = candidate
        binding = "θ"

    peak_reduction = shiftable / tou.peak_hours if tou.peak_hours else 0.0

    return ShiftPlan(
        station=station,
        peak_energy_kwh=peak_e,
        valley_energy_kwh=valley_e,
        valley_margin_kwh=margin,
        theta=assumptions.theta,
        candidate_kwh=candidate,
        shiftable_kwh=shiftable,
        binding_constraint=binding,
        peak_reduction_kw=peak_reduction,
        peak_window_hours=tou.peak_hours,
        valley_window_hours=tou.valley_hours,
    )


# --------------------------------------------------------------------------- #
# 敏感性分析
# --------------------------------------------------------------------------- #

#: 敏感性扫描的转移比例网格
THETA_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def sensitivity_table(
    station: str,
    metrics: StationLoadMetrics,
    assumptions: Assumptions,
    carbon: CarbonParams,
    tou: TouConfig = DEFAULT_TOU,
    *,
    capacity_kw: float | None = None,
    theta_grid: tuple[float, ...] = THETA_GRID,
) -> pd.DataFrame:
    """给定转移比例下的敏感性表。

    每一格都是显式公式的结果，没有一个是编出来的：

    | 列 | 公式 |
    | --- | --- |
    | 转移电量 | `min(E_峰 × θ, H_谷)`，**日均** |
    | 削峰量 | `转移电量 / 峰时小时数` |
    | 碳减排 | `转移电量 × (峰时边际因子 − 谷时边际因子)`，**日均** |
    | 用户节省 | `转移电量 × (峰时服务费上浮 + 额外积分价值)`，**日均** |
    | 运营商服务费变化 | `转移电量 × (谷时费率 − 峰时费率)`，**日均** |
    | 月度需量电费节省 | `削峰量 × 需量电价`（仅当假定当日峰值即当月最大需量） |

    **全表按「日」为口径**：峰谷电量先折算成日均，再与单日容量比较。
    否则 28 天的累计电量与单日容量直接相除，会让每一行得出相同数字。
    """
    n_days = max(float(metrics.n_days), 1.0)
    peak_e = metrics.total_kwh * metrics.peak_energy_pct / 100.0 / n_days
    valley_e = metrics.total_kwh * metrics.valley_energy_pct / 100.0 / n_days
    margin = valley_margin(capacity_kw, tou, valley_e)

    valley_rate = assumptions.service_fee_base
    peak_rate = assumptions.service_fee_base + assumptions.service_fee_peak_uplift
    extra_points_value = (
        (assumptions.valley_points_multiplier - 1.0)
        * assumptions.points_per_kwh
        * assumptions.points_value_yuan
    )

    rows = []
    for theta in theta_grid:
        candidate = peak_e * theta
        if margin is None:
            shifted = candidate
            constrained = ""
        elif margin < candidate:
            shifted = margin
            constrained = "超出谷时可接纳能力"
        else:
            shifted = candidate
            constrained = ""

        peak_reduction = shifted / tou.peak_hours if tou.peak_hours else 0.0
        carbon_result = carbon_accounting(0.0, carbon, shifted_kwh=shifted)
        user_saving = shifted * (assumptions.service_fee_peak_uplift + extra_points_value)
        operator_delta = shifted * (valley_rate - peak_rate)
        demand_saving = (
            peak_reduction * assumptions.demand_charge_yuan_per_kw_month
            if assumptions.assume_peak_sets_monthly_demand
            else float("nan")
        )

        rows.append(
            {
                "转移比例 θ": theta,
                "转移电量 (kWh/日)": round(shifted, 1),
                "削峰量 (kW)": round(peak_reduction, 1),
                "碳减排 (kg CO₂/日)": round(carbon_result.shift_saving_kg, 1),
                "用户节省 (元/日)": round(user_saving, 1),
                "运营商服务费变化 (元/日)": round(operator_delta, 1),
                "月度需量电费节省 (元/月)": (
                    round(demand_saving, 0) if np.isfinite(demand_saving) else "—"
                ),
                "备注": constrained,
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 文字建议：必须随场站变化
# --------------------------------------------------------------------------- #

def weekend_ratio(df: pd.DataFrame) -> float | None:
    """周末日均电量 / 工作日日均电量。

    这是**由数据导出**的形态特征，用来判断「工作日策略」与「周末策略」
    是否需要分开设计。园区约为 0.2，商场大于 1，枢纽接近 1。
    """
    if df.empty or "是否周末" not in df.columns:
        return None

    daily = (
        df.groupby(["日期", "是否周末"], observed=True)["充电量_kWh"].sum().reset_index()
    )
    means = daily.groupby("是否周末", observed=True)["充电量_kWh"].mean()
    if True not in means.index or False not in means.index:
        return None
    weekday_mean = float(means.loc[False])
    if weekday_mean <= 0:
        return None
    return float(means.loc[True]) / weekday_mean


def narrative(
    plan: ShiftPlan,
    metrics: StationLoadMetrics,
    assumptions: Assumptions,
    weekend_ratio_value: float | None,
    has_prediction: bool,
    band_mae: float,
) -> list[str]:
    """生成该场站的诊断与建议。

    **每一条都引用一个实际计算出来的数字**，并且**随选中场站变化** ——
    这正是「演示」与「模型」的分界。原来的实现在这一点上是坏的：
    选任意场站都产出同一段写死的文字。
    """
    out: list[str] = []

    # ---- 1. 峰时负荷压力：由峰时电量占比决定措辞 -------------------------- #
    peak_pct = metrics.peak_energy_pct
    if peak_pct >= 45:
        out.append(
            f"**峰时负荷压力高**：峰时段（{plan.peak_window_hours} 小时）电量占比 "
            f"**{peak_pct:.1f}%**，明显高于按小时数平均的 "
            f"{plan.peak_window_hours / 24 * 100:.0f}%，削峰收益最大。"
        )
    elif peak_pct >= 30:
        out.append(
            f"**峰时负荷压力中等**：峰时段电量占比 **{peak_pct:.1f}%**，"
            f"略高于按小时数平均的 {plan.peak_window_hours / 24 * 100:.0f}%。"
        )
    else:
        out.append(
            f"**峰时负荷压力低**：峰时段电量占比仅 **{peak_pct:.1f}%**，"
            f"低于按小时数平均的 {plan.peak_window_hours / 24 * 100:.0f}%，"
            "该场站的削峰需求不迫切，重点应放在提升谷时利用率。"
        )

    # ---- 2. 约束来源：θ 还是物理容量 --------------------------------------- #
    if plan.valley_margin_kwh is None:
        out.append(
            "**无法判断谷时可接纳能力**：未提供该场站的额定容量，"
            f"因此可转移电量按 θ 单独计算（{plan.shiftable_kwh:,.0f} kWh）。"
            "填写额定容量后可得到受物理约束的真实上限。"
        )
    elif plan.binding_constraint == "谷时可用裕度":
        out.append(
            f"**受谷时可用裕度限制**：按 θ={plan.theta:.2f} 本可转移 "
            f"{plan.candidate_kwh:,.0f} kWh，但谷时段仅有 "
            f"**{plan.valley_margin_kwh:,.0f} kWh** 的可用裕度"
            f"（额定容量 × {plan.valley_window_hours} 谷时小时 − 谷时实际电量），"
            f"因此实际可转移 **{plan.shiftable_kwh:,.0f} kWh**。"
            "**这说明瓶颈不在用户意愿，而在谷时段本身已经接近满载** —— "
            "继续加码激励不会有效，应考虑扩容或把转移目标改到平时段。"
        )
    else:
        out.append(
            f"**受假设 θ 限制**：谷时段可用裕度 {plan.valley_margin_kwh:,.0f} kWh，"
            f"大于按 θ={plan.theta:.2f} 计算的可延展电量 {plan.candidate_kwh:,.0f} kWh，"
            f"因此实际可转移 **{plan.shiftable_kwh:,.0f} kWh**，"
            f"对应削峰 **{plan.peak_reduction_kw:,.1f} kW**。"
            "提高 θ 仍有余量，瓶颈在用户参与意愿而非场站容量。"
        )

    # ---- 3. 工作日 / 周末：由数据导出的形态差异 ---------------------------- #
    if weekend_ratio_value is not None:
        if weekend_ratio_value < 0.5:
            out.append(
                f"**周末负荷塌陷**：周末日均电量仅为工作日的 "
                f"**{weekend_ratio_value:.2f} 倍**。该场站的峰谷问题本质上是"
                "**工作日问题**，策略应只在工作日投放；周末可关闭峰时上浮，"
                "避免无谓地影响少数用户。"
            )
        elif weekend_ratio_value > 1.15:
            out.append(
                f"**周末负荷更高**：周末日均电量为工作日的 "
                f"**{weekend_ratio_value:.2f} 倍**。峰时上浮与积分激励需要"
                "**周末加强、工作日减弱**，与常规「削峰填谷」的默认配置相反。"
            )
        else:
            out.append(
                f"**工作日与周末负荷接近**：比值为 **{weekend_ratio_value:.2f}**，"
                "无需为两者设计差异化策略。"
            )

    # ---- 4. 负荷平坦度 ----------------------------------------------------- #
    out.append(
        f"**负荷平坦度**：负荷率 {metrics.load_factor:.3f}（平均 {metrics.mean_kw:,.1f} kW "
        f"/ 峰值 {metrics.peak_kw:,.1f} kW），峰谷差率 {metrics.peak_valley_ratio_pct:.1f}%，"
        f"波动率 {metrics.volatility:.2f}。"
        + (
            "负荷率偏低说明资产在大部分时间闲置，削峰填谷的空间大。"
            if metrics.load_factor < 0.35
            else "负荷率尚可，进一步提升需依靠谷时段电量增长而非峰值削减。"
        )
    )

    # ---- 5. 不确定性：绝不省略 --------------------------------------------- #
    if has_prediction:
        out.append(
            f"**关于预测精度**：上方的 24 小时预测曲线带有 ±{band_mae:,.1f} kWh 的"
            "不确定性带，该带宽取自**多步（24 小时递归）回测 MAE**，"
            "而不是单步 MAE —— 单步 MAE 假定已知前一小时真值，会系统性低估"
            "提前一天的预测误差。带宽覆盖的是平均误差量级，不是置信区间。"
        )
    else:
        out.append(
            "**未提供预测**：当前数据不足以完成 168 小时预热，"
            "因此本页只展示描述性指标，不展示预测曲线。"
        )

    # ---- 6. 不做的部分 ----------------------------------------------------- #
    out.append(
        "**本建议不包含用户行为模型**：上述可转移电量是在给定 θ 下的"
        "**技术可行性上限**，不是对用户实际响应比例的预测。"
        "「需要多大激励才能让用户真的转移这么多」需要真实用户行为数据才能回答。"
    )

    return out
