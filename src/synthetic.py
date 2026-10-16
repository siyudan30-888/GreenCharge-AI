"""合成充电场站数据生成器（SYNTHETIC DATA GENERATOR）

⚠️ 本模块产出的是**模拟数据**，不是真实充电场站数据。

## 为什么需要它

竞赛现场（2026-10-31）才会提供三个真实场站的充电数据，赛前拿不到。
本模块用固定随机种子生成形态可信的三场站数据，使整条「分析 → 建模 → 策略」
链路可以被开发、验证和演示。接入真实数据时只需替换数据源
（见 `src/schema.py` 的 `load_sessions`），特征、模型与指标完全不变。

## 生成模型（两层，模仿真实数据的产生方式）

**① 订单到达 —— 泊松计数过程**

    λ(s,d,h) = base_s · shape_s(h) · weekend_mult_s^[is_weekend] · day_effect(s,d)
    n(s,d,h) ~ Poisson(λ)

- `shape_s(h)`：每个场站一张 24 小时的形态表，归一化到均值 1。
- `day_effect(s,d) ~ LogNormal(0, σ_day)`：每个「场站 × 日历日」抽一次，
  代表天气、促销活动、节假日等当天整体冷热。

**② 单次充电 —— 时长与电量物理挂钩**

    D ~ LogNormal(μ_s, σ_s)   截断到 [10, 600] 分钟
    E = P_s · (D/60) · η,     η ~ Uniform(0.60, 0.95)

即「电量 = 额定功率 × 充电时长 × 充电效率」。这让 `充电时长_分钟` 成为电量的
真实解释变量，两者互相自洽 —— 否则数据一眼就能看出是编的。

## 为什么用对数正态日效应（这一段是本项目最重要的方法学防线）

真实场站有好日子和坏日子。**同时，它也是预测模型无法精确还原生成器的原因**：
`day_effect` 是逐日独立抽取的隐变量，模型看不到它，只能从滞后特征里间接估计。
因此模型在测试集上的误差是真实的，而不是「背下了生成器公式」。

配套的两条报告纪律（详见 README「关于合成数据」）：
1. 头条指标报**相对基线的提升**，而不是绝对 MAE —— 提升幅度与数据来源无关；
2. 明确注明：指标验证的是**建模流程的正确性**，绝对精度是模拟数据的属性，
   不代表真实场站的预测精度。

## 已知简化（答辩时必须能主动说出）

1. **无排队模型**：订单到达与充电时长独立抽取，理论上有小概率使同时充电数
   超过桩数。需要精确容量约束时应改用 M/G/c 排队模型。
2. **不建模电池容量上限**：实际充电会在电池充满时提前结束，本生成器不含该截断。
3. **跨零点的订单全部计入起始小时**：23:30 开始、持续 3 小时的订单算作 23 点，
   因此夜间负荷被轻微低估。
4. **日效应按场站独立抽取**：同城三个场站实际会共享天气扰动（正相关），
   本生成器未建模该共享成分。

## 可复现性

随机数发生器在**函数内部**创建（`np.random.default_rng(seed)`），不使用
`np.random.seed()` 修改全局状态 —— 后者在多线程的 Streamlit 服务里是真实的
正确性隐患。因此 `generate()` 是纯函数：同样输入永远得到逐位相同的输出。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 全局常量
# --------------------------------------------------------------------------- #

#: 所有输出行都带这一列，使「模拟 vs 真实」在数据层面就不可混淆
SYNTHETIC_LABEL = "模拟数据"

#: 固定随机种子。改动它会让全部指标变化，因此 README 里引用的数字必须与此一致。
SYNTHETIC_SEED = 42

#: 生成天数。取 28 天（4 个完整周）是为了让「周周期」滞后特征 lag_168 有足够样本：
#: 预热 168 小时后每站剩 21 天 = 504 个小时样本。
SYNTHETIC_N_DAYS = 28

#: 起始日期。选在竞赛日（2026-10-31）之前的 28 天，使数据落在合理的时间窗内。
#: 该日期为周一，使「周」的切分与日历周对齐。
SYNTHETIC_START_DATE = "2026-09-28"

#: 日效应强度的对数正态 sigma。含义：好日子约是坏日子的
#: exp(2 × 1.2816 × σ) ≈ 1.76 倍（σ=0.22）。
SYNTHETIC_SIGMA_DAY = 0.22

#: 单次充电效率的取值范围（交流/直流综合）。用于 E = P·(D/60)·η。
SYNTHETIC_ETA_RANGE = (0.60, 0.95)

#: 单次充电时长的物理截断（分钟）
SYNTHETIC_DURATION_BOUNDS = (10.0, 600.0)


# --------------------------------------------------------------------------- #
# 场站画像：本项目的全部数据假设都集中在这里
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StationProfile:
    """一个场站画像。**这个对象本身就是可答辩性**：

    指着它可以说「这些是我的假设，每条的理由写在注释里，改掉它整个系统重算」。
    界面上「模拟数据生成假设」面板渲染的就是这张表。
    """

    name: str
    """场站名称。**除本模块外，任何地方都不得硬编码场站名** ——
    竞赛现场的真实场站名一定不同，必须用 df["场站名称"].unique() 动态发现。"""

    short_code: str
    """订单号 / 桩号前缀，如 "A"。"""

    archetype: str
    """形态类别（商场 / 园区 / 枢纽），用于界面展示与答辩叙述。"""

    shape: tuple[float, ...]
    """24 元素的小时形态表（0 时到 23 时）。构造时自动归一化到均值 1，
    因此它只决定**形状**，不决定**总量**（总量由 base_orders_per_hour 决定）。"""

    base_orders_per_hour: float
    """基准小时订单到达率（单/小时），再乘以形态系数得到实际 λ。"""

    weekend_mult: float
    """周末倍率。园区接近停摆（0.22），商场更忙（1.35），枢纽大致持平（0.95）。"""

    charger_power_kw: float
    """单桩额定功率（kW）。商场用直流 60kW，园区用交流 7kW，枢纽用直流 120kW。"""

    n_chargers: int
    """充电桩数量。用于计算「谷时可用裕度」与「桩利用率」两个指标。"""

    duration_median_min: float
    """单次充电时长的中位数（分钟）。对数正态的 mu 由它取对数得到。"""

    duration_sigma: float
    """对数时长标准差。越大表示快充/慢充混杂越严重。"""

    note: str
    """一句话说明这个场站的运营特征，会显示在界面上。"""

    def __post_init__(self) -> None:
        shape = np.asarray(self.shape, dtype=float)
        if shape.shape != (24,):
            raise ValueError(f"{self.name}: shape 必须是 24 个元素，收到 {shape.shape}")
        if np.any(shape <= 0):
            raise ValueError(f"{self.name}: shape 必须全为正数")
        object.__setattr__(self, "shape", tuple((shape / shape.mean()).tolist()))

    @property
    def duration_mu(self) -> float:
        """对数正态的 mu。用中位数反推，比直接用均值更稳定也更好解释。"""
        return float(np.log(self.duration_median_min))

    @property
    def rated_capacity_kw(self) -> float:
        """场站额定容量（kW）= 桩数 × 单桩功率。"""
        return self.n_chargers * self.charger_power_kw

    def as_row(self) -> dict:
        """渲染到界面假设面板用的一行。每个值都带单位。"""
        return {
            "场站": self.name,
            "形态": self.archetype,
            "基准订单率 (单/h)": self.base_orders_per_hour,
            "周末倍率 (倍)": self.weekend_mult,
            "单桩功率 (kW)": self.charger_power_kw,
            "桩数 (个)": self.n_chargers,
            "额定容量 (kW)": self.rated_capacity_kw,
            "充电时长中位数 (min)": self.duration_median_min,
            "时长离散度 σ": self.duration_sigma,
            "运营特征": self.note,
        }


# --------------------------------------------------------------------------- #
# 三个场站画像
#
# 选这三者，是因为它们产生**性质不同的曲线形状**，而不只是水平不同 ——
# 这正是「多场站负荷诊断」非平凡、预测有意思的原因。
# --------------------------------------------------------------------------- #

_PROFILE_A = StationProfile(
    name="A-商业中心",
    short_code="A",
    archetype="商场",
    # 夜间低谷 → 午间小峰（12 点，午餐/逛街）→ 晚高峰（18–21 点）
    shape=(
        0.15, 0.08, 0.05, 0.04, 0.04, 0.05, 0.08, 0.15,   # 00–07 夜间低谷
        0.30, 0.50, 0.70, 0.95, 1.20, 1.05, 0.85, 0.80,   # 08–15 日间爬升，12 点午间小峰
        0.90, 1.20, 1.75, 2.10, 1.95, 1.40, 0.60, 0.30,   # 16–23 晚高峰 18–21 点
    ),
    base_orders_per_hour=6.0,
    weekend_mult=1.35,      # 周末逛街的人更多
    charger_power_kw=60.0,  # 商场以直流快充为主
    n_chargers=24,
    duration_median_min=45.0,   # 逛街顺带补电，充到够用就走
    duration_sigma=0.45,
    note="夜间低谷、午间小峰、晚高峰显著；周末更忙；以短时补电为主。",
)

_PROFILE_B = StationProfile(
    name="B-产业园区",
    short_code="B",
    archetype="园区",
    # 夜间近零 → 日间平台（09–16 点，通勤车停一整天）
    shape=(
        0.05, 0.03, 0.02, 0.02, 0.02, 0.03, 0.08, 0.30,   # 00–07 基本无车
        0.85, 1.60, 1.85, 1.90, 1.60, 1.70, 1.85, 1.80,   # 08–15 上班到达后形成平台
        1.55, 1.10, 0.60, 0.30, 0.15, 0.10, 0.07, 0.05,   # 16–23 下班陆续拔枪
    ),
    base_orders_per_hour=3.0,
    weekend_mult=0.22,      # 周末园区几乎停摆 —— 这是本场站最鲜明的特征
    charger_power_kw=7.0,   # 交流慢充，车停一整天
    n_chargers=60,
    duration_median_min=210.0,  # 中位 3.5 小时，长时停放
    duration_sigma=0.55,
    note="日间平台型负荷，周末几乎停摆；长时停放交流慢充，单均电量受 7kW 功率限制。",
)

_PROFILE_C = StationProfile(
    name="C-交通枢纽",
    short_code="C",
    archetype="枢纽",
    # 整夜持续基线（出租车夜间补电）+ 早晚双峰（7 点 / 18 点）
    shape=(
        0.55, 0.45, 0.40, 0.38, 0.40, 0.50, 0.85, 1.45,   # 00–07 夜间基线明显高于另两站
        1.30, 1.00, 0.90, 0.90, 0.95, 0.95, 1.00, 1.05,   # 08–15 早高峰后回落
        1.15, 1.45, 1.70, 1.55, 1.25, 1.00, 0.85, 0.70,   # 16–23 晚高峰 18 点
    ),
    base_orders_per_hour=8.0,
    weekend_mult=0.95,      # 通勤出行下降、休闲出行上升，大致抵消
    charger_power_kw=120.0, # 枢纽以大功率直流快充为主
    n_chargers=16,
    duration_median_min=35.0,   # 快充，短时，方差大
    duration_sigma=0.60,
    note="整夜存在持续基线（营运车辆夜间补电）+ 早晚双峰；快充、短时、离散度大。",
)

#: 三站画像。界面与脚本一律遍历此列表，绝不硬编码单个场站名。
STATION_PROFILES: tuple[StationProfile, ...] = (_PROFILE_A, _PROFILE_B, _PROFILE_C)


# --------------------------------------------------------------------------- #
# 输出 schema
# --------------------------------------------------------------------------- #

#: 生成器产出的列。这就是**原始数据契约** ——
#: 真实数据预期也提供这几列，其余派生列一律由 src/schema.py 统一生成。
RAW_COLUMNS: tuple[str, ...] = (
    "场站名称",
    "订单ID",
    "充电开始时间",
    "充电量_kWh",
    "充电时长_分钟",
    "充电桩编号",
    "数据来源",
)


# --------------------------------------------------------------------------- #
# 生成
# --------------------------------------------------------------------------- #

def generate(
    seed: int = SYNTHETIC_SEED,
    n_days: int = SYNTHETIC_N_DAYS,
    start_date: str = SYNTHETIC_START_DATE,
    profiles: tuple[StationProfile, ...] = STATION_PROFILES,
) -> pd.DataFrame:
    """生成三场站模拟充电订单明细。

    纯函数：相同参数永远返回逐位相同的 DataFrame。见模块 docstring「可复现性」。

    Returns
    -------
    DataFrame，列为 RAW_COLUMNS，按「充电开始时间」升序。**index 已重置为 0..n-1**，
    因为调用方（Streamlit 缓存）会跨会话共享这个对象，绝不能依赖或原地修改 index。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(pd.Timestamp(start_date), periods=n_days, freq="D")
    is_weekend = (dates.dayofweek >= 5).to_numpy()

    frames: list[pd.DataFrame] = []

    for prof in profiles:
        shape = np.asarray(prof.shape, dtype=float)

        # ① 日效应：每个「场站 × 日历日」抽一次。这是模型看不到的隐变量，
        #    也是模型误差真实存在的根本原因。
        day_effect = rng.lognormal(mean=0.0, sigma=SYNTHETIC_SIGMA_DAY, size=n_days)

        # ② 到达强度 λ，形状 (n_days, 24)
        lam = (
            prof.base_orders_per_hour
            * shape[None, :]
            * np.where(is_weekend[:, None], prof.weekend_mult, 1.0)
            * day_effect[:, None]
        )

        # ③ 泊松抽样得到每天的逐小时订单数
        counts = rng.poisson(lam)

        # ④ 展开成一单一行
        day_idx, hour_idx = np.nonzero(counts)
        repeats = counts[day_idx, hour_idx]
        day_idx = np.repeat(day_idx, repeats)
        hour_idx = np.repeat(hour_idx, repeats)
        n_sessions = day_idx.size

        if n_sessions == 0:
            continue

        # ⑤ 单次时长（对数正态，物理截断）
        duration = rng.lognormal(
            mean=prof.duration_mu, sigma=prof.duration_sigma, size=n_sessions
        )
        duration = np.clip(duration, *SYNTHETIC_DURATION_BOUNDS)

        # ⑥ 电量 = 额定功率 × 充电时长 × 充电效率
        eta = rng.uniform(*SYNTHETIC_ETA_RANGE, size=n_sessions)
        energy = prof.charger_power_kw * (duration / 60.0) * eta

        # ⑦ 起始分钟随机化，使时间戳不是整点 —— 真实数据一定有分钟级抖动
        minute = rng.integers(0, 60, size=n_sessions)

        start_time = (
            dates[day_idx]
            + pd.to_timedelta(hour_idx, unit="h")
            + pd.to_timedelta(minute, unit="m")
        )

        charger_no = rng.integers(1, prof.n_chargers + 1, size=n_sessions)

        frames.append(
            pd.DataFrame(
                {
                    "场站名称": prof.name,
                    "订单ID": [
                        f"{prof.short_code}{i:06d}" for i in range(n_sessions)
                    ],
                    "充电开始时间": start_time,
                    "充电量_kWh": np.round(energy, 2),
                    "充电时长_分钟": np.round(duration, 1),
                    "充电桩编号": [
                        f"{prof.short_code}-{c:02d}" for c in charger_no
                    ],
                    "数据来源": SYNTHETIC_LABEL,
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("充电开始时间", kind="stable").reset_index(drop=True)
    return out[list(RAW_COLUMNS)]


# --------------------------------------------------------------------------- #
# 供界面使用的辅助函数
# --------------------------------------------------------------------------- #

def profiles_table(profiles: tuple[StationProfile, ...] = STATION_PROFILES) -> pd.DataFrame:
    """把场站画像渲染成表，供「模拟数据生成假设」面板展示。"""
    return pd.DataFrame([p.as_row() for p in profiles])


def shape_table(profiles: tuple[StationProfile, ...] = STATION_PROFILES) -> pd.DataFrame:
    """把 24 小时形态表渲染成表（小时 × 场站），供答辩时逐格解释。

    注意：这里返回的是**归一化后**的形态系数，均值恒为 1。
    """
    data = {"小时": list(range(24))}
    for p in profiles:
        data[p.name] = [round(v, 3) for v in p.shape]
    return pd.DataFrame(data)


#: 生成器的已知简化，直接渲染到界面，避免答辩时被问倒。
SIMULATION_CAVEATS: tuple[str, ...] = (
    "无排队模型：订单到达与时长独立抽取，理论上同时充电数可能略超桩数。",
    "不建模电池充满后的提前结束，因此长时停放的订单电量不设上限。",
    "跨零点订单全部计入起始小时，夜间负荷被轻微低估。",
    "日效应按场站独立抽取，未建模同城场站之间的天气相关性。",
)
