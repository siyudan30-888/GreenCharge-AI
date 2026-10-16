"""命令行评估：跑出可复现的指标报告。

**这个脚本的输出会被逐字粘贴进 README。** 理由见 README「为什么把指标抄进文档」：

> 评委的机器不是这台机器。如果应用在对方机器上因为环境问题跑不起来，
> 至少要让他们看到这套方法在**某台机器上确实跑出过这些数字**，
> 而且任何人都能用一条命令复现。证据必须在应用跑不起来时依然成立。

本脚本**复用 `src/models.py` 的 `train_and_evaluate()`**，不重复实现任何逻辑 ——
因此 README 里引用的数字与界面上显示的数字必然一致，不存在两套口径。

用法：
    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 允许以 `python scripts/evaluate.py` 的方式从仓库根目录运行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，中文与 emoji 会乱码或直接抛 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # 某些环境下 stdout 不支持重配置，忽略即可
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import metrics, models, synthetic  # noqa: E402
from src.features import build_features  # noqa: E402
from src.schema import DEFAULT_TOU, load_sessions  # noqa: E402

WIDTH = 78


def header(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def section(title: str) -> None:
    print()
    print(f"—— {title} " + "-" * max(WIDTH - len(title) - 4, 3))


def main() -> int:
    started = time.perf_counter()

    header("GreenCharge-AI  模型评估报告")
    print(f"生成工具   : scripts/evaluate.py")
    print(f"随机种子   : {synthetic.SYNTHETIC_SEED}（固定，任何机器上结果一致）")
    print(f"库版本     : pandas {pd.__version__} / numpy {np.__version__}")
    print()
    print("⚠️  本报告基于**内置模拟数据**，不是真实充电场站数据。")
    print("    指标验证的是**建模流程**（防泄漏特征、时序划分、基准对照、评估口径）")
    print("    的正确性；绝对精度是模拟数据的属性，不代表真实场站的预测精度。")

    # ------------------------------------------------------------------ 数据 #
    section("1. 数据规模")
    result = load_sessions()
    df = result.df
    span = result.time_range
    print(f"数据来源   : {result.source}")
    print(f"订单总数   : {len(df):,} 单")
    print(f"场站数量   : {len(result.stations)} 个 —— {' / '.join(result.stations)}")
    print(f"时间跨度   : {span[0]:%Y-%m-%d %H:%M} → {span[1]:%Y-%m-%d %H:%M}"
          f"（{(span[1] - span[0]).days + 1} 天）")
    print(f"总电量     : {df['充电量_kWh'].sum():,.1f} kWh")
    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠ {w}")

    section("2. 各场站订单特征（形态不同的直接证据）")
    station_stats = (
        df.groupby("场站名称", observed=True)
        .agg(
            订单数=("充电量_kWh", "size"),
            总电量_kWh=("充电量_kWh", "sum"),
            单均电量_kWh=("充电量_kWh", "mean"),
            单均时长_min=("充电时长_分钟", "mean"),
        )
        .round(2)
        .reset_index()
    )
    print(station_stats.to_string(index=False))

    # ------------------------------------------------------------ 负荷指标 #
    section("3. 多场站负荷特征")
    # 资产参数（桩数 × 单桩功率）逐场站不同，且**只有模拟数据才已知** ——
    # 用户上传的真实数据里没有这两列，那时客观地显示「—」而不是编一个。
    capacity_map = None
    charger_power_map = None
    if result.is_synthetic:
        capacity_map = {p.name: p.rated_capacity_kw for p in synthetic.STATION_PROFILES}
        charger_power_map = {p.name: p.charger_power_kw for p in synthetic.STATION_PROFILES}
    analysis = metrics.analyze_load(
        df, DEFAULT_TOU, capacity_kw=capacity_map, charger_power_kw=charger_power_map
    )
    print(metrics.metrics_table(analysis.metrics).to_string(index=False))
    print()
    if capacity_map:
        print("资产假设（模拟数据自带，真实数据需另行提供）：")
        for p in synthetic.STATION_PROFILES:
            print(
                f"  {p.name}: {p.n_chargers} 桩 × {p.charger_power_kw:g} kW "
                f"= 额定容量 {p.rated_capacity_kw:,.0f} kW"
            )
        print()
    print("指标口径说明：")
    print("  负荷率     = 平均功率 / **实测**峰值功率（不是合同容量）")
    print("  峰谷差率   = 峰谷差 / **实测**峰值功率 × 100%")
    print("  谷电利用率b= 谷时**日均**电量 / (额定容量 × 谷时小时数) × 100%")
    print("  桩利用率   = Σ充电时长 / (桩数 × 统计时长) × 100%")
    print("  注：额定容量相关指标在未提供容量时为「—」，不用默认值冒充。")

    section("4. 站间相关系数（逐小时功率曲线）")
    if analysis.correlation is not None:
        print(analysis.correlation.round(3).to_string())

    # ---------------------------------------------------------------- 建模 #
    section("5. 特征与数据划分")
    feats = build_features(df)
    print(f"聚合粒度   : 场站 × 小时（缺失小时已补零，避免滞后特征在时间轴上错位）")
    print(f"预热丢弃   : {feats.n_dropped_warmup} 行（lag_168 需要 168 小时前置数据）")
    print(f"有效样本   : {len(feats.frame):,} 行 × {len(feats.feature_columns)} 个特征")
    print(f"特征清单   : {', '.join(feats.feature_columns)}")

    section("6. 模型对比（同一测试集，同一行集）")
    report = models.train_and_evaluate(feats)
    print(f"训练集     : {report.n_train:,} 行（{feats.frame['时刻'].min():%Y-%m-%d} → "
          f"{report.split_time:%Y-%m-%d %H:%M}）")
    print(f"测试集     : {report.n_test:,} 行（{report.split_time:%Y-%m-%d %H:%M} 之后）")
    print("划分方式   : **按时间顺序**留出最后 20%，不做随机打乱")
    print()
    print(f"主模型超参 : {report.hgb_config}"
          "（在**训练集内部**的时间顺序验证集上选出，测试集未参与）")
    if not report.hgb_search.empty:
        print()
        print("超参扫描前 5 名（内部验证集，共 "
              f"{len(report.hgb_search)} 个候选）：")
        print(report.hgb_search.head(5).to_string(index=False))
    print()
    print(report.metrics.to_string(index=False))
    print()
    print(f"最强基线   : {report.baseline_name}（MAE {report.baseline_mae:.3f} kWh）")
    print(f"相对提升   : {report.improvement_pct:+.1f}%   ← **头条指标**")
    print()
    print("为什么头条是「相对提升」而不是绝对 MAE：")
    print("  绝对 MAE 是模拟数据的属性，会随生成器参数变化；")
    print("  而「比最强基线好多少」衡量的是建模流程本身的价值，与数据来源无关。")

    if report.notes:
        print()
        for note in report.notes:
            print(f"  {note}")

    section("7. 分场站误差（主模型 vs 季节性朴素基线）")
    print(report.per_station.to_string(index=False))
    print()
    print("主动披露误差最大的场站 —— 一个好看的均值不如一个诚实的分布。")

    if not report.importance.empty:
        section("8. 排列重要性 Top 8（测试集，scoring = −MAE）")
        print(report.importance.head(8).round(4).to_string(index=False))

    if report.multistep_mae is not None:
        section(f"9. 多步回测（{report.multistep_horizon} 小时递归预测）")
        print(f"单步 MAE   : {report.primary_mae:.3f} kWh")
        print(f"多步 MAE   : {report.multistep_mae:.3f} kWh")
        ratio = report.multistep_mae / report.primary_mae if report.primary_mae else float("nan")
        print(f"倍率       : {ratio:.2f}×")
        print()
        print("单步 MAE 是「已知前一小时真值」条件下的误差；有序充电需要的是")
        print("提前**一整天**的预测。从第 2 小时起，lag_1 用的是模型自己的预测值，")
        print("误差会累积。策略页面画不确定性带时采用的是**多步 MAE**，")
        print("因为用单步 MAE 会系统性低估不确定性。")

    # -------------------------------------------------------------- 聚类 #
    section("10. 日负荷形态聚类（无监督）")
    clusters = metrics.cluster_daily_shapes(analysis.curve)
    if clusters is None:
        print("样本量不足，未进行聚类。")
    else:
        print("k 由轮廓系数扫描决定，**不是因为有 3 个场站就定 k=3**：")
        print()
        print(clusters.k_scores.round(4).to_string(index=False))
        print()
        print(f"选定 k = {clusters.k}（轮廓系数 {clusters.silhouette:.4f}）")
        print()
        print("归一化是关键：每天曲线除以自身均值，剥离水平、保留形状。")
        print("否则聚类只会按「哪个场站电量多」分组，得到的簇等价于场站本身。")
        print()
        print(metrics.station_shape_labels(clusters).to_string(index=False))

    # ---------------------------------------------------------- 敏感性表 #
    section("11. 有序充电敏感性分析（以第一个场站为例）")
    from src.strategy import Assumptions, plan_shift, sensitivity_table
    from src.metrics import CarbonParams

    assumptions = Assumptions()
    carbon = CarbonParams()
    first_station = feats.stations[0]
    m0 = next(m for m in analysis.metrics if m.station == first_station)
    capacity = synthetic.STATION_PROFILES[0].rated_capacity_kw

    plan = plan_shift(first_station, m0, assumptions, DEFAULT_TOU, capacity_kw=capacity)
    print(f"场站       : {first_station}")
    print(f"θ          : {plan.theta:.2f}（可调假设，非实测值）")
    print(f"E_峰       : {plan.peak_energy_kwh:,.1f} kWh")
    print(f"H_谷裕度   : {plan.valley_margin_kwh:,.1f} kWh（额定容量 × 谷时小时 − 谷时电量）")
    print(f"可转移电量 : min(E_峰 × θ, H_谷) = {plan.shiftable_kwh:,.1f} kWh")
    print(f"约束来源   : {plan.binding_constraint}")
    print()
    print(sensitivity_table(first_station, m0, assumptions, carbon, DEFAULT_TOU, capacity_kw=capacity).to_string(index=False))
    print()
    from src.strategy import SENSITIVITY_DISCLAIMER
    print(SENSITIVITY_DISCLAIMER)

    # ---------------------------------------------------------- 确定性检查 #
    section("12. 确定性检查")
    same = synthetic.generate().equals(synthetic.generate())
    print(f"generate() 连续两次调用结果逐位相同：{same}")
    print(f"（随机数发生器在函数内部创建，不使用 np.random.seed 修改全局状态）")

    elapsed = time.perf_counter() - started
    header(f"完成，用时 {elapsed:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
