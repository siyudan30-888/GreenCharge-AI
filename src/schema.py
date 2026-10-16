"""数据契约、时段划分与统一的载入 / 校验 / 派生路径。

**本模块是数据的唯一入口。** 无论数据来自内置模拟生成器还是用户上传的 CSV，
都必须经过 `load_sessions()`，因此两条路径不可能出现 schema 漂移。

## 原始数据契约

必需三列（其余可选）：

| 列名 | 类型 | 含义 |
| --- | --- | --- |
| `场站名称` | 文本 | 场站标识。**场站数量与名称全部动态发现**，不预设 |
| `充电开始时间` | 时间 | 订单开始充电的时刻，精确到分钟 |
| `充电量_kWh` | 数值 | 本次充电电量 |

可选列：`充电时长_分钟`（缺失时「桩利用率」「谷时可用裕度」不可用）、
`订单ID`、`充电桩编号`、`充电开始小时`。

常见别名会自动映射（见 `COLUMN_ALIASES`），例如「站点名称」「电量」「充电度数」
都能识别 —— 竞赛现场拿到的真实数据不会正好用本项目的列名。

## 编码

按 `utf-8-sig` → `gb18030` → `utf-8` 依次回退。`utf-8-sig` 放第一位是刻意的：
Excel 导出的 CSV 带 UTF-8 BOM，首列名会变成 `﻿场站名称`，
表现为一个非常诡异的「明明有这一列却报缺列」的 KeyError。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import IO, Sequence

import pandas as pd

# --------------------------------------------------------------------------- #
# 列名契约与别名
# --------------------------------------------------------------------------- #

REQUIRED_COLUMNS: tuple[str, ...] = ("场站名称", "充电开始时间", "充电量_kWh")

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "充电时长_分钟",
    "订单ID",
    "充电桩编号",
    "充电开始小时",
)

#: 派生列，由 `derive()` 统一生成，**不接受**外部提供
DERIVED_COLUMNS: tuple[str, ...] = ("日期", "充电开始小时", "时段类型", "是否周末")

#: 别名表：规范列名 → 可接受的写法（小写比较，忽略首尾空白与 BOM）
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "场站名称": ("场站名称", "站点名称", "场站", "站点", "站名", "station", "station_name"),
    "充电开始时间": (
        "充电开始时间", "开始时间", "充电时间", "充电开始",
        "start_time", "charge_start_time", "timestamp",
    ),
    "充电量_kWh": (
        "充电量_kwh", "充电量", "电量", "充电电量", "充电度数", "度数",
        "energy_kwh", "kwh", "energy",
    ),
    "充电时长_分钟": (
        "充电时长_分钟", "充电时长", "时长", "充电分钟数",
        "duration_min", "duration", "duration_minutes",
    ),
    "订单ID": ("订单id", "订单编号", "订单号", "order_id", "orderid", "id"),
    "充电桩编号": ("充电桩编号", "桩号", "充电桩", "充电枪", "pile_id", "charger_id"),
    "充电开始小时": ("充电开始小时", "小时", "hour", "start_hour"),
}


def _normalize_key(name: object) -> str:
    """列名归一化：去 BOM、去首尾空白、转小写。"""
    return str(name).replace("﻿", "").strip().lower()


# --------------------------------------------------------------------------- #
# 时段划分
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TouConfig:
    """分时电价时段划分。

    ⚠️ **以下默认值仅为示例，上线前必须按当地现行分时电价文件核对替换。**
    本项目不宣称知道任何地区的现行时段边界；把它做成可配置对象、并在界面上
    开放编辑，正是为了不让一个未经核实的假设伪装成事实。
    """

    valley: tuple[int, ...]
    peak: tuple[int, ...]
    flat: tuple[int, ...]

    def __post_init__(self) -> None:
        all_hours: list[int] = sorted(self.valley + self.peak + self.flat)
        if all_hours != list(range(24)):
            raise ValueError(
                "时段划分必须恰好覆盖 0–23 且互不重叠；"
                f"当前覆盖 {len(all_hours)} 个小时，重复或缺失："
                f"{sorted({h for h in all_hours if all_hours.count(h) > 1})}"
            )

    @property
    def peak_hours(self) -> int:
        return len(self.peak)

    @property
    def valley_hours(self) -> int:
        return len(self.valley)

    def label_of(self, hour: int) -> str:
        if hour in self.peak:
            return PEAK
        if hour in self.valley:
            return VALLEY
        return FLAT

    def as_hour_frame(self) -> pd.DataFrame:
        """渲染成「小时 → 时段」对照表，供界面展示。"""
        return pd.DataFrame(
            {
                "小时": list(range(24)),
                "时段": [self.label_of(h) for h in range(24)],
            }
        )


PEAK, FLAT, VALLEY = "峰时", "平时", "谷时"

#: 可辩护的默认划分（华东地区常见的两峰一谷结构）：
#:   谷 00–08（8h）｜峰 08–12 + 17–21（8h）｜平 12–17 + 21–24（8h）
DEFAULT_TOU = TouConfig(
    valley=(0, 1, 2, 3, 4, 5, 6, 7),
    peak=(8, 9, 10, 11, 17, 18, 19, 20),
    flat=(12, 13, 14, 15, 16, 21, 22, 23),
)

TOU_SOURCE_NOTE = (
    "**示例划分，非事实陈述。** 默认值为 谷 00–08 / 峰 08–12 与 17–21 / 平其余，"
    "是华东地区常见的两峰一谷结构。本平台不宣称知道任何地区的现行时段，"
    "上线前须按当地现行分时电价文件核对替换。"
)


# --------------------------------------------------------------------------- #
# 结果对象与异常
# --------------------------------------------------------------------------- #

class SessionDataError(Exception):
    """数据不可用。异常信息是**面向用户的中文说明**，可直接上屏，不含 traceback。"""


@dataclass
class LoadResult:
    """一次载入的完整结果。把警告随数据一起带出来，避免用异常处理可恢复的问题。"""

    df: pd.DataFrame
    source: str
    """数据来源描述，如「内置模拟数据」或「用户上传：xxx.csv」"""
    is_synthetic: bool
    n_rows: int
    stations: list[str]
    time_range: tuple[pd.Timestamp, pd.Timestamp]
    has_duration: bool
    """是否具备「充电时长_分钟」，决定桩利用率与谷时可用裕度是否可用"""
    tou: TouConfig
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 读取与解码
# --------------------------------------------------------------------------- #

def _decode_bytes(raw: bytes) -> tuple[str, str]:
    """尝试多种编码解码 CSV 字节流，返回 (文本, 实际使用的编码)。

    顺序有讲究：utf-8-sig 放第一，因为它同时能处理「带 BOM 的 UTF-8」和
    「不带 BOM 的 UTF-8」两种情况；gb18030 是 GBK/GB2312 的超集，覆盖
    中文 Windows 上 Excel 导出的另一大类文件。
    """
    errors: list[str] = []
    for enc in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError as exc:
            errors.append(f"{enc}: {exc.reason}")
    raise SessionDataError(
        "无法识别文件的文字编码，已尝试 utf-8-sig / gb18030 / utf-8 均失败。\n\n"
        "请用 Excel 另存为「CSV UTF-8（逗号分隔）」后重新上传。\n\n"
        "各编码的失败原因：\n" + "\n".join(f"- {e}" for e in errors)
    )


def read_csv_any(source: str | bytes | IO[bytes]) -> tuple[pd.DataFrame, str]:
    """宽容地读取 CSV，返回 (DataFrame, 编码)。列名已做归一化匹配前的原样保留。"""
    if isinstance(source, bytes):
        raw = source
    elif hasattr(source, "read"):
        raw = source.read()
    else:
        with open(source, "rb") as fh:
            raw = fh.read()

    if not raw.strip():
        raise SessionDataError("文件是空的，没有任何内容可以读取。")

    text, encoding = _decode_bytes(raw)

    try:
        df = pd.read_csv(
            io.StringIO(text),
            dtype=str,          # 全部按文本读入，类型转换由本模块统一负责并给出友好报错
            keep_default_na=True,
            skipinitialspace=True,
        )
    except Exception as exc:  # pandas 的解析异常类型很杂，统一转成用户可读信息
        raise SessionDataError(f"CSV 解析失败：{exc}") from exc

    # 去掉完全空白的行与列（Excel 导出常见）
    df = df.dropna(how="all").loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df, encoding


def _map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """把实际读到的列名映射到规范列名。返回 (重命名后的 df, 原始列名列表)。"""
    actual = list(df.columns)
    lookup: dict[str, str] = {}
    for col in actual:
        lookup.setdefault(_normalize_key(col), col)

    rename: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            hit = lookup.get(_normalize_key(alias))
            if hit is not None and hit not in rename:
                rename[hit] = canonical
                break

    if rename:
        df = df.rename(columns=rename)
    return df, [str(c) for c in actual]


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #

def _validate(df: pd.DataFrame, raw_columns: Sequence[str]) -> list[str]:
    """就地校验并转换类型。返回警告列表；不可恢复的问题抛 SessionDataError。"""
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SessionDataError(
            "上传的文件缺少必需的列：" + "、".join(f"「{c}」" for c in missing) + "\n\n"
            f"**实际读到的列名**（共 {len(raw_columns)} 列）：\n"
            + "、".join(f"「{c}」" for c in raw_columns) + "\n\n"
            "字段名匹配不区分大小写并会去除首尾空白；若列名是英文或其他写法，"
            "请参照下方「数据字典」改用标准列名，或使用页面上的「下载示例 CSV」。"
        )

    # ---- 场站名称 ---------------------------------------------------------- #
    station = df["场站名称"]
    if station.isna().any():
        n_bad = int(station.isna().sum())
        df.drop(index=df.index[station.isna()], inplace=True)
        warnings.append(f"有 {n_bad} 行缺少场站名称，已丢弃。")
        station = df["场站名称"]
    df["场站名称"] = station.astype(str).str.strip()
    if (df["场站名称"] == "").any():
        n_bad = int((df["场站名称"] == "").sum())
        df.drop(index=df.index[df["场站名称"] == ""], inplace=True)
        warnings.append(f"有 {n_bad} 行场站名称为空白，已丢弃。")
    if df.empty:
        raise SessionDataError("清洗后没有任何有效数据行，请检查文件内容。")

    # ---- 充电开始时间 ------------------------------------------------------ #
    ts = pd.to_datetime(df["充电开始时间"], errors="coerce")
    bad_ts = int(ts.isna().sum())
    if bad_ts == len(ts):
        sample = df["充电开始时间"].dropna().head(3).tolist()
        raise SessionDataError(
            "「充电开始时间」列无法解析为时间。\n\n"
            f"该列的前几个值是：{sample}\n\n"
            "建议使用 `YYYY-MM-DD HH:MM:SS` 或 `YYYY/MM/DD HH:MM` 格式。"
        )
    if bad_ts:
        warnings.append(f"有 {bad_ts} 行时间无法解析，已丢弃。")
        keep = ts.notna()
        df.drop(index=df.index[~keep], inplace=True)
        ts = ts[keep]
    df["充电开始时间"] = ts

    # ---- 充电量 ------------------------------------------------------------ #
    energy = pd.to_numeric(df["充电量_kWh"], errors="coerce")
    bad_e = int(energy.isna().sum())
    if bad_e == len(energy):
        sample = df["充电量_kWh"].dropna().head(3).tolist()
        raise SessionDataError(
            "「充电量_kWh」列无法解析为数值。\n\n"
            f"该列的前几个值是：{sample}\n\n"
            "该列应为纯数字，不含单位（如不要写「35.2 kWh」，应写 35.2）。"
        )
    if bad_e:
        warnings.append(f"有 {bad_e} 行电量无法解析为数值，已丢弃。")
        df.drop(index=df.index[energy.isna()], inplace=True)
        energy = energy[energy.notna()]
        ts = df["充电开始时间"]
    df["充电量_kWh"] = energy.astype(float)

    negatives = int((df["充电量_kWh"] < 0).sum())
    if negatives:
        df.drop(index=df.index[df["充电量_kWh"] < 0], inplace=True)
        warnings.append(f"有 {negatives} 行电量为负值（可能是退款冲正），已丢弃。")

    if df.empty:
        raise SessionDataError("清洗后没有任何有效数据行，请检查文件内容。")

    # ---- 可选：充电时长 ---------------------------------------------------- #
    if "充电时长_分钟" in df.columns:
        dur = pd.to_numeric(df["充电时长_分钟"], errors="coerce")
        bad_d = int(dur.isna().sum())
        if bad_d == len(dur):
            df.drop(columns=["充电时长_分钟"], inplace=True)
            warnings.append("「充电时长_分钟」列无法解析为数值，已忽略该列。")
        else:
            if bad_d:
                warnings.append(f"「充电时长_分钟」有 {bad_d} 行为空或非法，已置为缺失。")
            dur = dur.where(dur > 0)
            df["充电时长_分钟"] = dur.astype(float)

    # ---- 可选：其余列 ------------------------------------------------------ #
    for col in ("订单ID", "充电桩编号"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "数据来源" not in df.columns:
        df["数据来源"] = "用户上传"

    return warnings


# --------------------------------------------------------------------------- #
# 派生
# --------------------------------------------------------------------------- #

def derive(df: pd.DataFrame, tou: TouConfig = DEFAULT_TOU) -> pd.DataFrame:
    """生成派生列。**两条数据路径共用这一个函数**，因此 schema 不可能漂移。

    派生列：`日期`、`充电开始小时`、`时段类型`、`是否周末`。

    注意：调用方拿到的可能是 Streamlit 跨会话共享的缓存对象，
    因此这里一律**返回新对象**，绝不原地修改入参。
    """
    out = df.copy()
    ts = out["充电开始时间"]

    out["日期"] = ts.dt.normalize()
    out["充电开始小时"] = ts.dt.hour.astype("int64")
    out["时段类型"] = out["充电开始小时"].map(tou.label_of)
    # 时段类型映射结果需要显式声明为字符串类别，避免下游 groupby 出现空类别
    out["时段类型"] = out["时段类型"].astype("category")
    out["是否周末"] = (ts.dt.dayofweek >= 5)

    # 若数据源自带「充电开始小时」，以时间戳为准并核对是否一致
    out = out.sort_values("充电开始时间", kind="stable").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# 唯一入口
# --------------------------------------------------------------------------- #

def load_sessions(
    uploaded: bytes | IO[bytes] | None = None,
    *,
    tou: TouConfig = DEFAULT_TOU,
    demo_frame: pd.DataFrame | None = None,
) -> LoadResult:
    """**本项目的唯一数据入口。**

    Parameters
    ----------
    uploaded:
        用户上传的文件内容（字节或类文件对象）。为 None 时使用内置模拟数据。
    tou:
        时段划分配置。
    demo_frame:
        预先算好的模拟数据（由调用方传入 `synthetic.generate()` 的结果，
        以便复用 Streamlit 缓存）。为 None 时本函数自行生成。

    Returns
    -------
    LoadResult

    Raises
    ------
    SessionDataError
        信息为面向用户的中文说明，可直接 `st.error()` 上屏。
    """
    from . import synthetic  # 延迟导入，避免 metrics/models 反向依赖时出现环

    warnings: list[str] = []

    if uploaded is None:
        raw = demo_frame if demo_frame is not None else synthetic.generate()
        df = raw.copy()
        source = "内置模拟数据"
        is_synthetic = True
        if "数据来源" not in df.columns:
            df["数据来源"] = synthetic.SYNTHETIC_LABEL
    else:
        df, encoding = read_csv_any(uploaded)
        raw_columns = [str(c) for c in df.columns]
        df, raw_columns = _map_columns(df)
        warnings.extend(_validate(df, raw_columns))
        source = "用户上传"
        is_synthetic = False
        if encoding != "utf-8-sig":
            warnings.append(f"文件按 {encoding} 编码解析（非 UTF-8）。")

    df = derive(df, tou)

    return LoadResult(
        df=df,
        source=source,
        is_synthetic=is_synthetic,
        n_rows=len(df),
        stations=sorted(df["场站名称"].unique().tolist()),
        time_range=(df["充电开始时间"].min(), df["充电开始时间"].max()),
        has_duration="充电时长_分钟" in df.columns,
        tou=tou,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# 供界面使用
# --------------------------------------------------------------------------- #

def data_dictionary() -> pd.DataFrame:
    """数据字典，渲染到界面上，供评委核对字段。"""
    rows = [
        ("场站名称", "必需", "文本", "场站标识；场站数量与名称由数据动态发现"),
        ("充电开始时间", "必需", "时间", "订单开始充电的时刻，建议精确到分钟"),
        ("充电量_kWh", "必需", "数值", "本次充电电量，纯数字不含单位"),
        ("充电时长_分钟", "可选", "数值", "缺失时「桩利用率」与「谷时可用裕度」不可用"),
        ("订单ID", "可选", "文本", "订单唯一标识"),
        ("充电桩编号", "可选", "文本", "充电枪/桩编号"),
        ("日期", "派生", "日期", "由 充电开始时间 归一化得到"),
        ("充电开始小时", "派生", "整数", "0–23，由 充电开始时间 提取"),
        ("时段类型", "派生", "类别", "峰时 / 平时 / 谷时，按当前时段划分判定"),
        ("是否周末", "派生", "布尔", "由 充电开始时间 的星期推出"),
    ]
    return pd.DataFrame(rows, columns=["列名", "必需性", "类型", "说明"])


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """导出为带 BOM 的 UTF-8 CSV —— 正是 Excel 能正确打开中文的那种格式。

    页面上提供下载按钮，是为了证明「下载 → 重新上传」这条往返链路真的可用。
    """
    return df.to_csv(index=False).encode("utf-8-sig")
