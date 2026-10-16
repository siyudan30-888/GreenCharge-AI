import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="GreenCharge-AI 充电与碳管理平台", layout="wide")

st.title("⚡ GreenCharge-AI：电动汽车充电场站负荷诊断与碳普惠核算系统")
st.markdown("面向 **多场站充电运营分析、有序充电调度与碳减排核算** 的智能决策原型系统")

# 侧边栏：数据上传与参数设置
st.sidebar.header("📂 场站数据源配置")
uploaded_file = st.sidebar.file_uploader("上传场站充电数据 (CSV)", type=["csv"])
carbon_factor = st.sidebar.slider("度电碳减排核算系数 (kg CO₂/kWh)", 0.40, 0.80, 0.58)

@st.cache_data
def load_demo_data():
    np.random.seed(42)
    stations = ["A场站-商业中心", "B场站-产业园区", "C场站-公用交通枢纽"]
    hours = np.random.choice(range(24), size=600, p=[
        0.03,0.02,0.01,0.01,0.01,0.02, 0.03,0.05,0.07,0.06,0.05,0.06,
        0.07,0.06,0.04,0.04,0.05,0.07, 0.09,0.07,0.04,0.02,0.02,0.01
    ])
    df = pd.DataFrame({
        "场站名称": np.random.choice(stations, 600),
        "充电开始小时": hours,
        "充电量_kWh": np.round(np.random.gamma(shape=3.0, scale=10.0, size=600), 1),
        "充电时长_分钟": np.random.randint(20, 120, size=600)
    })
    df["时段类型"] = df["充电开始小时"].apply(
        lambda h: "峰时 (10-12, 18-22)" if h in [10,11,18,19,20,21] 
        else ("谷时 (00-08)" if h < 8 else "平时")
    )
    df["碳减排量_kg"] = np.round(df["充电量_kWh"] * carbon_factor, 2)
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("已加载自定义数据！")
else:
    df = load_demo_data()
    st.sidebar.info("当前展示：3个典型充电场站模拟基准数据")

# 核心指标卡 (KPI)
total_kwh = df["充电量_kWh"].sum()
total_carbon = (df["充电量_kWh"] * carbon_factor).sum()
peak_ratio = (df[df["时段类型"].str.startswith("峰时")]["充电量_kWh"].sum() / total_kwh) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("累计服务订单数", f"{len(df)} 单")
col2.metric("总充电负荷 (kWh)", f"{total_kwh:,.1f} kWh")
col3.metric("累计碳减排核算量", f"{total_carbon:,.1f} kg CO₂")
col4.metric("电网高峰期充电占比", f"{peak_ratio:.1f}%", "-需引导错峰" if peak_ratio > 30 else "负荷健康")

st.divider()

# 图表区
c1, c2 = st.columns(2)
with c1:
    st.subheader("📊 三大场站 24 小时充电负荷分布热力趋势")
    hourly_load = df.groupby(["充电开始小时", "场站名称"])["充电量_kWh"].sum().reset_index()
    fig1 = px.line(hourly_load, x="充电开始小时", y="充电量_kWh", color="场站名称", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.subheader("🌱 各场站峰谷电量结构与碳普惠贡献")
    station_summary = df.groupby(["场站名称", "时段类型"])["充电量_kWh"].sum().reset_index()
    fig2 = px.bar(station_summary, x="场站名称", y="充电量_kWh", color="时段类型", barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

# AI 策略生成区
st.divider()
st.subheader("🤖 AI 有序充电与碳积分激励策略生成")
selected_station = st.selectbox("选择需要诊断的目标场站：", df["场站名称"].unique())

if st.button("生成 AI 运营与错峰调度方案"):
    st_df = df[df["场站名称"] == selected_station]
    st_peak = (st_df[st_df["时段类型"].str.startswith("峰时")]["充电量_kWh"].sum() / st_df["充电量_kWh"].sum()) * 100
    st.success(f"**【{selected_station} 诊断结果】** 当前峰时充电负荷占比为 **{st_peak:.1f}%**，累计创造碳减排 **{st_df['充电量_kWh'].sum()*carbon_factor:.1f} kg**。")
    st.markdown(f"""
    * **削峰填谷动态定价建议**：建议在 18:00-21:00 电网负荷高峰期上调服务费 0.15元/kWh，在 00:00-07:00 低谷期发放电价折扣券。
    * **碳普惠积分激励策略**：针对响应错峰调度、在谷时充电的车主，额外奖励 **1.5倍 个人碳积分**（可用于抵扣停车费或兑换充电券），预计可转移 **{st_peak*0.25:.1f}%** 的高峰负荷。
    """)
