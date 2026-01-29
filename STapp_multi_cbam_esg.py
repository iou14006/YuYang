
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime as dt

# =========================
# Skybit-PI Command Center
# Multi-Workorder + ERP Cost + Carbon Finance (CBAM/ESG) (Demo)
# =========================

st.set_page_config(page_title="YUYUANG Skybit-PI Command Center", layout="wide")

# ---------- Demo data generators ----------
def generate_demo_workorders(n: int = 18, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    now = dt.datetime.now()

    styles = ["JC40S+30D全襯彈力布", "T/C 65/35 斜紋布", "尼龍彈力布", "機能防潑水布", "吸濕排汗布"]
    lines = ["LINE-A", "LINE-B", "LINE-C"]
    customers = ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E"]

    rows = []
    for i in range(n):
        wo = f"{3017000+i:08d}"
        so = f"SO-{now:%y%m}-{1000+i}"
        line = rng.choice(lines)
        style = rng.choice(styles)
        customer = rng.choice(customers)

        plan_m = int(rng.integers(1500, 6500))
        done_m = int(rng.integers(200, plan_m-100))
        speed = float(rng.uniform(25, 65))  # m/min
        target_temp = float(rng.uniform(160, 178))
        actual_temp = float(target_temp + rng.normal(0, 6))
        esp = float(np.clip(rng.normal(92, 3), 80, 98))

        due = now + dt.timedelta(hours=int(rng.integers(8, 72)))
        sell_price = float(rng.uniform(22.5, 29.0))  # NT$/m

        # demo metadata for ESG/CBAM reporting
        plant = rng.choice(["TW-Plant-01", "TW-Plant-02"])
        product_group = rng.choice(["Knit", "Woven"])
        incoterms = rng.choice(["FOB", "CIF", "DAP"])
        market = rng.choice(["EU", "US", "JP", "TW"])

        rows.append(dict(
            wo=wo, so=so, line=line, customer=customer, style=style,
            plan_m=plan_m, done_m=done_m, speed_mmin=speed,
            target_temp=target_temp, actual_temp=actual_temp,
            esp=esp, due=due, sell_price=sell_price,
            plant=plant, product_group=product_group, incoterms=incoterms, market=market
        ))
    return pd.DataFrame(rows)

def calc_cost_and_status(
    df: pd.DataFrame,
    elec_price_nt_per_kwh: float,
    ef_kgco2_per_kwh: float,
    carbon_price_nt_per_t: float,
    labor_nt_per_hr: float,
    machine_nt_per_hr: float,
) -> pd.DataFrame:
    out = df.copy()

    # --- unit energy model (demo) ---
    temp_dev = (out["actual_temp"] - out["target_temp"]).abs()
    kwh_per_m = 0.12 + temp_dev * 0.008  # demo function: base + penalty
    energy_nt_per_m = kwh_per_m * elec_price_nt_per_kwh

    # --- convert speed to m/hr ---
    m_per_hr = out["speed_mmin"] * 60.0
    labor_nt_per_m = labor_nt_per_hr / m_per_hr
    machine_nt_per_m = machine_nt_per_hr / m_per_hr

    # overhead + quality/risk penalty (demo)
    esp_penalty = np.clip((90 - out["esp"]) / 100.0, 0, 0.2)
    overhead_nt_per_m = 0.55 + (esp_penalty * 3.0)

    # process deviation loss
    deviation_loss_nt_per_m = temp_dev * 0.18

    unit_cost_nt_per_m = (
        energy_nt_per_m
        + labor_nt_per_m
        + machine_nt_per_m
        + overhead_nt_per_m
        + deviation_loss_nt_per_m
    )

    # carbon (factory gate-to-gate electricity only, demo)
    kgco2_per_m = kwh_per_m * ef_kgco2_per_kwh
    internal_carbon_nt_per_m = (kgco2_per_m / 1000.0) * carbon_price_nt_per_t

    # profit
    profit_nt_per_m = out["sell_price"] - unit_cost_nt_per_m
    profit_with_internal_carbon_nt_per_m = out["sell_price"] - (unit_cost_nt_per_m + internal_carbon_nt_per_m)

    # schedule
    remain_m = (out["plan_m"] - out["done_m"]).clip(lower=0)
    eta_hr = remain_m / m_per_hr.replace(0, np.nan)

    # OTD status
    now = dt.datetime.now()
    eta_finish = now + pd.to_timedelta(eta_hr.fillna(0), unit="h")
    slack_hr = (out["due"] - eta_finish).dt.total_seconds() / 3600.0

    def otd_label(x):
        if x >= 2:
            return "🟢 準交"
        if x >= -2:
            return "🟡 風險"
        return "🔴 逾期"

    out["kwh_per_m"] = kwh_per_m
    out["unit_cost_nt_per_m"] = unit_cost_nt_per_m
    out["carbon_kgco2_per_m"] = kgco2_per_m
    out["internal_carbon_nt_per_m"] = internal_carbon_nt_per_m
    out["profit_nt_per_m"] = profit_nt_per_m
    out["profit_with_internal_carbon_nt_per_m"] = profit_with_internal_carbon_nt_per_m
    out["remain_m"] = remain_m
    out["eta_hr"] = eta_hr
    out["otd"] = slack_hr.apply(otd_label)
    out["profit_nt_per_hr"] = profit_nt_per_m * m_per_hr

    return out

def compute_cbam_esg_finance(
    df: pd.DataFrame,
    cbam_enabled: bool,
    cbam_price_eur_per_t: float,
    eur_twd: float,
    cbam_coverage_ratio: float,
    cbam_admin_fee_nt_per_order: float,
    baseline_kgco2_per_m: float,
    green_discount_bps: float,
    order_value_nt_per_m: float,
) -> pd.DataFrame:
    out = df.copy()

    # EU CBAM "certificate-like" cost (demo placeholder): kgCO2 -> tCO2, * price, * coverage
    cbam_price_nt_per_t = cbam_price_eur_per_t * eur_twd
    out["cbam_price_nt_per_t"] = cbam_price_nt_per_t

    cbam_nt_per_m = (out["carbon_kgco2_per_m"] / 1000.0) * cbam_price_nt_per_t * cbam_coverage_ratio
    if not cbam_enabled:
        cbam_nt_per_m = cbam_nt_per_m * 0.0

    out["cbam_nt_per_m"] = cbam_nt_per_m

    # allocate a flat admin fee to each order line (demo): fee / planned meters
    out["cbam_admin_nt_per_m"] = cbam_admin_fee_nt_per_order / out["plan_m"].clip(lower=1)

    # ESG score (demo): lower intensity => higher score
    # score ~ 50..95 based on ratio vs baseline
    ratio = (out["carbon_kgco2_per_m"] / baseline_kgco2_per_m).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    score = 90 - (ratio - 1.0) * 25
    out["esg_score"] = np.clip(score, 50, 95)

    # Green finance benefit (demo): if intensity below baseline, get interest discount on working capital
    # We model as "benefit per m" = order value * discount_rate (bps) * intensity_gap_factor
    gap = (1.0 - ratio).clip(lower=0)  # only rewards improvements
    discount_rate = (green_discount_bps / 10000.0) * gap  # bps -> decimal
    out["green_finance_benefit_nt_per_m"] = order_value_nt_per_m * discount_rate

    # profit layers
    out["profit_with_cbam_nt_per_m"] = out["profit_with_internal_carbon_nt_per_m"] - out["cbam_nt_per_m"] - out["cbam_admin_nt_per_m"]
    out["profit_net_finance_nt_per_m"] = out["profit_with_cbam_nt_per_m"] + out["green_finance_benefit_nt_per_m"]

    return out

# ---------- Sidebar controls ----------
st.sidebar.title("⚙️ 參數設定 (Demo)")
ar_mode = st.sidebar.checkbox("👓 開啟 AR 疊加資訊", value=False)

st.sidebar.markdown("#### 成本參數")
elec_price = st.sidebar.number_input("電價 (NT$/kWh)", min_value=1.0, max_value=10.0, value=3.2, step=0.1)
labor_hr = st.sidebar.number_input("人工成本 (NT$/hr)", min_value=200.0, max_value=1200.0, value=520.0, step=10.0)
machine_hr = st.sidebar.number_input("機台折舊/維護 (NT$/hr)", min_value=200.0, max_value=2000.0, value=760.0, step=20.0)

st.sidebar.markdown("#### 內部碳價（管理/ESG）")
ef_kwh = st.sidebar.number_input("排放係數 (kgCO2/kWh)", min_value=0.05, max_value=1.2, value=0.52, step=0.01)
internal_carbon_price = st.sidebar.number_input("內部碳價 (NT$/tCO2e)", min_value=0.0, max_value=8000.0, value=1200.0, step=50.0)

st.sidebar.markdown("#### CBAM / 金融化參數（示意）")
cbam_enabled = st.sidebar.checkbox("啟用 CBAM 成本情境 (Demo)", value=True)
cbam_price_eur = st.sidebar.number_input("CBAM 碳價 (EUR/tCO2e)", min_value=0.0, max_value=300.0, value=85.0, step=1.0)
eur_twd = st.sidebar.number_input("匯率 (TWD/EUR)", min_value=20.0, max_value=50.0, value=34.5, step=0.1)
cbam_coverage = st.sidebar.slider("CBAM 覆蓋比例（free allocation/適用比例）", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
cbam_admin_fee = st.sidebar.number_input("CBAM 申報/稽核/管理費 (NT$/訂單)", min_value=0.0, max_value=20000.0, value=2800.0, step=100.0)

st.sidebar.markdown("#### 綠色金融（示意）")
baseline_intensity = st.sidebar.number_input("基準碳強度 (kgCO2/m)", min_value=0.001, max_value=1.0, value=0.08, step=0.005)
green_discount_bps = st.sidebar.number_input("利率折減上限 (bps)", min_value=0.0, max_value=300.0, value=60.0, step=5.0)
order_value_nt_per_m = st.sidebar.number_input("訂單價值基底 (NT$/m，用於融資折讓)", min_value=0.0, max_value=200.0, value=26.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.info("💡 Demo：CBAM/ESG/金融化為『情境模擬』；正式版需接：產品邊界、排放因子版本、供應鏈材料、稽核軌跡。")

# ---------- Data ----------
if "wo_df" not in st.session_state:
    st.session_state.wo_df = generate_demo_workorders()

wo_df = st.session_state.wo_df
wo_live = calc_cost_and_status(wo_df, elec_price, ef_kwh, internal_carbon_price, labor_hr, machine_hr)
wo_live = compute_cbam_esg_finance(
    wo_live,
    cbam_enabled=cbam_enabled,
    cbam_price_eur_per_t=cbam_price_eur,
    eur_twd=eur_twd,
    cbam_coverage_ratio=cbam_coverage,
    cbam_admin_fee_nt_per_order=cbam_admin_fee,
    baseline_kgco2_per_m=baseline_intensity,
    green_discount_bps=green_discount_bps,
    order_value_nt_per_m=order_value_nt_per_m,
)

# ---------- Header ----------
st.title("🏭 裕源紡織：Skybit-PI 智能決策戰情室")
tabs = st.tabs([
    "現場執行面板 (Live)",
    "多工單即時列表 (Portfolio)",
    "模型統計分析 (Analytics)",
    "碳成本 + 金融化（CBAM / ESG）",
    "AR 盈虧體驗 (Experience)"
])

# ============= TAB: Live =============
with tabs[0]:
    st.subheader("🛠️ 現場執行即時監控 (MES Integrated)")

    current = wo_live.iloc[0].to_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("當前工單", current["wo"])
    c2.metric("碼表讀數", f"{int(current['done_m'])} M", "正常生產")
    c3.metric("烘箱溫度", f"{current['actual_temp']:.1f} °C", f"目標 {current['target_temp']:.1f}°C", delta_color="inverse")
    c4.metric("即時盈虧估算", f"NT$ {current['profit_nt_per_hr']:.0f}/hr", current["otd"], delta_color="inverse")

    st.write("---")
    st.markdown("#### 工單決策級狀態 (即時)")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("即時單位成本", f"NT$ {current['unit_cost_nt_per_m']:.2f} / m")
    s2.metric("每米盈虧", f"NT$ {current['profit_nt_per_m']:.2f} / m")
    s3.metric("ETA", f"{current['eta_hr']:.1f} hr", f"剩餘 {int(current['remain_m'])} m")
    s4.metric("內部碳成本", f"NT$ {current['internal_carbon_nt_per_m']:.2f} / m", f"{current['carbon_kgco2_per_m']:.3f} kgCO₂/m")

    st.write("---")
    st.markdown("#### 碳成本 + 金融化（即時）")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("CBAM 成本", f"NT$ {current['cbam_nt_per_m']:.2f} / m")
    f2.metric("CBAM 管理費分攤", f"NT$ {current['cbam_admin_nt_per_m']:.2f} / m")
    f3.metric("綠色金融折讓", f"+NT$ {current['green_finance_benefit_nt_per_m']:.2f} / m")
    f4.metric("ESG 分數 (示意)", f"{current['esg_score']:.0f} / 100")

    st.write("---")
    p1, p2, p3 = st.columns(3)
    p1.info(f"布種：{current['style']}｜市場：{current['market']}")
    p2.warning(f"訂單：{current['so']}｜計畫：{int(current['plan_m'])} m｜線別：{current['line']}｜廠別：{current['plant']}")
    p3.success(f"ESP 效率：{current['esp']:.1f}%｜客戶：{current['customer']}")

# ============= TAB: Portfolio =============
with tabs[1]:
    st.subheader("📦 多工單即時列表 (Portfolio)")

    f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
    with f1:
        line_filter = st.selectbox("線別", ["ALL"] + sorted(wo_live["line"].unique().tolist()))
    with f2:
        otd_filter = st.selectbox("準交狀態", ["ALL", "🟢 準交", "🟡 風險", "🔴 逾期"])
    with f3:
        sort_by = st.selectbox("排序", ["profit_net_finance_nt_per_m", "profit_nt_per_m", "profit_nt_per_hr", "eta_hr", "unit_cost_nt_per_m"])
    with f4:
        q = st.text_input("搜尋 (工單/訂單/客戶/布種)")

    view = wo_live.copy()
    if line_filter != "ALL":
        view = view[view["line"] == line_filter]
    if otd_filter != "ALL":
        view = view[view["otd"] == otd_filter]
    if q:
        ql = q.lower()
        mask = (
            view["wo"].str.lower().str.contains(ql) |
            view["so"].str.lower().str.contains(ql) |
            view["customer"].str.lower().str.contains(ql) |
            view["style"].str.lower().str.contains(ql)
        )
        view = view[mask]

    view = view.sort_values(sort_by, ascending=(sort_by in ["eta_hr", "unit_cost_nt_per_m"]))

    cols = {
        "otd": "準交",
        "wo": "工單",
        "so": "訂單",
        "line": "線別",
        "market": "市場",
        "customer": "客戶",
        "style": "布種",
        "done_m": "已生產(m)",
        "plan_m": "計畫(m)",
        "remain_m": "剩餘(m)",
        "eta_hr": "ETA(hr)",
        "unit_cost_nt_per_m": "成本(NT$/m)",
        "profit_nt_per_m": "盈虧(NT$/m)",
        "internal_carbon_nt_per_m": "內部碳(NT$/m)",
        "cbam_nt_per_m": "CBAM(NT$/m)",
        "green_finance_benefit_nt_per_m": "綠金折讓(NT$/m)",
        "profit_net_finance_nt_per_m": "淨盈虧(NT$/m)",
    }

    show = view[list(cols.keys())].rename(columns=cols).copy()
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.caption("提示：下方選一張工單，查看『分層損益（含碳/CBAM/綠金）』與成本桶分解。")
    wo_pick = st.selectbox("選擇工單查看細節", view["wo"].tolist(), index=0 if len(view) else None)

    if len(view):
        row = view[view["wo"] == wo_pick].iloc[0]

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("準交狀態", row["otd"], f"ETA {row['eta_hr']:.1f} hr")
        a2.metric("每米盈虧", f"NT$ {row['profit_nt_per_m']:.2f}/m")
        a3.metric("含碳+CBAM後", f"NT$ {row['profit_with_cbam_nt_per_m']:.2f}/m")
        a4.metric("淨盈虧(含綠金)", f"NT$ {row['profit_net_finance_nt_per_m']:.2f}/m")

        # Cost bucket breakdown (demo reconstruction)
        temp_dev = abs(row["actual_temp"] - row["target_temp"])
        kwh_per_m = row["kwh_per_m"]
        energy = kwh_per_m * elec_price
        mhr = row["speed_mmin"] * 60.0
        labor = labor_hr / mhr
        machine = machine_hr / mhr
        overhead = 0.55 + (max(0, (90 - row["esp"])) / 100.0) * 3.0
        deviation = temp_dev * 0.18

        bucket = pd.DataFrame({
            "bucket": ["能源", "人工", "機台", "製造費用", "偏差損失"],
            "nt_per_m": [energy, labor, machine, overhead, deviation]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bucket["bucket"], y=bucket["nt_per_m"], name="成本桶 (NT$/m)"))
        fig.update_layout(title=f"成本歸屬分解（工單 {row['wo']}）", yaxis_title="NT$/m")
        st.plotly_chart(fig, use_container_width=True)

        # Profit waterfall (layers)
        wf = pd.DataFrame({
            "layer": ["售價", "生產成本", "內部碳", "CBAM", "CBAM管理費", "綠金折讓", "淨盈虧"],
            "value": [
                row["sell_price"],
                -row["unit_cost_nt_per_m"],
                -row["internal_carbon_nt_per_m"],
                -row["cbam_nt_per_m"],
                -row["cbam_admin_nt_per_m"],
                row["green_finance_benefit_nt_per_m"],
                row["profit_net_finance_nt_per_m"],
            ]
        })

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=wf["layer"], y=wf["value"], name="NT$/m"))
        fig2.update_layout(title="分層損益（NT$/m）— 生產 → 內部碳 → CBAM → 綠金", yaxis_title="NT$/m")
        st.plotly_chart(fig2, use_container_width=True)

# ============= TAB: Analytics =============
with tabs[2]:
    st.subheader("📊 模型統計與黃金工藝分析（Demo）")

    agg = wo_live.copy()
    agg["speed_bin"] = pd.cut(agg["speed_mmin"], bins=[20, 30, 40, 50, 60, 70])
    g = agg.groupby("speed_bin", observed=True).agg(
        profit_mean=("profit_nt_per_m", "mean"),
        cost_mean=("unit_cost_nt_per_m", "mean"),
        net_mean=("profit_net_finance_nt_per_m", "mean"),
        count=("wo", "count")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["speed_bin"].astype(str), y=g["profit_mean"], name="平均每米盈虧（未含碳）"))
    fig.add_trace(go.Scatter(x=g["speed_bin"].astype(str), y=g["net_mean"], name="平均每米淨盈虧（含碳/CBAM/綠金）"))
    fig.add_trace(go.Bar(x=g["speed_bin"].astype(str), y=g["cost_mean"], name="平均單位成本", opacity=0.35))
    fig.update_layout(title="車速分箱 vs 成本/盈虧（Portfolio 統計）", xaxis_title="車速區間 (m/min)")
    st.plotly_chart(fig, use_container_width=True)

# ============= TAB: Carbon Finance =============
with tabs[3]:
    st.subheader("🌍 碳成本 + 金融化（CBAM / ESG 版）")

    st.markdown("#### 1) 工單碳強度與成本（gate-to-gate 示意）")
    cdf = wo_live[[
        "wo", "so", "plant", "market", "product_group",
        "kwh_per_m", "carbon_kgco2_per_m",
        "internal_carbon_nt_per_m",
        "cbam_nt_per_m", "cbam_admin_nt_per_m",
        "esg_score", "green_finance_benefit_nt_per_m",
        "profit_nt_per_m", "profit_net_finance_nt_per_m"
    ]].copy()

    show = cdf.rename(columns={
        "wo": "工單", "so": "訂單", "plant": "廠別", "market": "市場", "product_group": "品類",
        "kwh_per_m": "kWh/m", "carbon_kgco2_per_m": "kgCO₂/m",
        "internal_carbon_nt_per_m": "內部碳(NT$/m)",
        "cbam_nt_per_m": "CBAM(NT$/m)", "cbam_admin_nt_per_m": "CBAM管理費(NT$/m)",
        "esg_score": "ESG分數", "green_finance_benefit_nt_per_m": "綠金折讓(NT$/m)",
        "profit_nt_per_m": "盈虧(NT$/m)", "profit_net_finance_nt_per_m": "淨盈虧(NT$/m)"
    })

    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("#### 2) Portfolio 總覽：碳費用與財務影響")
    total_m = wo_live["plan_m"].sum()
    avg_intensity = wo_live["carbon_kgco2_per_m"].mean()
    avg_internal = wo_live["internal_carbon_nt_per_m"].mean()
    avg_cbam = (wo_live["cbam_nt_per_m"] + wo_live["cbam_admin_nt_per_m"]).mean()
    avg_green = wo_live["green_finance_benefit_nt_per_m"].mean()
    avg_profit = wo_live["profit_nt_per_m"].mean()
    avg_net = wo_live["profit_net_finance_nt_per_m"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("平均碳強度", f"{avg_intensity:.3f} kgCO₂/m")
    k2.metric("平均內部碳成本", f"NT$ {avg_internal:.2f}/m")
    k3.metric("平均 CBAM+管理費", f"NT$ {avg_cbam:.2f}/m")
    k4.metric("平均綠金折讓", f"+NT$ {avg_green:.2f}/m")
    k5.metric("平均淨盈虧", f"NT$ {avg_net:.2f}/m", f"未含碳：{avg_profit:.2f}/m")

    st.markdown("#### 3) 匯出（供 CBAM/品牌 PCF/金融機構回填）")
    report = wo_live.copy()
    report["boundary"] = "gate-to-gate (electricity only) - DEMO"
    report["ef_version"] = "EF-DEMO-v1"
    report["data_quality"] = "A (simulated)"

    export_cols = [
        "wo", "so", "plant", "market", "product_group", "incoterms",
        "plan_m", "kwh_per_m", "carbon_kgco2_per_m",
        "internal_carbon_nt_per_m", "cbam_nt_per_m", "cbam_admin_nt_per_m",
        "esg_score", "green_finance_benefit_nt_per_m",
        "boundary", "ef_version", "data_quality"
    ]
    out_csv = report[export_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ 匯出 CBAM/ESG/綠金回填清單 (CSV)", out_csv, file_name="cbam_esg_finance_report_demo.csv", mime="text/csv")

    st.info(
    """備註：
本頁的 CBAM/ESG/綠金為「情境金融化」展示。
正式導入時，會把：
- 材料 / 運輸 / 外包
- 排放因子版本控管
- 稽核軌跡
- 產品邊界（PCF）
完整整合進來。
"""
)
# ============= TAB: Experience (AR) =============
with tabs[4]:
    if ar_mode:
        st.header("👓 AR 眼鏡即時視野（Demo）")
        st.write("在現場機台或布捲上疊加：工單、每米盈虧、準交、碳成本、CBAM、綠金折讓。")
        st.image(
            "https://img.freepik.com/free-photo/smart-factory-concept-with-ar-glasses_23-2149171724.jpg",
            caption="AR 虛擬疊加（示意）：🟢 準交 ｜ 淨盈虧 +2.1 NT$/m ｜ 碳強度 0.07 kg/m"
        )
    else:
        st.info("請在側邊欄開啟 AR 模式，以展示『未來工廠管理』樣貌。")

st.markdown("---")
if st.button("🏁 產出訂單獲利與碳中和報告 (爭取訂單專用)"):
    st.balloons()
    st.success("報告已生成（Demo）：已包含即時成本歸屬、準交、每米盈虧、內部碳、CBAM與綠金情境。")
