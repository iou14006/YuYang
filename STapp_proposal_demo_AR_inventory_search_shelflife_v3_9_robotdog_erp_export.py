import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from io import BytesIO

# PDF export (schedule suggestion)
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _REPORTLAB_OK = True
except Exception:
    _REPORTLAB_OK = False


# ==========================================================
# YuYang / Skybit-PI — Stenter Customer Proposal Demo
# Two on-site pages -> Decision Layer (Profit/OTD/Carbon)
# + Physics (Steam/Exhaust T&RH/Airflow) + Event Timeline
# ==========================================================

st.set_page_config(page_title="YuYang Proposal Demo", layout="wide")

# -----------------------------
# PDF: Schedule Suggestion Export
# -----------------------------
def _find_cjk_font_paths() -> list:
    """Return candidate CJK font file paths for PDF export (cross-platform)."""
    cand = []
    # Windows common fonts
    cand += [
        r"C:\Windows\Fonts\msjh.ttc",  # Microsoft JhengHei
        r"C:\Windows\Fonts\msjhbd.ttc",
        r"C:\Windows\Fonts\msyh.ttc",  # Microsoft YaHei
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    # Linux common fonts (container / servers)
    cand += [
        "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
        "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
    ]
    return [p for p in cand if os.path.exists(p)]

def _register_cjk_font() -> tuple[str, str]:
    """Register a CJK font for ReportLab and return (regular, bold) font names."""
    if not _REPORTLAB_OK:
        return ("Helvetica", "Helvetica-Bold")

    # If already registered, reuse
    if "CJK" in pdfmetrics.getRegisteredFontNames():
        return ("CJK", "CJK-Bold")

    font_paths = _find_cjk_font_paths()
    # Prefer TTF over TTC/OTF for maximum compatibility
    preferred = []
    for p in font_paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".ttf":
            preferred.append(p)
    font_paths = preferred + [p for p in font_paths if p not in preferred]

    reg_name = "CJK"
    bold_name = "CJK-Bold"
    # Try register first workable font
    for p in font_paths:
        try:
            pdfmetrics.registerFont(TTFont(reg_name, p, subfontIndex=0))
            # Use same face as bold if no bold available
            pdfmetrics.registerFont(TTFont(bold_name, p, subfontIndex=0))
            return (reg_name, bold_name)
        except Exception:
            continue

    # fallback
    return ("Helvetica", "Helvetica-Bold")

def build_schedule_pdf_from_queue(
    queue_df: pd.DataFrame,
    now_ts: dt.datetime | None = None,
    rate_m_per_hr: float = 600.0,
    changeover_hr_same_line: float = 0.25,
) -> bytes:
    """Build a PDF (bytes) of schedule suggestion table from the priority queue."""
    if now_ts is None:
        now_ts = dt.datetime.now()

    if queue_df is None or len(queue_df) == 0:
        return b""

    # defensive copy and normalize types
    q = queue_df.copy()
    if "due" in q.columns:
        q["due"] = pd.to_datetime(q["due"], errors="coerce")

    # Ensure key columns exist
    for c in ["score","wo","flow_card","line","customer","otd","due","shelf_age_days","inventory_m",
              "shelf_loss_nt_per_m","profit_with_carbon_and_shelf_nt_per_m","total_quality_loss_nt","reasons"]:
        if c not in q.columns:
            q[c] = np.nan

    # Sort: score desc then due asc
    q = q.sort_values(["score","due"], ascending=[False, True], na_position="last").reset_index(drop=True)

    # Estimate start/finish per line (simple line-wise sequential model)
    line_state = {}
    for ln in q["line"].dropna().unique().tolist():
        line_state[ln] = {"t": now_ts, "has_job": False}

    est_start_list, est_finish_list, eta_status_list, reason_list, chg_cost_list = [], [], [], [], []

    for _, r in q.iterrows():
        ln = r.get("line", "UNKNOWN")
        if ln not in line_state:
            line_state[ln] = {"t": now_ts, "has_job": False}
        stt = line_state[ln]
        chg_hr = changeover_hr_same_line if stt["has_job"] else 0.0

        start = stt["t"] + dt.timedelta(hours=float(chg_hr))
        qty = r.get("inventory_m", np.nan)
        if pd.isna(qty) or float(qty) <= 0:
            qty = 1000.0  # demo fallback
        dur_hr = float(qty) / max(float(rate_m_per_hr), 1e-6)
        finish = start + dt.timedelta(hours=dur_hr)

        due = r.get("due", pd.NaT)
        eta_status = "可準交"
        if pd.notna(due) and finish > due.to_pydatetime():
            eta_status = "可能延誤"

        # changeover cost (demo): first job 0, subsequent jobs fixed
        chg_cost = 0 if not stt["has_job"] else 500

        # reasons
        reasons = r.get("reasons", "")
        if not isinstance(reasons, str) or reasons.strip() == "":
            parts = []
            try:
                age = float(r.get("shelf_age_days", np.nan))
                if age >= 30:
                    parts.append("庫齡高")
                elif age >= 20:
                    parts.append("庫齡超標")
            except Exception:
                pass
            if str(r.get("otd", "")) == "有風險":
                parts.append("交期逼近")
            try:
                prof = float(r.get("profit_with_carbon_and_shelf_nt_per_m", np.nan))
                if not np.isnan(prof) and prof < 0:
                    parts.append("含碳+庫齡為負")
            except Exception:
                pass
            reasons = " / ".join(parts) if parts else "常規"

        est_start_list.append(start)
        est_finish_list.append(finish)
        eta_status_list.append(eta_status)
        reason_list.append(reasons)
        chg_cost_list.append(chg_cost)

        stt["t"] = finish
        stt["has_job"] = True

    q["est_start"] = est_start_list
    q["est_finish"] = est_finish_list
    q["eta_status"] = eta_status_list
    q["sched_reason"] = reason_list
    q["changeover_cost_nt"] = chg_cost_list

    reg_font, bold_font = _register_cjk_font()

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName=bold_font, fontSize=18, leading=22, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=bold_font, fontSize=12, leading=14, spaceBefore=6, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=reg_font, fontSize=9, leading=11)

    elems = []
    elems.append(Paragraph("優先排產 - 排程建議表（Demo輸出）", title))
    elems.append(Paragraph(
        f"生成時間：{now_ts.strftime('%Y-%m-%d %H:%M')} ｜假設產能：{rate_m_per_hr:.0f} m/hr ｜同線換線時間：{changeover_hr_same_line:.2f} hr",
        body
    ))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("排序規則與輸出欄位說明", h2))
    elems.append(Paragraph(
        "排序依 Score 由高到低（庫齡超標、品質損失、交期風險、含碳+庫齡盈虧為負等因子加權）。"
        "表內提供每筆工單的排序理由、預估換線成本、預估開工/完工時間與準交判斷，供生管做週排/日排參考。",
        body
    ))
    elems.append(Spacer(1, 8))

    header = ["Rank","WO","流程卡","線別","客戶","交期","準交(現況)","預估開工","預估完工","ETA狀態",
              "庫齡(天)","卷長(m)","品質損失(NT$/m)","含碳+庫齡盈虧(NT$/m)","換線成本(NT$)","Score","排序理由"]
    data = [header]

    for i, r in q.iterrows():
        def _fmt_dt(x):
            try:
                return x.strftime("%m-%d %H:%M")
            except Exception:
                return ""
        data.append([
            str(i + 1),
            str(r.get("wo","")),
            str(r.get("flow_card","")),
            str(r.get("line","")),
            str(r.get("customer","")),
            _fmt_dt(r.get("due", None)) if pd.notna(r.get("due", pd.NaT)) else "",
            str(r.get("otd","")),
            _fmt_dt(r.get("est_start", None)),
            _fmt_dt(r.get("est_finish", None)),
            str(r.get("eta_status","")),
            "" if pd.isna(r.get("shelf_age_days", np.nan)) else f"{int(float(r.get('shelf_age_days')))}",
            "" if pd.isna(r.get("inventory_m", np.nan)) else f"{float(r.get('inventory_m')):.0f}",
            "" if pd.isna(r.get("shelf_loss_nt_per_m", np.nan)) else f"{float(r.get('shelf_loss_nt_per_m')):.4f}",
            "" if pd.isna(r.get("profit_with_carbon_and_shelf_nt_per_m", np.nan)) else f"{float(r.get('profit_with_carbon_and_shelf_nt_per_m')):.4f}",
            "" if pd.isna(r.get("changeover_cost_nt", np.nan)) else f"{int(float(r.get('changeover_cost_nt'))):,}",
            "" if pd.isna(r.get("score", np.nan)) else f"{float(r.get('score')):.3f}",
            str(r.get("sched_reason","")),
        ])

    tbl = Table(data, repeatRows=1)
    ts = TableStyle([
        ("FONT", (0,0), (-1,0), bold_font, 9),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E9EEF6")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#102A43")),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#CBD2D9")),
        ("FONT", (0,1), (-1,-1), reg_font, 8),
        ("ALIGN", (0,1), (0,-1), "CENTER"),
        ("ALIGN", (5,1), (9,-1), "CENTER"),
        ("ALIGN", (10,1), (11,-1), "RIGHT"),
        ("ALIGN", (12,1), (15,-1), "RIGHT"),
        ("ALIGN", (16,1), (16,-1), "LEFT"),
    ])

    # highlight risky rows
    for ridx in range(1, len(data)):
        eta = data[ridx][9]
        try:
            shelf = int(data[ridx][10]) if data[ridx][10] != "" else 0
        except Exception:
            shelf = 0
        try:
            prof = float(data[ridx][13]) if data[ridx][13] != "" else 0.0
        except Exception:
            prof = 0.0
        if eta == "可能延誤":
            ts.add("BACKGROUND", (0, ridx), (-1, ridx), colors.HexColor("#FFF5F5"))
        if shelf >= 30:
            ts.add("TEXTCOLOR", (10, ridx), (10, ridx), colors.HexColor("#B91C1C"))
        if prof < 0:
            ts.add("TEXTCOLOR", (13, ridx), (13, ridx), colors.HexColor("#7C2D12"))

    tbl.setStyle(ts)
    elems.append(tbl)

    doc.build(elems)
    return buf.getvalue()


# -----------------------------
# ERP Export (Excel/PDF) for RobotDog Maintenance Tickets + PR/PO
# -----------------------------

def build_erp_excel(tickets_df: pd.DataFrame, pr_df: pd.DataFrame, po_df: pd.DataFrame) -> bytes:
    """Export ERP artifacts to an Excel workbook with multiple sheets."""
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        (tickets_df if tickets_df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="MaintenanceTickets")
        (pr_df if pr_df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="PR")
        (po_df if po_df is not None else pd.DataFrame()).to_excel(writer, index=False, sheet_name="PO")

        # Simple pivot summary
        if tickets_df is not None and len(tickets_df) > 0:
            s = tickets_df.copy()
            if "created_ts" in s.columns:
                s["created_date"] = pd.to_datetime(s["created_ts"], errors="coerce").dt.date
            piv = s.pivot_table(index=["line"], values=["est_material_cost_nt"], aggfunc="sum", fill_value=0)
            piv.reset_index().to_excel(writer, index=False, sheet_name="Summary")
        else:
            pd.DataFrame([{ "note": "No tickets" }]).to_excel(writer, index=False, sheet_name="Summary")

    return out.getvalue()


def build_erp_pdf(tickets_df: pd.DataFrame, pr_df: pd.DataFrame, po_df: pd.DataFrame, title: str = "ERP 匯出（Maintenance / PR / PO）") -> bytes:
    """Build a compact PDF report for ERP export (tickets + PR + PO)."""
    if not _REPORTLAB_OK:
        return b""

    reg_font, bold_font = _register_cjk_font()

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()
    ttl = ParagraphStyle("ttl", parent=styles["Title"], fontName=bold_font, fontSize=16, leading=20, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=bold_font, fontSize=11, leading=14, spaceBefore=6, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName=reg_font, fontSize=9, leading=11)

    elems = []
    elems.append(Paragraph(title, ttl))
    elems.append(Paragraph(f"產出時間：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))
    elems.append(Spacer(1, 6))

    def _table_from_df(df: pd.DataFrame, max_rows: int = 18):
        if df is None or len(df) == 0:
            return Table([["(empty)"]])
        view = df.copy().head(max_rows)
        # stringify datetimes for PDF
        for c in view.columns:
            if "ts" in c or "date" in c:
                view[c] = pd.to_datetime(view[c], errors="coerce").astype(str)
        data = [list(view.columns)] + view.astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,0), bold_font),
            ("FONTNAME", (0,1), (-1,-1), reg_font),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        return tbl

    elems.append(Paragraph("Maintenance Tickets", h2))
    elems.append(_table_from_df(tickets_df, max_rows=16))
    elems.append(Spacer(1, 8))

    elems.append(Paragraph("PR（請購）", h2))
    elems.append(_table_from_df(pr_df, max_rows=18))
    elems.append(Spacer(1, 8))

    elems.append(Paragraph("PO（採購單）", h2))
    elems.append(_table_from_df(po_df, max_rows=18))

    doc.build(elems)
    return buf.getvalue()

# -----------------------------
# Psychrometrics (engineering approximation)
# -----------------------------
def p_ws_tetens_pa(T_c: float) -> float:
    # Saturation vapor pressure (Pa), Tetens approximation
    return 610.94 * np.exp((17.625 * T_c) / (T_c + 243.04))

def humidity_ratio_w(T_c: float, RH_0to1: float, p_atm_pa: float = 101325.0) -> float:
    RH = float(np.clip(RH_0to1, 0.0, 1.0))
    p_ws = p_ws_tetens_pa(float(T_c))
    p_v = RH * p_ws
    return 0.62198 * (p_v / max(p_atm_pa - p_v, 1e-6))

def dryer_physics(
    speed_mmin: float,
    steam_kgph: float,
    airflow_m3ph: float,
    t_in_c: float,
    rh_in: float,
    t_out_c: float,
    rh_out: float,
    rho_air: float,
    p_atm: float,
    h_fg_kjkg: float,
    h_steam_kjkg: float,
    cp_air_kjkgk: float,
    eta_base: float,
) -> dict:
    """
    Physics-inspired (auditable) model:
    - moisture removal estimated from inlet/outlet humidity ratio & airflow
    - drying heat requirement = latent + sensible / efficiency
    - steam heat supply from steam mass flow and enthalpy drop
    """
    v_hr = max(speed_mmin * 60.0, 1e-6)  # m/hr
    steam_kgs = steam_kgph / 3600.0
    vdot_m3s = airflow_m3ph / 3600.0

    w_in = humidity_ratio_w(t_in_c, rh_in, p_atm)
    w_out = humidity_ratio_w(t_out_c, rh_out, p_atm)

    m_da_dot = rho_air * vdot_m3s               # kg_dry_air/s (approx)
    m_w_dot = m_da_dot * max(0.0, (w_out - w_in))  # kg_water/s

    # base latent + sensible (kW)
    q_evap_kw = m_w_dot * h_fg_kjkg
    q_air_kw = m_da_dot * cp_air_kjkgk * max(0.0, (t_out_c - t_in_c))

    # effective efficiency degrades when exhaust RH is high (driving force reduced)
    eta_eff = eta_base * (1.0 - 0.60 * max(0.0, (rh_out - 0.75)))  # demo degradation
    eta_eff = float(np.clip(eta_eff, 0.20, 0.95))

    q_req_kw = (q_evap_kw + q_air_kw) / max(eta_eff, 1e-6)
    q_steam_kw = steam_kgs * h_steam_kjkg

    kwh_th_per_m_req = q_req_kw / v_hr
    kwh_th_per_m_steam = q_steam_kw / v_hr

    steam_util = q_req_kw / max(q_steam_kw, 1e-6)  # >1 implies insufficient supply or losses

    return dict(
        w_in=w_in, w_out=w_out,
        m_w_dot_kgps=m_w_dot,
        q_evap_kw=q_evap_kw,
        q_air_kw=q_air_kw,
        eta_eff=eta_eff,
        q_req_kw=q_req_kw,
        q_steam_kw=q_steam_kw,
        kwh_th_per_m_req=kwh_th_per_m_req,
        kwh_th_per_m_steam=kwh_th_per_m_steam,
        steam_util=steam_util,
    )

# -----------------------------
# Demo data aligned to on-site pages
# -----------------------------
def generate_demo_orders(n: int = 14, seed: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    now = dt.datetime.now()

    recipes = [f"25B{rng.integers(8000, 8999)}A" for _ in range(n)]
    wos = [f"{rng.integers(25000000, 25999999)}" for _ in range(n)]
    markets = rng.choice(["EU", "US", "JP", "TW"], size=n)
    customers = rng.choice(["Brand A", "Brand B", "Brand C", "Brand D"], size=n)
    lines = rng.choice(["LINE-A", "LINE-B", "LINE-C", "LINE-D"], size=n)
    flow_cards = [f"03{rng.integers(0,999999):06d}" for _ in range(n)]

    rows = []
    for i in range(n):
        speed = float(rng.uniform(20.0, 45.0))      # m/min
        zone_set = float(rng.uniform(170, 195))
        zone_temps = (zone_set + rng.normal(0, 2.5, size=8)).clip(155, 210)

        fan_hz = (35 + rng.normal(0, 1.2, size=8)).clip(25, 50)
        width_mm = (rng.normal(1655, 4.5, size=8)).clip(1540, 1700)

        ex_f_hz = float(np.clip(40 + rng.normal(0, 2.0), 25, 55))
        ex_b_hz = float(np.clip(40 + rng.normal(0, 2.0), 25, 55))
        cool_hz = float(np.clip(30 + rng.normal(0, 1.5), 15, 45))

        voltage = float(np.clip(380 + rng.normal(0, 6), 350, 410))
        current = float(np.clip(120 + rng.normal(0, 15), 60, 200))
        power_kw = float(max(15.0, (voltage * current * 0.85) / 1000.0))

        steam_kgph = float(rng.uniform(700, 1600))
        airflow_m3ph = float(rng.uniform(9000, 26000))
        t_in = float(rng.uniform(24, 34))
        rh_in = float(rng.uniform(0.45, 0.75))
        t_out = float(rng.uniform(58, 88))
        rh_out = float(rng.uniform(0.60, 0.95))

        plan_m = int(rng.integers(1800, 7000))
        done_m = int(rng.integers(300, plan_m - 100))
        due = now + dt.timedelta(hours=int(rng.integers(6, 72)))

        sell_price = float(rng.uniform(22.5, 29.0))

        row = dict(
            ts=now,
            line=str(lines[i]),
            flow_card=str(flow_cards[i]),
            start_ts=now - dt.timedelta(hours=float(rng.uniform(0.5, 10.0))),
            wo=wos[i],
            barcode=recipes[i],
            recipe=recipes[i],
            market=markets[i],
            customer=customers[i],
            speed_mmin=speed,
            plan_m=plan_m,
            done_m=done_m,
            due=due,
            exhaust_front_hz=ex_f_hz,
            exhaust_back_hz=ex_b_hz,
            cooling_hz=cool_hz,
            voltage_v=voltage,
            current_a=current,
            power_kw=power_kw,
            steam_kgph=steam_kgph,
            airflow_m3ph=airflow_m3ph,
            inlet_temp_c=t_in,
            inlet_rh=rh_in,
            exhaust_temp_c=t_out,
            exhaust_rh=rh_out,
            sell_price_nt_per_m=sell_price,
        )
        # zone fields (match the 1..8 layout)
        for z in range(1, 9):
            row[f"zone_temp_{z}"] = float(zone_temps[z-1])
            row[f"fan_hz_{z}"] = float(fan_hz[z-1])
            row[f"width_mm_{z}"] = float(width_mm[z-1])
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# AR (Accounts Receivable) — proposal-friendly cash view
# -----------------------------
def generate_demo_ar(orders: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Create demo AR entries based on orders.
    In real deployment, replace with ERP AR / invoice tables.
    """
    rng = np.random.default_rng(seed)
    now = dt.datetime.now()

    # Payment terms by market (demo)
    term_days_map = {"EU": 60, "US": 45, "JP": 60, "TW": 30}

    rows = []
    for r in orders.itertuples(index=False):
        # Assume invoice at partial progress for demo; multiple invoices possible in real ERP
        invoiced_m = int(max(200, min(r.done_m, r.plan_m) * rng.uniform(0.70, 0.95)))
        invoice_amount = float(invoiced_m * r.sell_price_nt_per_m)

        # Partial received
        paid_ratio = float(np.clip(rng.normal(0.35, 0.25), 0.0, 0.95))
        paid_amount = invoice_amount * paid_ratio
        ar_amount = max(0.0, invoice_amount - paid_amount)

        term_days = int(term_days_map.get(r.market, 45))
        invoice_date = now - dt.timedelta(days=int(rng.integers(5, 50)))
        due_date = invoice_date + dt.timedelta(days=term_days)

        # Some overdue cases
        if rng.random() < 0.35:
            due_date = now - dt.timedelta(days=int(rng.integers(1, 35)))

        days_overdue = int((now - due_date).days)
        days_to_due = int((due_date - now).days)

        # Risk score: tie to profit + overdue + market
        market_risk = {"EU": 0.35, "US": 0.25, "JP": 0.20, "TW": 0.15}.get(r.market, 0.25)
        risk = 0.30 * market_risk + 0.45 * (1.0 if days_overdue > 0 else 0.0)
        risk = float(np.clip(risk + rng.normal(0, 0.05), 0.0, 1.0))

        if days_overdue > 30:
            bucket = "90+"
        elif days_overdue > 0:
            bucket = "1-30"
        elif days_to_due <= 7:
            bucket = "0-7"
        else:
            bucket = "current"

        rows.append(dict(
            wo=r.wo,
            barcode=r.barcode,
            customer=r.customer,
            market=r.market,
            invoice_no=f"INV-{str(r.wo)[-6:]}-{int(rng.integers(10,99))}",
            invoice_date=invoice_date,
            due_date=due_date,
            term_days=term_days,
            invoiced_m=invoiced_m,
            invoice_amount_nt=invoice_amount,
            paid_amount_nt=paid_amount,
            ar_amount_nt=ar_amount,
            days_overdue=days_overdue,
            bucket=bucket,
            risk_score=risk,
        ))

    ar = pd.DataFrame(rows)

    def risk_label(x: float) -> str:
        if x >= 0.70: return "🔴 高風險"
        if x >= 0.45: return "🟡 注意"
        return "🟢 正常"
    ar["risk"] = ar["risk_score"].apply(risk_label)

    return ar


# -----------------------------
# Inventory / Shelf-life — fabric aging cost (proposal-friendly)
# -----------------------------
def generate_demo_inventory(orders: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Create demo inventory rolls for shelf-life management.
    In real deployment, replace with WMS/ERP receiving records.
    """
    rng = np.random.default_rng(seed)
    now = dt.datetime.now()

    rows = []
    for r in orders.itertuples(index=False):
        # Assume each work order produces 1~3 rolls in inventory
        n_roll = int(rng.integers(1, 4))
        produced_m = int(max(0, r.done_m))
        if produced_m <= 0:
            continue
        splits = rng.dirichlet(np.ones(n_roll))
        for j in range(n_roll):
            qty_m = int(max(80, produced_m * splits[j]))
            inbound_days_ago = int(rng.integers(0, 38))  # some rolls are old
            inbound_date = now - dt.timedelta(days=inbound_days_ago)
            rows.append(dict(
                roll_id=f"ROLL-{r.wo}-{j+1}",
                wo=r.wo,
                flow_card=getattr(r, "flow_card", None),
                barcode=r.barcode,
                customer=r.customer,
                market=r.market,
                line=getattr(r, "line", None),
                inbound_date=inbound_date,
                qty_m=qty_m,
            ))
    inv = pd.DataFrame(rows)
    return inv

def apply_shelf_life(
    orders_enriched: pd.DataFrame,
    inventory_rolls: pd.DataFrame,
    shelf_threshold_days: int = 20,
    loss_nt_per_m_per_day: float = 0.12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute shelf-age risk & quality loss cost.
    - When shelf age > threshold, add a 'quality risk loss' cost (NT$/m).
    """
    now = dt.datetime.now()
    inv = inventory_rolls.copy()
    inv["shelf_age_days"] = (now - inv["inbound_date"]).dt.days.clip(lower=0)

    # roll-level loss
    over = (inv["shelf_age_days"] - int(shelf_threshold_days)).clip(lower=0)
    inv["quality_loss_nt"] = over * float(loss_nt_per_m_per_day) * inv["qty_m"]
    inv["quality_loss_nt_per_m"] = (inv["quality_loss_nt"] / inv["qty_m"].replace(0, np.nan)).fillna(0.0)

    def shelf_label(days: int) -> str:
        if days > shelf_threshold_days + 10:
            return "🔴 超齡"
        if days > shelf_threshold_days:
            return "🟡 接近風險"
        return "🟢 正常"
    inv["shelf_risk"] = inv["shelf_age_days"].apply(shelf_label)

    # WO-level aggregation (use worst age / weighted loss per m)
    wo_agg = inv.groupby("wo", as_index=False).apply(
        lambda g: pd.Series({
            "shelf_age_days": int(g["shelf_age_days"].max()),
            "shelf_loss_nt_per_m": float((g["quality_loss_nt"].sum()) / max(g["qty_m"].sum(), 1e-6)),
            "shelf_risk": shelf_label(int(g["shelf_age_days"].max())),
            "inventory_m": int(g["qty_m"].sum()),
        })
    ).reset_index(drop=True)

    out = orders_enriched.merge(wo_agg, on="wo", how="left")
    out["shelf_age_days"] = out["shelf_age_days"].fillna(0).astype(int)
    out["shelf_loss_nt_per_m"] = out["shelf_loss_nt_per_m"].fillna(0.0)
    out["shelf_risk"] = out["shelf_risk"].fillna("🟢 正常")
    out["inventory_m"] = out["inventory_m"].fillna(0).astype(int)

    out["profit_with_carbon_and_shelf_nt_per_m"] = out["profit_with_carbon_nt_per_m"] - out["shelf_loss_nt_per_m"]
    return out, inv

def attach_ar_profit(ar: pd.DataFrame, orders_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Map unit economics to AR rows so we can show 'AR backed profit' and 'carbon adjusted profit'.
    """
    m = orders_enriched.set_index("wo")
    out = ar.copy()

    out["sell_price_nt_per_m"] = out["wo"].map(lambda x: float(m.loc[x, "sell_price_nt_per_m"]) if x in m.index else 0.0)
    out["unit_cost_nt_per_m"] = out["wo"].map(lambda x: float(m.loc[x, "unit_cost_nt_per_m"]) if x in m.index else 0.0)
    out["internal_carbon_nt_per_m"] = out["wo"].map(lambda x: float(m.loc[x, "internal_carbon_nt_per_m"]) if x in m.index else 0.0)
    out["profit_with_carbon_nt_per_m"] = out["wo"].map(lambda x: float(m.loc[x, "profit_with_carbon_nt_per_m"]) if x in m.index else 0.0)
    out["shelf_loss_nt_per_m"] = out["wo"].map(lambda x: float(m.loc[x, "shelf_loss_nt_per_m"]) if x in m.index else 0.0)
    out["profit_with_carbon_and_shelf_nt_per_m"] = out["wo"].map(lambda x: float(m.loc[x, "profit_with_carbon_and_shelf_nt_per_m"]) if x in m.index else 0.0)
    out["carbon_kgco2_per_m"] = out["wo"].map(lambda x: float(m.loc[x, "carbon_kgco2_per_m"]) if x in m.index else 0.0)

    # Margin ratio (auditable approximation)
    out["margin_ratio_carbon"] = (out["sell_price_nt_per_m"] - (out["unit_cost_nt_per_m"] + out["internal_carbon_nt_per_m"] + out["shelf_loss_nt_per_m"])) / out["sell_price_nt_per_m"].replace(0, np.nan)
    out["margin_ratio_carbon"] = out["margin_ratio_carbon"].replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0)

    out["ar_profit_with_carbon_nt"] = out["ar_amount_nt"] * out["margin_ratio_carbon"]
    return out

# -----------------------------
# Decision layer (profit/OTD/carbon)
# -----------------------------
def enrich_decision_layer(
    df: pd.DataFrame,
    # cost
    elec_price_nt_per_kwh: float,
    labor_nt_per_hr: float,
    machine_nt_per_hr: float,
    overhead_nt_per_m: float,
    steam_price_proxy_nt_per_kwhth: float,
    # physics
    p_atm: float,
    rho_air: float,
    eta_base: float,
    h_fg_kjkg: float,
    h_steam_kjkg: float,
    cp_air_kjkgk: float,
    # emissions
    ef_elec: float,
    ef_steam: float,
    internal_carbon_nt_per_t: float,
) -> pd.DataFrame:
    out = df.copy()
    phys = []

    for r in out.itertuples(index=False):
        ph = dryer_physics(
            speed_mmin=r.speed_mmin,
            steam_kgph=r.steam_kgph,
            airflow_m3ph=r.airflow_m3ph,
            t_in_c=r.inlet_temp_c,
            rh_in=r.inlet_rh,
            t_out_c=r.exhaust_temp_c,
            rh_out=r.exhaust_rh,
            rho_air=rho_air,
            p_atm=p_atm,
            h_fg_kjkg=h_fg_kjkg,
            h_steam_kjkg=h_steam_kjkg,
            cp_air_kjkgk=cp_air_kjkgk,
            eta_base=eta_base,
        )
        phys.append(ph)

    out["eta_eff"] = [p["eta_eff"] for p in phys]
    out["kwh_th_per_m_req"] = [p["kwh_th_per_m_req"] for p in phys]
    out["kwh_th_per_m_steam"] = [p["kwh_th_per_m_steam"] for p in phys]
    out["steam_util"] = [p["steam_util"] for p in phys]

    m_per_hr = out["speed_mmin"] * 60.0
    out["kwh_elec_per_m_est"] = (out["power_kw"] / m_per_hr.replace(0, np.nan)).fillna(0.0)

    energy_nt_per_m = out["kwh_elec_per_m_est"] * elec_price_nt_per_kwh + out["kwh_th_per_m_req"] * steam_price_proxy_nt_per_kwhth
    labor_nt_per_m = (labor_nt_per_hr / m_per_hr.replace(0, np.nan)).fillna(0.0)
    machine_nt_per_m = (machine_nt_per_hr / m_per_hr.replace(0, np.nan)).fillna(0.0)

    out["unit_cost_nt_per_m"] = energy_nt_per_m + labor_nt_per_m + machine_nt_per_m + overhead_nt_per_m
    out["profit_nt_per_m"] = out["sell_price_nt_per_m"] - out["unit_cost_nt_per_m"]

    out["carbon_kgco2_per_m"] = out["kwh_elec_per_m_est"] * ef_elec + out["kwh_th_per_m_req"] * ef_steam
    out["internal_carbon_nt_per_m"] = (out["carbon_kgco2_per_m"] / 1000.0) * internal_carbon_nt_per_t
    out["profit_with_carbon_nt_per_m"] = out["sell_price_nt_per_m"] - (out["unit_cost_nt_per_m"] + out["internal_carbon_nt_per_m"])

    out["remain_m"] = (out["plan_m"] - out["done_m"]).clip(lower=0)
    out["eta_hr"] = (out["remain_m"] / m_per_hr.replace(0, np.nan)).fillna(np.inf)
    now = dt.datetime.now()
    finish = now + pd.to_timedelta(out["eta_hr"], unit="h")
    slack_hr = (out["due"] - finish).dt.total_seconds() / 3600.0

    def otd_label(x: float) -> str:
        if x >= 2:
            return "🟢 準交"
        if x >= -2:
            return "🟡 風險"
        return "🔴 逾期"
    out["otd"] = slack_hr.apply(otd_label)

    return out

# -----------------------------
# Event engine (ties order page & asset page)
# -----------------------------
def detect_events(row: pd.Series) -> list[dict]:
    now = dt.datetime.now()
    events: list[dict] = []

    # exhaust RH
    if row["exhaust_rh"] > 0.88:
        events.append(dict(ts=now, severity="🔴", event="EXH_RH_HIGH", subsystem="Exhaust/Dehumid",
                           explain=f"排風RH={row['exhaust_rh']*100:.1f}% 偏高 → 蒸發驅動力下降、能耗上升",
                           impact_nt_per_m=-0.9))
    elif row["exhaust_rh"] > 0.80:
        events.append(dict(ts=now, severity="🟡", event="EXH_RH_ELEVATED", subsystem="Exhaust/Dehumid",
                           explain=f"排風RH={row['exhaust_rh']*100:.1f}% 偏高 → 能耗/品質風險上升",
                           impact_nt_per_m=-0.4))

    # airflow
    if row["airflow_m3ph"] < 11000:
        events.append(dict(ts=now, severity="🔴", event="AIRFLOW_DROP", subsystem="Fan/Airflow",
                           explain=f"風量={row['airflow_m3ph']:.0f} m³/h 偏低 → 排濕能力不足",
                           impact_nt_per_m=-0.7))
    elif row["airflow_m3ph"] < 12500:
        events.append(dict(ts=now, severity="🟡", event="AIRFLOW_LOW", subsystem="Fan/Airflow",
                           explain=f"風量={row['airflow_m3ph']:.0f} m³/h 偏低 → 建議檢查VFD/濾網/皮帶",
                           impact_nt_per_m=-0.3))

    # steam util
    if row["steam_util"] > 1.20:
        events.append(dict(ts=now, severity="🔴", event="STEAM_UTIL_HIGH", subsystem="Steam/Heater",
                           explain=f"熱需求/蒸氣供熱比={row['steam_util']:.2f} → 可能漏風/散熱/蒸氣不足",
                           impact_nt_per_m=-0.8))
    elif row["steam_util"] > 1.05:
        events.append(dict(ts=now, severity="🟡", event="STEAM_EFF_LOW", subsystem="Steam/Heater",
                           explain=f"熱需求/蒸氣供熱比={row['steam_util']:.2f} → 效率偏低，建議查排風/疏水/保溫",
                           impact_nt_per_m=-0.35))

    # width stability (chain/tension)
    widths = [row[f"width_mm_{i}"] for i in range(1, 9)]
    width_std = float(np.std(widths))
    if width_std > 6.0:
        events.append(dict(ts=now, severity="🔴", event="WIDTH_UNSTABLE", subsystem="Chain/Tension",
                           explain=f"定型寬度波動STD={width_std:.1f}mm → 張力/鏈條/夾具可能異常",
                           impact_nt_per_m=-0.6))
    elif width_std > 4.0:
        events.append(dict(ts=now, severity="🟡", event="WIDTH_VARIANCE", subsystem="Chain/Tension",
                           explain=f"定型寬度波動STD={width_std:.1f}mm → 建議檢查張力/夾具磨耗",
                           impact_nt_per_m=-0.25))

    # power
    if row["voltage_v"] < 360 or row["voltage_v"] > 400:
        events.append(dict(ts=now, severity="🔴", event="VOLTAGE_ANOMALY", subsystem="Power/VFD",
                           explain=f"電壓={row['voltage_v']:.0f}V 異常 → 可能影響風車/變頻穩定",
                           impact_nt_per_m=-0.5))

    return events

# -----------------------------
# Robot Dog Inspection (PoC via CSV/JSON; here demo generator)
# -----------------------------
def generate_demo_robotdog_runs(lines=("LINE-A","LINE-B","LINE-C","LINE-D"), seed: int = 123, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    now = dt.datetime.now()
    subsystems = ["Fan & Airflow","Steam / Heater","Exhaust / Dehumid","Chain / Tension","Cooling","Power / VFD","Safety"]
    anomaly_types = [
        "HOTSPOT_PANEL", "STEAM_LEAK", "BEARING_NOISE", "BELT_SLIP",
        "DUCT_BLOCKAGE", "OIL_LEAK", "OBSTACLE", "ABNORMAL_VIB"
    ]
    rows = []
    for i in range(n):
        line = rng.choice(lines)
        subsystem = rng.choice(subsystems)
        atype = rng.choice(anomaly_types)
        severity = rng.choice(["🟢","🟡","🔴"], p=[0.65, 0.25, 0.10])

        ir_max_c = float(rng.uniform(45, 110))     # thermal max
        noise_db = float(rng.uniform(45, 95))      # acoustic
        vib_rms  = float(rng.uniform(0.2, 4.0))    # vibration
        gas_ppm  = float(rng.uniform(0, 200))      # gas proxy
        conf     = float(np.clip(rng.normal(0.78, 0.12), 0.2, 0.99))

        rows.append(dict(
            ts=now - dt.timedelta(minutes=int(rng.integers(1, 720))),
            line=str(line),
            subsystem=subsystem,
            anomaly_type=atype,
            severity=severity,
            ir_max_c=ir_max_c,
            noise_db=noise_db,
            vib_rms=vib_rms,
            gas_ppm=gas_ppm,
            confidence=conf,
            evidence_uri=f"robotdog://run/{i}",
        ))
    return pd.DataFrame(rows).sort_values("ts")

def robotdog_to_events(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert robot dog observations into the same 'event timeline' schema.
    Columns: ts, severity, event, subsystem, explain, impact_nt_per_m, line, evidence_uri
    """
    if obs is None or len(obs) == 0:
        return pd.DataFrame(columns=["ts","severity","event","subsystem","explain","impact_nt_per_m","line","evidence_uri"])

    def impact_rule(r) -> float:
        base = {"🟢": -0.05, "🟡": -0.25, "🔴": -0.80}.get(r["severity"], -0.1)
        bump = 0.0
        if r["anomaly_type"] in ("STEAM_LEAK","DUCT_BLOCKAGE"): bump -= 0.25
        if r["anomaly_type"] in ("BEARING_NOISE","ABNORMAL_VIB","BELT_SLIP"): bump -= 0.20
        if r["anomaly_type"] in ("HOTSPOT_PANEL","OIL_LEAK"): bump -= 0.15
        if r["anomaly_type"] == "OBSTACLE": bump -= 0.10
        return float((base + bump) * float(r["confidence"]))

    def explain_rule(r) -> str:
        return (
            f"機器狗巡檢：{r['anomaly_type']}｜IRmax={r['ir_max_c']:.1f}°C｜"
            f"Noise={r['noise_db']:.1f}dB｜Vib={r['vib_rms']:.2f}｜Gas={r['gas_ppm']:.0f}ppm｜"
            f"conf={r['confidence']:.2f}"
        )

    out = obs.copy()
    out["event"] = out["anomaly_type"].astype(str)
    out["explain"] = out.apply(explain_rule, axis=1)
    out["impact_nt_per_m"] = out.apply(impact_rule, axis=1)
    if "line" not in out.columns:
        out["line"] = "UNKNOWN"

    return out[["ts","severity","event","subsystem","explain","impact_nt_per_m","line","evidence_uri"]]


def build_asset_cards(row: pd.Series) -> pd.DataFrame:
    fan_hz_mean = float(np.mean([row[f"fan_hz_{i}"] for i in range(1, 9)]))
    width_std = float(np.std([row[f"width_mm_{i}"] for i in range(1, 9)]))

    cards = [
        dict(subsystem="Fan & Airflow", kpi1="airflow_m3ph", v1=float(row["airflow_m3ph"]), kpi2="fan_hz_mean", v2=fan_hz_mean),
        dict(subsystem="Steam / Heater", kpi1="steam_kgph", v1=float(row["steam_kgph"]), kpi2="steam_util", v2=float(row["steam_util"])),
        dict(subsystem="Exhaust / Dehumid", kpi1="exhaust_rh", v1=float(row["exhaust_rh"]), kpi2="exhaust_front_hz", v2=float(row["exhaust_front_hz"])),
        dict(subsystem="Chain / Tension", kpi1="width_std_mm", v1=width_std, kpi2="speed_mmin", v2=float(row["speed_mmin"])),
        dict(subsystem="Cooling", kpi1="cooling_hz", v1=float(row["cooling_hz"]), kpi2="exhaust_temp_c", v2=float(row["exhaust_temp_c"])),
        dict(subsystem="Power / VFD", kpi1="voltage_v", v1=float(row["voltage_v"]), kpi2="current_a", v2=float(row["current_a"])),
    ]
    df = pd.DataFrame(cards)

    def health(sub: str, v1: float, v2: float) -> str:
        if sub == "Exhaust / Dehumid":
            return "🔴" if v1 > 0.88 else ("🟡" if v1 > 0.80 else "🟢")
        if sub == "Fan & Airflow":
            return "🔴" if v1 < 10500 else ("🟡" if v1 < 12500 else "🟢")
        if sub == "Steam / Heater":
            return "🔴" if v2 > 1.20 else ("🟡" if v2 > 1.05 else "🟢")
        if sub == "Chain / Tension":
            return "🔴" if v1 > 6.0 else ("🟡" if v1 > 4.0 else "🟢")
        if sub == "Power / VFD":
            return "🔴" if (v1 < 360 or v1 > 400) else ("🟡" if (v1 < 370 or v1 > 395) else "🟢")
        return "🟢"
    df["health"] = [health(r.subsystem, r.v1, r.v2) for r in df.itertuples(index=False)]
    return df

# -----------------------------
# UI: Sidebar (proposal-friendly)
# -----------------------------
st.sidebar.title("🎛️ Proposal Demo Controls")

mode = st.sidebar.radio("呈現模式", ["💼 經營 / ESG / 金融模式（建議客戶）", "👷 工程師模式（現場/稽核）"], index=0)
show_formulas = st.sidebar.checkbox("📐 顯示物理模型公式", value=(mode.startswith("👷")))
show_on_site_refs = st.sidebar.checkbox("🖼️ 顯示現場兩頁參考圖", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 成本與碳參數（可快速做情境）")
elec_price = st.sidebar.number_input("電價 NT$/kWh", 1.0, 12.0, 3.2, 0.1)
steam_price_proxy = st.sidebar.number_input("蒸氣單價 Proxy NT$/kWh_th", 0.5, 10.0, 2.2, 0.1)
labor_hr = st.sidebar.number_input("人工 NT$/hr", 200.0, 1600.0, 520.0, 10.0)
machine_hr = st.sidebar.number_input("機台 NT$/hr", 200.0, 3000.0, 900.0, 20.0)
overhead_m = st.sidebar.number_input("製造費用 NT$/m", 0.0, 5.0, 0.65, 0.05)

ef_elec = st.sidebar.number_input("電力 EF kgCO2/kWh", 0.05, 1.5, 0.52, 0.01)
ef_steam = st.sidebar.number_input("蒸氣 EF kgCO2/kWh_th", 0.05, 1.5, 0.25, 0.01)
internal_carbon = st.sidebar.number_input("內部碳價 NT$/tCO2e", 0.0, 8000.0, 1200.0, 50.0)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 布料時效庫存（Shelf-life）")
shelf_threshold_days = st.sidebar.number_input("庫齡門檻(天)", 5, 90, 20, 1)
loss_nt_per_m_per_day = st.sidebar.number_input("庫齡損失係數 NT$/m/天（超過門檻後）", 0.0, 5.0, 0.12, 0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 物理參數（現場可校準）")
p_atm = st.sidebar.number_input("大氣壓 Pa", value=101325.0, step=100.0)
rho_air = st.sidebar.number_input("空氣密度 kg/m³", value=1.20, step=0.01)
eta_base = st.sidebar.slider("乾燥效率 η", 0.30, 0.95, 0.75, 0.01)
h_fg = st.sidebar.number_input("水蒸發潛熱 h_fg kJ/kg", value=2257.0, step=10.0)
h_steam = st.sidebar.number_input("蒸氣有效焓差 h_steam kJ/kg", value=2000.0, step=50.0)
cp_air = st.sidebar.number_input("空氣比熱 cp kJ/kg/K", value=1.005, step=0.001)

# -----------------------------
# Data
# -----------------------------
if "orders_raw" not in st.session_state:
    st.session_state.orders_raw = generate_demo_orders()

orders = enrich_decision_layer(
    st.session_state.orders_raw,
    elec_price_nt_per_kwh=elec_price,
    labor_nt_per_hr=labor_hr,
    machine_nt_per_hr=machine_hr,
    overhead_nt_per_m=overhead_m,
    steam_price_proxy_nt_per_kwhth=steam_price_proxy,
    p_atm=p_atm,
    rho_air=rho_air,
    eta_base=eta_base,
    h_fg_kjkg=h_fg,
    h_steam_kjkg=h_steam,
    cp_air_kjkgk=cp_air,
    ef_elec=ef_elec,
    ef_steam=ef_steam,
    internal_carbon_nt_per_t=internal_carbon,
)


# -----------------------------
# Demo Inventory (Shelf-life)
# -----------------------------
if "inventory_demo" not in st.session_state:
    # Inventory is created from the enriched orders (so we have customer/market mapping)
    st.session_state.inventory_demo = generate_demo_inventory(orders)

orders, inventory = apply_shelf_life(
    orders_enriched=orders,
    inventory_rolls=st.session_state.inventory_demo,
    shelf_threshold_days=int(shelf_threshold_days),
    loss_nt_per_m_per_day=float(loss_nt_per_m_per_day),
)

# -----------------------------
# Demo AR (Accounts Receivable)
# -----------------------------
if "ar_demo" not in st.session_state:
    st.session_state.ar_demo = generate_demo_ar(orders)

ar = attach_ar_profit(st.session_state.ar_demo, orders)
# -----------------------------
# Priority queue (production scheduling) — session state
# -----------------------------
if "priority_queue" not in st.session_state:
    st.session_state.priority_queue = pd.DataFrame(columns=[
        "rank", "score", "wo", "flow_card", "line", "customer",
        "otd", "due", "shelf_age_days", "shelf_loss_nt_per_m",
        "total_quality_loss_nt", "profit_with_carbon_and_shelf_nt_per_m",
        "inventory_m", "created_at", "reasons"
    ])

# -----------------------------
# Header
# -----------------------------
st.title("🏭 YuYang — Demo（兩頁監測 → 立刻變成錢與碳）")

tabs = st.tabs([
"① 多工單盤（Executive Portfolio）",
    "② 工單頁（對齊現場第 1 頁）",
    "③ 設備頁（對齊現場第 2 頁）",
    "④ 事件時間線（原因＝錢）",
    "⑤ AR 眼鏡（現場即時疊加）",
    "⑥ AR（應收帳款）+ 即時成本盈虧",
    "⑦ 簡報/操作-對國際大廠",
    "⑧ 機器狗巡檢（Robot Dog）",

])

# -----------------------------
# Helpers: formula panel
# -----------------------------
def render_formulas():
    st.markdown("### 📐 乾燥段物理模型（可稽核）")
    st.markdown("**(A) 含濕比（由 T、RH 推算）**")
    st.latex(r"p_v = RH\cdot p_{ws}(T)")
    st.latex(r"w = 0.62198\cdot \frac{p_v}{p_{atm}-p_v}")
    st.markdown("**(B) 蒸發水量（由風量與含濕比差）**")
    st.latex(r"\dot m_{da}\approx \rho_{air}\cdot \dot V")
    st.latex(r"\dot m_w = \dot m_{da}\cdot (w_{out}-w_{in})")
    st.markdown("**(C) 熱需求（蒸發潛熱 + 空氣顯熱）**")
    st.latex(r"\dot Q_{req}=\frac{\dot m_w h_{fg}+\dot m_{da}c_p(T_{out}-T_{in})}{\eta}")
    st.markdown("**(D) 蒸氣供熱（由蒸氣流量）**")
    st.latex(r"\dot Q_{steam}=\dot m_s\cdot h_{steam}")
    st.markdown("**(E) kWh/m → kgCO₂/m → NT$/m**")
    st.latex(r"v_{hr}=60v,\;\;E_{th}(kWh/m)=\frac{\dot Q_{req}(kW)}{v_{hr}(m/hr)}")
    st.latex(r"I(kgCO_2/m)=E_{elec}EF_{elec}+E_{th}EF_{steam}")
    st.latex(r"C_{carbon}(NT\$/m)=\frac{I}{1000}\cdot P_{carbon}")

# -----------------------------
# Tab 1: Portfolio
# -----------------------------
with tabs[0]:
    st.subheader("① 多工單盤（先看錢，再看原因）")

    # --------------------------------------------------
    # 1) 滿線壓力下：跨線別工單即時尋找（搜尋引擎）
    # --------------------------------------------------
    with st.container(border=True):
        st.markdown("### 🔎 跨線別工單即時尋找（滿線壓力下的定位）")
        key = st.text_input("輸入流程卡號 / Barcode / 工單（例：03017270）", value="", placeholder="03017270")
        if key:
            k = key.strip().lower()
            hit = orders[
                orders["flow_card"].astype(str).str.lower().str.contains(k)
                | orders["barcode"].astype(str).str.lower().str.contains(k)
                | orders["wo"].astype(str).str.lower().str.contains(k)
            ]
            if len(hit) == 0:
                st.warning("找不到該工單/流程卡（Demo 資料）。")
            else:
                r = hit.iloc[0]
                c1, c2, c3, c4, c5 = st.columns([1.2, 1.0, 1.1, 1.3, 1.6])
                c1.metric("流程卡號", str(r["flow_card"]))
                c2.metric("線別", str(r["line"]))
                c3.metric("工單", str(r["wo"]))
                c4.metric("準交", str(r["otd"]), f"ETA {float(r['eta_hr']):.1f} hr")
                pct = float(r["done_m"] / max(r["plan_m"], 1)) if float(r["plan_m"]) > 0 else 0.0
                c5.metric("碼表進度", f"{pct*100:.1f}%", f"{int(r['done_m']):,}/{int(r['plan_m']):,} m")
                st.progress(min(max(pct, 0.0), 1.0))
                # Quick status line
                late_flag = "⚠ 交期風險" if str(r["otd"]) != "🟢 準交" else "✅ 準交"
                shelf_flag = f"{str(r['shelf_risk'])} 庫齡 {int(r['shelf_age_days'])} 天 / 損失 {float(r['shelf_loss_nt_per_m']):.2f} NT$/m"
                st.caption(f"{late_flag} ｜ {shelf_flag}")

    f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.2, 1.8])
    with f1:
        otd_filter = st.selectbox("準交", ["ALL", "🟢 準交", "🟡 風險", "🔴 逾期"])
    with f2:
        market_filter = st.selectbox("市場", ["ALL"] + sorted(orders["market"].unique().tolist()))
    with f3:
        sort_by = st.selectbox("排序", ["profit_with_carbon_nt_per_m", "profit_nt_per_m", "carbon_kgco2_per_m", "eta_hr"])
    with f4:
        q = st.text_input("搜尋（工單/Barcode/客戶）")

    view = orders.copy()
    if otd_filter != "ALL":
        view = view[view["otd"] == otd_filter]
    if market_filter != "ALL":
        view = view[view["market"] == market_filter]
    if q:
        ql = q.lower()
        view = view[
            view["wo"].astype(str).str.lower().str.contains(ql)
            | view["barcode"].astype(str).str.lower().str.contains(ql)
            | view["customer"].astype(str).str.lower().str.contains(ql)
        ]

    view = view.sort_values(sort_by, ascending=(sort_by in ["carbon_kgco2_per_m", "eta_hr"]))

    show_cols = [
        "line", "flow_card",
        "otd", "wo", "barcode", "customer", "market",
        "speed_mmin", "eta_hr",
        "unit_cost_nt_per_m", "profit_nt_per_m", "internal_carbon_nt_per_m",
        "shelf_loss_nt_per_m", "profit_with_carbon_and_shelf_nt_per_m",
        "carbon_kgco2_per_m",
        "shelf_risk", "shelf_age_days", "inventory_m",
        "exhaust_rh", "airflow_m3ph", "steam_kgph", "steam_util"
    ]
    table = view[show_cols].copy()
    table["exhaust_rh"] = (table["exhaust_rh"] * 100.0).round(1).astype(str) + "%"

    table = table.rename(columns={
        "line": "線別", "flow_card": "流程卡號",
        "otd": "準交", "wo": "工單", "barcode": "Barcode", "customer": "客戶", "market": "市場",
        "speed_mmin": "速度(m/min)", "eta_hr": "ETA(hr)",
        "unit_cost_nt_per_m": "成本(NT$/m)", "profit_nt_per_m": "盈虧(NT$/m)",
        "internal_carbon_nt_per_m": "內部碳(NT$/m)",
        "shelf_loss_nt_per_m": "庫齡損失(NT$/m)",
        "profit_with_carbon_and_shelf_nt_per_m": "含碳+庫齡盈虧(NT$/m)",
        "carbon_kgco2_per_m": "kgCO2/m",
        "shelf_risk": "庫齡狀態", "shelf_age_days": "庫齡(天)", "inventory_m": "庫存(m)",
        "exhaust_rh": "排風RH", "airflow_m3ph": "風量(m3/h)", "steam_kgph": "蒸氣(kg/h)",
        "steam_util": "蒸氣利用比"
    })

    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("#### 💥 一鍵挑出『在燒錢』的工單")
    losers = view[view["profit_with_carbon_and_shelf_nt_per_m"] < 0].head(6)
    if len(losers):
        st.dataframe(losers[["line","flow_card","wo", "barcode", "otd", "profit_with_carbon_and_shelf_nt_per_m", "shelf_loss_nt_per_m", "carbon_kgco2_per_m", "exhaust_rh", "airflow_m3ph"]],
                     use_container_width=True, hide_index=True)
    else:
        st.success("✅ 目前沒有含碳後為負的工單（Demo 資料）。")

    st.markdown("---")
    st.markdown("### 🧵 布料時效庫存監控（Shelf-life Management）")
    st.write("當庫齡超過門檻，面板自動轉黃/紅並計入『品質風險損失』，提醒生管優先排產。")
    inv_view = inventory.sort_values(["shelf_age_days", "quality_loss_nt"], ascending=[False, False]).copy()

    # 保護：不同資料來源欄位可能不齊（例如真實 WMS 可能沒有 flow_card）
    desired_cols = ["shelf_risk","shelf_age_days","roll_id","wo","flow_card","line","barcode","customer","inbound_date","qty_m","quality_loss_nt","quality_loss_nt_per_m"]
    existing_cols = [c for c in desired_cols if c in inv_view.columns]
    inv_show = inv_view[existing_cols].copy()

    # 若缺 flow_card 但 orders 可回填（以 wo 對應）
    if "flow_card" not in inv_show.columns:
        wo2fc = orders.set_index("wo")["flow_card"].to_dict() if "flow_card" in orders.columns else {}
        inv_show["flow_card"] = inv_show["wo"].map(lambda x: wo2fc.get(x, ""))
    inv_show = inv_show.rename(columns={
        "shelf_risk":"狀態","shelf_age_days":"庫齡(天)","roll_id":"卷號","wo":"工單","flow_card":"流程卡號","line":"線別",
        "barcode":"Barcode","customer":"客戶","inbound_date":"入庫日","qty_m":"庫存(m)","quality_loss_nt":"品質風險損失(NT$)","quality_loss_nt_per_m":"損失(NT$/m)"
    })
    st.dataframe(inv_show, use_container_width=True, hide_index=True)

    st.markdown("#### 🔥 超齡卷 TOP10（損失最大）")
    aged_top = inventory[inventory["shelf_age_days"] > int(shelf_threshold_days)].copy()
    if len(aged_top) == 0:
        st.success("✅ 目前沒有超過門檻的庫存卷（Demo 資料）。")
    else:
        aged_top = aged_top.sort_values(["quality_loss_nt", "shelf_age_days"], ascending=[False, False]).head(10)
        top_cols = [c for c in ["shelf_risk","shelf_age_days","quality_loss_nt","roll_id","wo","flow_card","line","customer","barcode","inbound_date","qty_m","quality_loss_nt_per_m"] if c in aged_top.columns]
        top_df = aged_top[top_cols].copy()
        if "flow_card" not in top_df.columns:
            wo2fc = orders.set_index("wo")["flow_card"].to_dict() if "flow_card" in orders.columns else {}
            top_df["flow_card"] = top_df["wo"].map(lambda x: wo2fc.get(x, ""))
        top_df = top_df.rename(columns={
            "shelf_risk":"狀態","shelf_age_days":"庫齡(天)","quality_loss_nt":"品質風險損失(NT$)",
            "roll_id":"卷號","wo":"工單","flow_card":"流程卡號","line":"線別","customer":"客戶",
            "barcode":"Barcode","inbound_date":"入庫日","qty_m":"庫存(m)","quality_loss_nt_per_m":"損失(NT$/m)"
        })
        st.dataframe(top_df, use_container_width=True, hide_index=True)

    st.markdown("#### 🧾 客戶別庫齡風險排行")
    cust = inventory.copy()
    cust["is_over_threshold"] = cust["shelf_age_days"] > int(shelf_threshold_days)
    cust_rank = cust.groupby("customer", as_index=False).agg(
        rolls=("roll_id", "count"),
        over_rolls=("is_over_threshold", "sum"),
        over_ratio=("is_over_threshold", "mean"),
        max_age_days=("shelf_age_days", "max"),
        avg_age_days=("shelf_age_days", "mean"),
        total_loss_nt=("quality_loss_nt", "sum"),
        total_m=("qty_m", "sum"),
    )
    cust_rank["loss_nt_per_m"] = (cust_rank["total_loss_nt"] / cust_rank["total_m"].replace(0, np.nan)).fillna(0.0)
    # a simple "risk score" for ranking (auditable)
    cust_rank["risk_score"] = (cust_rank["over_ratio"] * 0.55 + (cust_rank["max_age_days"] / (int(shelf_threshold_days) + 20)) * 0.25 + (cust_rank["loss_nt_per_m"] / 2.0) * 0.20)
    cust_rank["risk_score"] = cust_rank["risk_score"].clip(0, 1.5)

    cust_rank = cust_rank.sort_values(["risk_score", "total_loss_nt"], ascending=[False, False])

    def cust_label(x: float) -> str:
        if x >= 0.85: return "🔴 高"
        if x >= 0.55: return "🟡 中"
        return "🟢 低"

    cust_show = cust_rank.copy()
    cust_show["level"] = cust_show["risk_score"].apply(cust_label)
    cust_show = cust_show.rename(columns={
        "level":"風險等級","customer":"客戶","risk_score":"風險分數","rolls":"卷數","over_rolls":"超齡卷數","over_ratio":"超齡占比",
        "max_age_days":"最大庫齡(天)","avg_age_days":"平均庫齡(天)","total_loss_nt":"總品質損失(NT$)","loss_nt_per_m":"損失(NT$/m)","total_m":"總庫存(m)"
    })
    cust_show["超齡占比"] = (cust_show["超齡占比"] * 100.0).round(1).astype(str) + "%"
    show_cols = ["風險等級","客戶","風險分數","卷數","超齡卷數","超齡占比","最大庫齡(天)","平均庫齡(天)","總庫存(m)","損失(NT$/m)","總品質損失(NT$)"]
    show_cols = [c for c in show_cols if c in cust_show.columns]
    st.dataframe(cust_show[show_cols],
                 use_container_width=True, hide_index=True)

    st.markdown("#### 🚀 一鍵把超齡卷對應工單送去「優先排產」隊列（排序權重）")
    with st.expander("設定排序權重（可用於提案：規則可稽核、可調參）", expanded=True):
        w_loss = st.slider("權重：品質損失（NT$）", 0.0, 2.0, 1.0, 0.05)
        w_age = st.slider("權重：庫齡超標程度", 0.0, 2.0, 0.9, 0.05)
        w_otd = st.slider("權重：交期風險（逾期/風險）", 0.0, 2.0, 0.8, 0.05)
        w_profit = st.slider("權重：含碳+庫齡為負（越負越優先）", 0.0, 2.0, 0.7, 0.05)

        st.caption("排序分數 = w_loss*loss_norm + w_age*age_norm + w_otd*otd_norm + w_profit*neg_profit_norm（全部 0~1 正規化）")

    # Build candidate WO list from inventory (only those over threshold)
    cand_rolls = inventory[inventory["shelf_age_days"] > int(shelf_threshold_days)].copy()
    if len(cand_rolls) == 0:
        st.info("目前沒有超齡卷，因此不需要推送優先排產。")
    else:
        wo_agg = cand_rolls.groupby("wo", as_index=False).agg(
            total_quality_loss_nt=("quality_loss_nt", "sum"),
            max_shelf_age_days=("shelf_age_days", "max"),
            inventory_m=("qty_m", "sum"),
        )

        # Defensive: if upstream inventory schema differs, ensure inventory_m exists
        if "inventory_m" not in wo_agg.columns:
            if "qty_m" in cand_rolls.columns:
                wo_agg["inventory_m"] = cand_rolls.groupby("wo")["qty_m"].sum().values
            elif "length_m" in cand_rolls.columns:
                wo_agg["inventory_m"] = cand_rolls.groupby("wo")["length_m"].sum().values
            else:
                wo_agg["inventory_m"] = 0.0
        wo_view = orders.merge(wo_agg, on="wo", how="inner")
        wo_view["flow_card"] = wo_view["flow_card"].astype(str)
        wo_view["age_over"] = (wo_view["max_shelf_age_days"] - int(shelf_threshold_days)).clip(lower=0)

        # otd norm
        def otd_norm(label: str) -> float:
            if "🔴" in label: return 1.0
            if "🟡" in label: return 0.6
            return 0.2
        wo_view["otd_norm"] = wo_view["otd"].apply(otd_norm)

        # profit norm (only negative matters)
        wo_view["neg_profit"] = (-wo_view["profit_with_carbon_and_shelf_nt_per_m"]).clip(lower=0)
        # normalize features (0~1)
        def norm01(s: pd.Series) -> pd.Series:
            s2 = s.astype(float)
            mx = float(s2.max()) if len(s2) else 1.0
            mn = float(s2.min()) if len(s2) else 0.0
            if mx - mn < 1e-9:
                return pd.Series(np.zeros(len(s2)), index=s2.index)
            return (s2 - mn) / (mx - mn)

        wo_view["loss_norm"] = norm01(wo_view["total_quality_loss_nt"])
        wo_view["age_norm"] = norm01(wo_view["age_over"])
        wo_view["profit_norm"] = norm01(wo_view["neg_profit"])

        wo_view["score"] = (
            float(w_loss) * wo_view["loss_norm"]
            + float(w_age) * wo_view["age_norm"]
            + float(w_otd) * wo_view["otd_norm"]
            + float(w_profit) * wo_view["profit_norm"]
        )

        # Preview top candidates
        prev = wo_view.sort_values("score", ascending=False).head(8).copy()
        prev = prev.rename(columns={
            "line":"線別","flow_card":"流程卡號","wo":"工單","customer":"客戶","otd":"準交","due":"交期",
            "max_shelf_age_days":"最大庫齡(天)","total_quality_loss_nt":"總品質損失(NT$)",
            "shelf_loss_nt_per_m":"庫齡損失(NT$/m)","profit_with_carbon_and_shelf_nt_per_m":"含碳+庫齡盈虧(NT$/m)",
            "score":"排序分數"
        })
        st.dataframe(prev[["排序分數","線別","流程卡號","工單","客戶","準交","交期","最大庫齡(天)","總品質損失(NT$)","庫齡損失(NT$/m)","含碳+庫齡盈虧(NT$/m)"]],
                     use_container_width=True, hide_index=True)

        cbtn1, cbtn2 = st.columns([1,1])
        with cbtn1:
            push = st.button("📤 一鍵推送：超齡卷對應工單 → 優先排產隊列", use_container_width=True)
        with cbtn2:
            clearq = st.button("🧹 清空優先排產隊列", use_container_width=True)

        if clearq:
            st.session_state.priority_queue = st.session_state.priority_queue.iloc[0:0].copy()
            st.success("已清空優先排產隊列。")

        if push:
            now_ts = dt.datetime.now()
            q = st.session_state.priority_queue.copy()
            add = wo_view.copy()
            add["created_at"] = now_ts
            add["reasons"] = add.apply(
                lambda r: f"庫齡超標{int(r['age_over'])}天｜損失NT${r['total_quality_loss_nt']:.0f}｜{r['otd']}",
                axis=1
            )
            add = add.rename(columns={
                "line":"line","flow_card":"flow_card","wo":"wo","customer":"customer","otd":"otd","due":"due",
                "max_shelf_age_days":"shelf_age_days","shelf_loss_nt_per_m":"shelf_loss_nt_per_m",
                "total_quality_loss_nt":"total_quality_loss_nt",
                "profit_with_carbon_and_shelf_nt_per_m":"profit_with_carbon_and_shelf_nt_per_m",
                "inventory_m":"inventory_m",
            })

            # Defensive: some columns may not exist depending on data source; create defaults
            if "inventory_m" not in add.columns:
                add["inventory_m"] = 0.0
            desired_cols = [
                "score","wo","flow_card","line","customer","otd","due",
                "shelf_age_days","shelf_loss_nt_per_m","total_quality_loss_nt",
                "profit_with_carbon_and_shelf_nt_per_m","inventory_m","created_at","reasons"
            ]
            desired_cols = [c for c in desired_cols if c in add.columns]
            add = add[desired_cols]

            # Defensive: concat requires uniquely named columns. Some upstream transforms
            # (especially when merging/renaming) can accidentally create duplicated
            # column names and trigger: InvalidIndexError: Reindexing only valid with uniquely valued Index objects.
            def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
                df = df.copy()
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated()].copy()
                return df

            q_base = q.drop(columns=["rank"], errors="ignore")
            q_base = _dedup_columns(q_base)
            add = _dedup_columns(add)

            q2 = pd.concat([q_base, add], ignore_index=True, sort=False)
            # de-dup by WO, keep the highest score
            q2 = q2.sort_values("score", ascending=False).drop_duplicates(subset=["wo"], keep="first")
            q2 = q2.sort_values("score", ascending=False).reset_index(drop=True)
            q2.insert(0, "rank", np.arange(1, len(q2) + 1))
            st.session_state.priority_queue = q2
            st.success(f"✅ 已推送 {len(add)} 筆工單到優先排產隊列（自動去重）。")

        if len(st.session_state.priority_queue):
            st.markdown("##### 📌 目前優先排產隊列（由高到低）")
            qshow = st.session_state.priority_queue.copy()
            qshow["due"] = pd.to_datetime(qshow["due"]).dt.strftime("%Y-%m-%d %H:%M")
            qshow = qshow.rename(columns={
                "rank":"順位","score":"分數","wo":"工單","flow_card":"流程卡號","line":"線別","customer":"客戶",
                "otd":"準交","due":"交期","shelf_age_days":"最大庫齡(天)","shelf_loss_nt_per_m":"庫齡損失(NT$/m)",
                "total_quality_loss_nt":"總品質損失(NT$)","profit_with_carbon_and_shelf_nt_per_m":"含碳+庫齡盈虧(NT$/m)",
                "inventory_m":"庫存(m)","reasons":"原因"
            })

            show_cols_q = ["順位","分數","線別","流程卡號","工單","客戶","準交","交期","最大庫齡(天)","庫存(m)","總品質損失(NT$)","含碳+庫齡盈虧(NT$/m)","原因"]
            show_cols_q = [c for c in show_cols_q if c in qshow.columns]
            st.dataframe(qshow[show_cols_q], use_container_width=True, hide_index=True)

            # -----------------------------
            # Demo Export: PDF Schedule Suggestion
            # -----------------------------
            with st.expander("📄 Demo 版輸出：排程建議表（PDF）", expanded=False):
                if not _REPORTLAB_OK:
                    st.warning("此環境未安裝 reportlab，無法輸出 PDF。請先 pip install reportlab")
                else:
                    c1, c2 = st.columns([1,1])
                    with c1:
                        rate_m_per_hr = st.number_input("假設產能 (m/hr)", min_value=50.0, max_value=5000.0, value=600.0, step=50.0)
                    with c2:
                        changeover_hr = st.number_input("同線換線時間 (hr)", min_value=0.0, max_value=5.0, value=0.25, step=0.05)

                    gen_pdf = st.button("生成 PDF（排程建議表）", use_container_width=True)
                    if gen_pdf:
                        pdf_bytes = build_schedule_pdf_from_queue(
                            st.session_state.priority_queue,
                            now_ts=dt.datetime.now(),
                            rate_m_per_hr=float(rate_m_per_hr),
                            changeover_hr_same_line=float(changeover_hr),
                        )
                        st.session_state.last_sched_pdf = pdf_bytes
                        st.success("✅ PDF 已生成，可直接下載。")

                    pdf_bytes = st.session_state.get("last_sched_pdf", b"")
                    if pdf_bytes:
                        st.download_button(
                            "⬇️ 下載：排程建議表_優先排產.pdf",
                            data=pdf_bytes,
                            file_name="排程建議表_優先排產.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

# -----------------------------
# Tab 2: Order View (aligned to on-site Page 1)
# -----------------------------
with tabs[1]:
    st.subheader("② 工單頁（對齊現場第 1 頁：8區溫度/風車/寬度 + 決策 KPI）")

    if show_on_site_refs:
        with st.expander("🖼️ 展開：現場原始監測畫面（兩頁）", expanded=False):
            c1, c2 = st.columns(2)
            img1 = "S__92315682_0.jpg"
            img2 = "S__92315681_0.jpg"
            with c1:
                if os.path.exists(img1):
                    st.image(img1, caption="現場頁 1：工單/配方/8區製程即時")
                else:
                    st.info("找不到現場頁 1 圖檔（請放在同資料夾）。")
            with c2:
                if os.path.exists(img2):
                    st.image(img2, caption="現場頁 2：整機運轉部位狀態")
                else:
                    st.info("找不到現場頁 2 圖檔（請放在同資料夾）。")

    pick = st.selectbox("選擇工單", orders["wo"].tolist())
    row = orders[orders["wo"] == pick].iloc[0]

    # Executive KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("流程卡號", str(row["flow_card"]))
    k2.metric("線別", str(row["line"]))
    k3.metric("Barcode/Recipe", str(row["barcode"]))
    k4.metric("準交", row["otd"], f"ETA {row['eta_hr']:.1f} hr")
    k5.metric("每米盈虧（含碳+庫齡）", f"NT$ {row['profit_with_carbon_and_shelf_nt_per_m']:.2f}/m",
              f"庫齡損失 {row['shelf_loss_nt_per_m']:.2f}/m")
    k6.metric("碳強度", f"{row['carbon_kgco2_per_m']:.3f} kgCO₂/m", f"庫齡 {int(row['shelf_age_days'])} 天 {row['shelf_risk']}")

    st.write("---")

    if mode.startswith("👷"):
        # 8 zones: show like the on-site screen
        zcols = st.columns(8)
        for i in range(1, 9):
            with zcols[i-1]:
                st.markdown(f"**Zone {i}**")
                st.metric("溫度", f"{row[f'zone_temp_{i}']:.0f}°C")
                st.metric("風車", f"{row[f'fan_hz_{i}']:.0f} Hz")
                st.metric("寬度", f"{row[f'width_mm_{i}']:.0f} mm")
    else:
        st.markdown("### ✅ 這張工單的『三個關鍵』")
        drivers = []
        if row["exhaust_rh"] > 0.80:
            drivers.append(("排風RH偏高", "蒸發驅動力下降 → 能耗上升"))
        if row["airflow_m3ph"] < 12500:
            drivers.append(("風量偏低", "排濕能力不足 → RH容易上升"))
        if row["steam_util"] > 1.05:
            drivers.append(("蒸氣效率偏低", "可能漏風/散熱/疏水不良"))
        if not drivers:
            drivers = [("狀態良好", "維持參數並監控漂移")]

        for t, s in drivers[:3]:
            st.write(f"- **{t}**：{s}")

    st.write("---")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("前排風", f"{row['exhaust_front_hz']:.0f} Hz")
    b2.metric("後排風", f"{row['exhaust_back_hz']:.0f} Hz")
    b3.metric("冷卻", f"{row['cooling_hz']:.0f} Hz")
    b4.metric("蒸氣流量", f"{row['steam_kgph']:.0f} kg/h", f"kWh_th/m {row['kwh_th_per_m_req']:.3f}")
    b5.metric("風量", f"{row['airflow_m3ph']:.0f} m³/h", f"排風RH {row['exhaust_rh']*100:.1f}%")
    b6.metric("用電估算", f"{row['power_kw']:.0f} kW", f"kWh_e/m {row['kwh_elec_per_m_est']:.3f}")

    # What-if panel (proposal killer feature)
    st.write("---")
    st.markdown("### 🎯 What‑if：調一個旋鈕，看錢與碳怎麼變")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        d_exh = st.slider("排風 +Hz", -10, 20, 5)
    with w2:
        d_air = st.slider("風量 +m³/h", -5000, 8000, 2000, step=500)
    with w3:
        d_steam = st.slider("蒸氣 +kg/h", -400, 700, 150, step=50)
    with w4:
        d_speed = st.slider("速度 +m/min", -8, 8, 1)

    # heuristic: increasing exhaust Hz & airflow reduces exhaust RH a bit (demo)
    rh_new = float(np.clip(row["exhaust_rh"] - 0.003 * d_exh - 0.000003 * d_air, 0.55, 0.95))
    airflow_new = float(max(5000.0, row["airflow_m3ph"] + d_air))
    steam_new = float(max(200.0, row["steam_kgph"] + d_steam))
    speed_new = float(max(10.0, row["speed_mmin"] + d_speed))

    ph_new = dryer_physics(
        speed_mmin=speed_new,
        steam_kgph=steam_new,
        airflow_m3ph=airflow_new,
        t_in_c=row["inlet_temp_c"],
        rh_in=row["inlet_rh"],
        t_out_c=row["exhaust_temp_c"],
        rh_out=rh_new,
        rho_air=rho_air,
        p_atm=p_atm,
        h_fg_kjkg=h_fg,
        h_steam_kjkg=h_steam,
        cp_air_kjkgk=cp_air,
        eta_base=eta_base,
    )

    # recompute deltas with the same cost settings
    m_per_hr_new = speed_new * 60.0
    kwh_e_new = float((row["power_kw"] / max(m_per_hr_new, 1e-6)))
    energy_nt_new = kwh_e_new * elec_price + ph_new["kwh_th_per_m_req"] * steam_price_proxy
    labor_nt_new = labor_hr / max(m_per_hr_new, 1e-6)
    machine_nt_new = machine_hr / max(m_per_hr_new, 1e-6)
    unit_cost_new = energy_nt_new + labor_nt_new + machine_nt_new + overhead_m
    carbon_new = kwh_e_new * ef_elec + ph_new["kwh_th_per_m_req"] * ef_steam
    carbon_cost_new = (carbon_new / 1000.0) * internal_carbon
    profit_new = row["sell_price_nt_per_m"] - unit_cost_new
    profit_carbon_new = row["sell_price_nt_per_m"] - (unit_cost_new + carbon_cost_new)

    base_profit = float(row["profit_nt_per_m"])
    base_profit_c = float(row["profit_with_carbon_nt_per_m"])
    base_c = float(row["carbon_kgco2_per_m"])

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("排風RH（估）", f"{rh_new*100:.1f}%", f"原 {row['exhaust_rh']*100:.1f}%")
    r2.metric("含碳盈虧（新）", f"NT$ {profit_carbon_new:.2f}/m", f"Δ {profit_carbon_new-base_profit_c:+.2f}")
    r3.metric("碳強度（新）", f"{carbon_new:.3f} kgCO₂/m", f"Δ {carbon_new-base_c:+.3f}")
    r4.metric("蒸氣kWh/m（新）", f"{ph_new['kwh_th_per_m_req']:.3f}", f"Δ {ph_new['kwh_th_per_m_req']-row['kwh_th_per_m_req']:+.3f}")

    if show_formulas:
        with st.expander("📐 展開：物理模型公式 + Tag 對應", expanded=False):
            st.markdown("**Tag 對應（現場已具備）**：蒸氣流量 / 排風溫濕度 / 風量 / 布速")
            render_formulas()

# -----------------------------
# Tab 3: Asset view
# -----------------------------
with tabs[2]:
    st.subheader("③ 設備頁（對齊現場第 2 頁：整機部位狀態 → 影響準交/盈虧/能耗）")

    pick = st.selectbox("選擇工單（用當下工單帶出設備影響）", orders["wo"].tolist(), key="asset_pick")
    row = orders[orders["wo"] == pick].iloc[0]
    cards = build_asset_cards(row)

    for _, r in cards.iterrows():
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2.3, 1.0, 1.0, 1.0])
            c1.markdown(f"## {r['health']} {r['subsystem']}")
            # impact scoring (proposal-friendly)
            impact_otd = "🟡" if row["otd"] != "🟢 準交" else "🟢"
            impact_profit = "🔴" if row["profit_with_carbon_nt_per_m"] < 0 else ("🟡" if row["profit_with_carbon_nt_per_m"] < 1 else "🟢")
            impact_energy = "🔴" if row["kwh_th_per_m_req"] > np.percentile(orders["kwh_th_per_m_req"], 70) else ("🟡" if row["kwh_th_per_m_req"] > np.percentile(orders["kwh_th_per_m_req"], 40) else "🟢")
            c2.metric("準交影響", impact_otd)
            c3.metric("盈虧影響", impact_profit)
            c4.metric("能耗影響", impact_energy)

            k1, k2, k3 = st.columns(3)
            if r["kpi1"] == "exhaust_rh":
                k1.write(f"**{r['kpi1']}**：{r['v1']*100:.1f}%")
            else:
                k1.write(f"**{r['kpi1']}**：{r['v1']:.2f}")
            k2.write(f"**{r['kpi2']}**：{r['v2']:.2f}")

            hint = "建議：維持監控。"
            if r["subsystem"] == "Exhaust / Dehumid" and row["exhaust_rh"] > 0.80:
                hint = "建議：提高排風/除濕、檢查排風風道阻塞、降低漏風。"
            elif r["subsystem"] == "Fan & Airflow" and row["airflow_m3ph"] < 12500:
                hint = "建議：檢查風車VFD/皮帶/濾網壓差；風量不足會讓RH上升。"
            elif r["subsystem"] == "Steam / Heater" and row["steam_util"] > 1.05:
                hint = "建議：檢查蒸氣壓/疏水器、保溫、漏風；避免熱被排風帶走。"
            elif r["subsystem"] == "Chain / Tension":
                widths = [row[f"width_mm_{i}"] for i in range(1, 9)]
                if float(np.std(widths)) > 4:
                    hint = "建議：寬度波動偏大，檢查鏈條/夾具磨耗、張力設定、導布。"
            st.caption(hint)

# -----------------------------
# Tab 4: Event timeline
# -----------------------------

with tabs[3]:
    st.subheader("④ 事件時間線：把『設備異常』翻譯成『工單損益原因』（PLC + 機器狗）")

    # init robotdog demo once (PoC mode: CSV/JSON can replace generator)
    if "robotdog_demo" not in st.session_state:
        try:
            lines = orders["line"].unique().tolist()
        except Exception:
            lines = ["LINE-A","LINE-B","LINE-C","LINE-D"]
        st.session_state.robotdog_demo = generate_demo_robotdog_runs(lines=lines)

    robot_obs = st.session_state.robotdog_demo

    pick = st.selectbox("選擇工單", orders["wo"].tolist(), key="event_pick")
    row = orders[orders["wo"] == pick].iloc[0]
    line = str(row["line"])

    # 1) PLC/physics rule events
    events_plc = detect_events(row)
    ev_plc = pd.DataFrame(events_plc) if events_plc else pd.DataFrame(columns=["ts","severity","event","subsystem","explain","impact_nt_per_m"])
    if len(ev_plc):
        ev_plc["line"] = line
        ev_plc["evidence_uri"] = "plc://"

    # 2) Robot dog events (filtered)
    win_hours = st.slider("巡檢事件回溯(小時)", 1, 72, 6)
    tmin = dt.datetime.now() - dt.timedelta(hours=int(win_hours))
    obs_line = robot_obs[(robot_obs["line"] == line) & (robot_obs["ts"] >= tmin)].copy()
    ev_robot = robotdog_to_events(obs_line)

    # merge
    ev_all = pd.concat([ev_plc, ev_robot], ignore_index=True, sort=False).sort_values("ts", ascending=True)

    if len(ev_all) == 0:
        st.success("✅ 目前未偵測到事件（PLC + 機器狗）。")
    else:
        show_cols = ["ts","severity","event","subsystem","explain","impact_nt_per_m","line","evidence_uri"]
        st.dataframe(ev_all[show_cols], use_container_width=True, hide_index=True)

        total_impact = float(ev_all["impact_nt_per_m"].sum()) if "impact_nt_per_m" in ev_all.columns else 0.0
        st.metric("預估損益影響（每米）", f"NT$ {total_impact:.2f}/m")

        st.markdown("### ✅ 建議動作（按影響排序）")
        ev2 = ev_all.sort_values("impact_nt_per_m")
        for _, e in ev2.iterrows():
            st.write(f"- {e['severity']} **{e['event']}**（{e['subsystem']}）[{e.get('evidence_uri','')}]：{e['explain']}｜影響≈ {float(e['impact_nt_per_m']):.2f} NT$/m")
with tabs[4]:
    st.subheader("⑤ AR 眼鏡（現場即時疊加）")

    st.markdown("""
🕶️ **AR 現場情境**
- 操作人員戴上 AR 眼鏡（HoloLens / RealWear）
- 看向定型機時，即時疊加：
  - 工單 / 準交
  - 每米盈虧（含碳）
  - 關鍵部位狀態（排風 / 風量 / 蒸氣）
""")

    pick = st.selectbox("模擬 AR 眼鏡目前看到的工單", orders["wo"].tolist(), key="ar_pick")
    row = orders[orders["wo"] == pick].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("工單", row["wo"])
    c2.metric("準交", row["otd"])
    c3.metric("每米盈虧（含碳）", f"NT$ {row['profit_with_carbon_nt_per_m']:.2f}/m",
              "🔴 燒錢" if row["profit_with_carbon_nt_per_m"] < 0 else "🟢 正常")
    c4.metric("碳強度", f"{row['carbon_kgco2_per_m']:.3f} kgCO₂/m")

    st.write("---")
    d1, d2, d3 = st.columns(3)
    d1.metric("排風 RH", f"{row['exhaust_rh']*100:.1f}%", "⚠ 偏高" if row["exhaust_rh"] > 0.8 else "正常")
    d2.metric("風量", f"{row['airflow_m3ph']:.0f} m³/h", "⚠ 偏低" if row["airflow_m3ph"] < 12500 else "正常")
    d3.metric("蒸氣利用比", f"{row['steam_util']:.2f}", "⚠ 浪費" if row["steam_util"] > 1.1 else "正常")

    if row["profit_with_carbon_nt_per_m"] < 0:
        st.error("❌ 即時判斷：這一米正在燒錢 → 建議立即調整排風/風量/速度")
    else:
        st.success("✅ 即時判斷：製程穩定")

# -----------------------------
# Tab 6: AR + Real-time margin
# -----------------------------
with tabs[5]:
    st.subheader("⑥ AR（應收帳款）+ 即時成本盈虧（把『收款』跟『工單損益』綁在一起）")

    c1, c2, c3, c4 = st.columns(4)
    total_ar = float(ar["ar_amount_nt"].sum())
    overdue_ar = float(ar.loc[ar["days_overdue"] > 0, "ar_amount_nt"].sum())
    high_risk_ar = float(ar.loc[ar["risk_score"] >= 0.70, "ar_amount_nt"].sum())
    ar_profit = float(ar["ar_profit_with_carbon_nt"].sum())

    c1.metric("AR 總額", f"NT$ {total_ar:,.0f}")
    c2.metric("逾期 AR", f"NT$ {overdue_ar:,.0f}", f"{(overdue_ar/max(total_ar,1e-6))*100:.1f}%")
    c3.metric("高風險 AR", f"NT$ {high_risk_ar:,.0f}", f"{(high_risk_ar/max(total_ar,1e-6))*100:.1f}%")
    c4.metric("AR 對應『含碳盈虧』", f"NT$ {ar_profit:,.0f}")

    st.markdown("---")
    f1, f2, f3 = st.columns([1.2, 1.0, 1.2])
    with f1:
        cust = st.selectbox("客戶", ["ALL"] + sorted(ar["customer"].unique().tolist()))
    with f2:
        bucket = st.selectbox("帳齡", ["ALL", "current", "0-7", "1-30", "90+"])
    with f3:
        risk = st.selectbox("風險", ["ALL", "🟢 正常", "🟡 注意", "🔴 高風險"])

    view = ar.copy()
    if cust != "ALL":
        view = view[view["customer"] == cust]
    if bucket != "ALL":
        view = view[view["bucket"] == bucket]
    if risk != "ALL":
        view = view[view["risk"] == risk]

    view = view.sort_values(["risk_score", "days_overdue", "ar_amount_nt"], ascending=[False, False, False])

    show = view[[
        "risk", "customer", "market", "invoice_no", "wo", "barcode",
        "invoice_date", "due_date", "days_overdue",
        "invoice_amount_nt", "paid_amount_nt", "ar_amount_nt",
        "profit_with_carbon_and_shelf_nt_per_m", "shelf_loss_nt_per_m", "carbon_kgco2_per_m",
        "ar_profit_with_carbon_nt"
    ]].copy()

    show = show.rename(columns={
        "risk": "風險", "customer": "客戶", "market": "市場", "invoice_no": "發票號",
        "wo": "工單", "barcode": "Barcode",
        "invoice_date": "開立日", "due_date": "到期日", "days_overdue": "逾期(天)",
        "invoice_amount_nt": "發票金額(NT$)", "paid_amount_nt": "已收(NT$)", "ar_amount_nt": "未收AR(NT$)",
        "profit_with_carbon_and_shelf_nt_per_m": "含碳+庫齡盈虧(NT$/m)", "shelf_loss_nt_per_m":"庫齡損失(NT$/m)", "carbon_kgco2_per_m": "kgCO2/m",
        "ar_profit_with_carbon_nt": "AR對應含碳盈虧(NT$)"
    })

    # ---- Customer level: Produced cloth x AR linkage (management view)
    if cust != "ALL":
        oc = orders[orders["customer"] == cust].copy()
        produced_m = float(oc["done_m"].sum())
        produced_profit = float((oc["done_m"] * oc["profit_with_carbon_and_shelf_nt_per_m"]).sum())
        produced_carbon_t = float((oc["done_m"] * oc["carbon_kgco2_per_m"]).sum() / 1000.0)

        ac = ar[ar["customer"] == cust].copy()
        ar_amt = float(ac["ar_amount_nt"].sum())
        overdue_amt = float(ac.loc[ac["days_overdue"] > 0, "ar_amount_nt"].sum())
        risk_avg = float(ac["risk_score"].mean()) if len(ac) else 0.0

        st.markdown("### 🔗 已生產的布 × 未收回的錢（同一個客戶）")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("已生產(米)", f"{produced_m:,.0f}")
        s2.metric("對應含碳+庫齡盈虧(NT$)", f"{produced_profit:,.0f}")
        s3.metric("未收 AR(NT$)", f"{ar_amt:,.0f}", f"逾期 {overdue_amt:,.0f}")
        s4.metric("AR 風險平均", f"{risk_avg:.2f}", "🔴" if risk_avg >= 0.70 else ("🟡" if risk_avg >= 0.45 else "🟢"))

        st.caption(f"碳量（估）：{produced_carbon_t:.2f} tCO₂e（以 kgCO₂/m × 已生產米數估算）")

    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("### 📌 AR 風險與改善建議（提案用）")
    st.write("- 把『逾期/高風險 AR』與『含碳後仍為負的工單』交叉，能快速定位：**哪些訂單在燒錢且回款慢**。")
    st.write("- 對 EU：可把 kgCO₂/m 轉成 CBAM 成本情境，做『報價/條款/融資』調整。")
    st.write("- 上線後：AR 直接由 ERP（Invoice/Receipt）餵入；工單損益與碳成本由 MES/PLC 即時更新。")

    with st.expander("⬇️ 匯出（CSV）", expanded=False):
        csv = show.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下載 AR + 含碳盈虧清單 CSV", data=csv, file_name="AR_profit_carbon_demo.csv", mime="text/csv")




with tabs[7]:
    st.subheader("⑧ 機器狗巡檢（Robot Dog）— 定型機四線智慧巡檢整合（PoC: CSV）")
    st.caption("升級版：RobotDog 巡檢事件 → 自動生成維修工單（含 PR/PO），並可點開 evidence（圖/熱像/音檔）。")

    # -----------------------------
    # PoC CSV ingestion
    # -----------------------------
    def load_robotdog_csv(uploaded) -> pd.DataFrame:
        df = pd.read_csv(uploaded)
        # normalize column names
        df.columns = [c.strip() for c in df.columns]
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        elif "timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            # fallback: now
            df["ts"] = dt.datetime.now()

        # required defaults
        for c, default in [
            ("line", "LINE-A"),
            ("subsystem", "Fan & Airflow"),
            ("anomaly_type", "UNKNOWN"),
            ("severity", "🟢"),
            ("ir_max_c", 0.0),
            ("noise_db", 0.0),
            ("vib_rms", 0.0),
            ("gas_ppm", 0.0),
            ("confidence", 0.75),
            ("evidence_uri", ""),
        ]:
            if c not in df.columns:
                df[c] = default

        # allow evidence columns (optional)
        for c in ["evidence_image", "evidence_thermal", "evidence_audio"]:
            if c not in df.columns:
                df[c] = ""

        # coerce dtypes
        for c in ["ir_max_c","noise_db","vib_rms","gas_ppm","confidence"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # sort
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        return df

    st.markdown("### 1) 資料接入（CSV）")
    st.write("上傳 RobotDog CSV（可由機器狗每次巡檢輸出）。若未上傳，使用內建 Demo。")
    csv_up = st.file_uploader("上傳 RobotDog 巡檢 CSV", type=["csv"], key="rd_csv")

    # -----------------------------
    # Evidence ingestion (upload files)
    # -----------------------------
    st.markdown("### 2) Evidence 上傳（圖 / 熱像 / 音檔）")
    st.write("可上傳多個檔案。CSV 的 evidence_image/evidence_thermal/evidence_audio 欄位填檔名即可對應。")
    ev_files = st.file_uploader(
        "上傳 evidence 檔案（jpg/png/webp/wav/mp3）",
        type=["jpg","jpeg","png","webp","wav","mp3","m4a"],
        accept_multiple_files=True,
        key="rd_ev_files"
    )
    if "rd_evidence_store" not in st.session_state:
        st.session_state.rd_evidence_store = {}  # filename -> bytes
    if ev_files:
        for f in ev_files:
            st.session_state.rd_evidence_store[f.name] = f.getvalue()

    # -----------------------------
    # Load observations
    # -----------------------------
    if csv_up is not None:
        try:
            obs = load_robotdog_csv(csv_up)
            st.success(f"✅ 已載入 CSV：{len(obs)} 筆巡檢觀測")
        except Exception as e:
            st.error(f"CSV 解析失敗：{e}")
            obs = pd.DataFrame()
    else:
        if "robotdog_demo" not in st.session_state:
            try:
                lines = orders["line"].unique().tolist()
            except Exception:
                lines = ["LINE-A","LINE-B","LINE-C","LINE-D"]
            st.session_state.robotdog_demo = generate_demo_robotdog_runs(lines=lines)
            # add evidence columns for demo
            st.session_state.robotdog_demo["evidence_image"] = ""
            st.session_state.robotdog_demo["evidence_thermal"] = ""
            st.session_state.robotdog_demo["evidence_audio"] = ""
        obs = st.session_state.robotdog_demo.copy()

    if len(obs) == 0:
        st.info("尚無巡檢資料。請上傳 CSV 或使用 Demo。")
        st.stop()

    # -----------------------------
    # Filters
    # -----------------------------
    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.6, 1.0])
    with c1:
        line_pick = st.selectbox("線別", ["ALL"] + sorted(obs["line"].unique().tolist()), key="rd_line2")
    with c2:
        sev_pick = st.selectbox("嚴重度", ["ALL","🔴","🟡","🟢"], key="rd_sev2")
    with c3:
        q = st.text_input("搜尋（anomaly/subsystem）", "", key="rd_q2")
    with c4:
        hrs = st.number_input("回溯(小時)", min_value=1, max_value=168, value=24, step=1, key="rd_hrs2")

    tmin = dt.datetime.now() - dt.timedelta(hours=int(hrs))
    view = obs[obs["ts"] >= tmin].copy()

    if line_pick != "ALL":
        view = view[view["line"] == line_pick]
    if sev_pick != "ALL":
        view = view[view["severity"] == sev_pick]
    if q:
        ql = q.lower()
        view = view[
            view["anomaly_type"].astype(str).str.lower().str.contains(ql) |
            view["subsystem"].astype(str).str.lower().str.contains(ql)
        ]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("巡檢事件數", f"{len(view)}")
    s2.metric("🔴 嚴重", f"{int((view['severity']=='🔴').sum())}")
    s3.metric("🟡 警告", f"{int((view['severity']=='🟡').sum())}")
    s4.metric("平均信心", f"{float(view['confidence'].mean() if len(view) else 0):.2f}")

    st.markdown("### 3) 巡檢觀測列表")
    st.dataframe(
        view.sort_values(["severity","confidence","ts"], ascending=[True, False, False]),
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Findings -> Events
    # -----------------------------
    st.markdown("### 4) Findings → 事件（可決策）")
    ev = robotdog_to_events(view)
    if len(ev) == 0:
        st.info("此時間窗內無巡檢事件。")
        st.stop()

    # Mapping: anomaly -> suggested action & PR items
    ACTION_MAP = {
        "STEAM_LEAK": ("檢查蒸氣管路/閥件，確認洩漏點並更換墊片/閥件", [("蒸氣管墊片/密封件", 2, 800), ("耐熱束帶/保溫材料", 1, 1200)]),
        "DUCT_BLOCKAGE": ("檢查風道/濾網堵塞，安排清潔與更換濾材", [("濾網/濾材", 4, 600), ("清潔耗材", 1, 500)]),
        "BEARING_NOISE": ("軸承異音：潤滑/校正，必要時更換軸承", [("軸承", 2, 2500), ("潤滑脂", 1, 450)]),
        "ABNORMAL_VIB": ("震動異常：檢查不平衡/鬆動，校正並緊固", [("固定螺栓/止鬆", 1, 300), ("動平衡/校正服務", 1, 6000)]),
        "BELT_SLIP": ("皮帶打滑：調整張力/更換皮帶", [("傳動皮帶", 2, 1800)]),
        "HOTSPOT_PANEL": ("電控箱熱點：檢查接點/負載，緊固並做熱影像複測", [("端子/接觸器", 1, 3500)]),
        "OIL_LEAK": ("漏油：確認油封/管件，清潔並更換油封", [("油封", 2, 900), ("吸油棉/清潔耗材", 1, 400)]),
        "OBSTACLE": ("安全：清除走道障礙物並加強區域標示", [("安全警示貼/圍欄", 1, 800)]),
        "UNKNOWN": ("請工程師現場複核，必要時加測", [("現場檢修工時", 1, 0)]),
    }

    def event_to_ticket_row(r: pd.Series) -> dict:
        at = str(r["event"])
        action, items = ACTION_MAP.get(at, ACTION_MAP["UNKNOWN"])
        # priority: 🔴 P1, 🟡 P2, 🟢 P3
        prio = {"🔴":"P1", "🟡":"P2", "🟢":"P3"}.get(str(r["severity"]), "P3")
        est_downtime = {"P1": 60, "P2": 30, "P3": 10}[prio]
        est_cost = 0.0
        for _, qty, unit in items:
            est_cost += float(qty) * float(unit)
        return {
            "ticket_id": "",
            "created_ts": dt.datetime.now(),
            "line": str(r.get("line","LINE-A")),
            "subsystem": str(r.get("subsystem","")),
            "issue": at,
            "severity": str(r.get("severity","🟢")),
            "priority": prio,
            "suggested_action": action,
            "impact_nt_per_m": float(r.get("impact_nt_per_m", 0.0)),
            "est_downtime_min": int(est_downtime),
            "est_material_cost_nt": float(est_cost),
            "status": "OPEN",
            "pr_id": "",
            "po_id": "",
            "evidence_image": str(r.get("evidence_image","")),
            "evidence_thermal": str(r.get("evidence_thermal","")),
            "evidence_audio": str(r.get("evidence_audio","")),
            "evidence_uri": str(r.get("evidence_uri","")),
        }

    # Attach confidence/signals/evidence from observations to event rows (best-effort)
    # In production, use observation_id / run_id for exact linkage.
    tmp_cols = ["ts","line","anomaly_type","confidence","ir_max_c","noise_db","vib_rms","gas_ppm","evidence_uri",
                "evidence_image","evidence_thermal","evidence_audio"]
    tmp = view[[c for c in tmp_cols if c in view.columns]].copy()
    tmp = tmp.rename(columns={"anomaly_type":"event"})
    ev = ev.merge(tmp, on=["ts","line","event"], how="left")

    # -----------------------------
    # Work Orders + PR/PO state
    # -----------------------------
    if "rd_tickets" not in st.session_state:
        st.session_state.rd_tickets = pd.DataFrame(columns=[
            "ticket_id","created_ts","line","subsystem","issue","severity","priority","suggested_action",
            "impact_nt_per_m","est_downtime_min","est_material_cost_nt","status","pr_id","po_id",
            "evidence_image","evidence_thermal","evidence_audio","evidence_uri"
        ])
    if "rd_pr" not in st.session_state:
        st.session_state.rd_pr = pd.DataFrame(columns=["pr_id","ticket_id","created_ts","status","item","qty","unit_cost_nt","amount_nt"])
    if "rd_po" not in st.session_state:
        st.session_state.rd_po = pd.DataFrame(columns=["po_id","pr_id","created_ts","vendor","status","amount_nt"])

    # -----------------------------
    # Auto-generate tickets
    # -----------------------------
    st.markdown("### 5) 自動生成維修工單（含 PR/PO）")
    colA, colB, colC = st.columns([1.1, 1.1, 2.0])
    with colA:
        min_conf = st.slider("最小信心門檻", 0.0, 0.99, 0.60, 0.01, key="rd_min_conf")
    with colB:
        include_green = st.checkbox("包含🟢（建議不勾）", value=False, key="rd_inc_green")
    with colC:
        st.caption("規則：🟡/🔴 且 confidence≥門檻 → 生成 Ticket；每個 Ticket 會自動生成 PR（材料需求），PR 可再轉 PO。")

    def _next_id(prefix: str, df: pd.DataFrame) -> str:
        if len(df) == 0:
            return f"{prefix}-0001"
        nums = []
        for x in df.iloc[:,0].astype(str).tolist():
            if x.startswith(prefix+"-"):
                try:
                    nums.append(int(x.split("-")[-1]))
                except Exception:
                    pass
        n = (max(nums) + 1) if nums else 1
        return f"{prefix}-{n:04d}"

    def generate_tickets_and_pr(ev_df: pd.DataFrame):
        # filter
        f = ev_df.copy()
        if not include_green:
            f = f[f["severity"].isin(["🟡","🔴"])]
        f = f[f["confidence"].astype(float) >= float(min_conf)] if "confidence" in f.columns else f

        if len(f) == 0:
            return 0

        tickets_new = []
        pr_new = []
        for _, r in f.iterrows():
            t = event_to_ticket_row(r)
            t["ticket_id"] = _next_id("MT", st.session_state.rd_tickets)
            tickets_new.append(t)

            # PR items
            items = ACTION_MAP.get(str(r["event"]), ACTION_MAP["UNKNOWN"])[1]
            pr_id = _next_id("PR", st.session_state.rd_pr)
            for item, qty, unit in items:
                amount = float(qty) * float(unit)
                pr_new.append({
                    "pr_id": pr_id,
                    "ticket_id": t["ticket_id"],
                    "created_ts": dt.datetime.now(),
                    "status": "DRAFT",
                    "item": item,
                    "qty": int(qty),
                    "unit_cost_nt": float(unit),
                    "amount_nt": float(amount),
                })
            # link ticket -> PR
            tickets_new[-1]["pr_id"] = pr_id

        st.session_state.rd_tickets = pd.concat([st.session_state.rd_tickets, pd.DataFrame(tickets_new)], ignore_index=True)
        st.session_state.rd_pr = pd.concat([st.session_state.rd_pr, pd.DataFrame(pr_new)], ignore_index=True)
        return len(tickets_new)

    gen_btn = st.button("🤖 一鍵生成維修工單 + PR（依規則）", type="primary", use_container_width=True, key="rd_gen")
    if gen_btn:
        nnew = generate_tickets_and_pr(ev)
        st.success(f"✅ 已新增 {nnew} 筆維修工單（並建立對應 PR）")

    # -----------------------------
    # Tickets table + selection
    # -----------------------------
    st.markdown("### 6) 維修工單（Maintenance Tickets）")
    tdf = st.session_state.rd_tickets.copy()
    if len(tdf) == 0:
        st.info("尚無維修工單。請先按『一鍵生成』。")
    else:
        st.dataframe(
            tdf.sort_values(["priority","created_ts"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True
        )

        sel = st.selectbox("選擇一筆工單查看 evidence / PR / PO", tdf["ticket_id"].astype(str).tolist(), key="rd_ticket_sel")
        trow = tdf[tdf["ticket_id"].astype(str) == str(sel)].iloc[0].to_dict()

        st.markdown("#### Evidence（點開查看）")
        evc1, evc2, evc3 = st.columns(3)

        def _render_evidence(col, fname: str, kind: str):
            if not fname:
                col.info("（無）")
                return
            store = st.session_state.rd_evidence_store
            if fname not in store:
                col.warning(f"找不到檔案：{fname}（請上傳 evidence）")
                return
            b = store[fname]
            if kind in ("image","thermal"):
                col.image(b, caption=fname, use_container_width=True)
            else:
                col.audio(b)

        with evc1:
            st.write("📷 可見光")
            _render_evidence(evc1, str(trow.get("evidence_image","")), "image")
        with evc2:
            st.write("🌡️ 熱像")
            _render_evidence(evc2, str(trow.get("evidence_thermal","")), "thermal")
        with evc3:
            st.write("🔊 音檔")
            _render_evidence(evc3, str(trow.get("evidence_audio","")), "audio")

        st.markdown("#### PR（請購）")
        pr_id = str(trow.get("pr_id",""))
        pr_df = st.session_state.rd_pr
        pr_view = pr_df[pr_df["pr_id"].astype(str) == pr_id].copy() if pr_id else pd.DataFrame()
        if len(pr_view) == 0:
            st.info("此工單尚無 PR。")
        else:
            st.dataframe(pr_view, use_container_width=True, hide_index=True)
            total_amt = float(pr_view["amount_nt"].sum())
            st.metric("PR 金額合計", f"NT$ {total_amt:,.0f}")

            # Approve PR -> create PO
            cA, cB = st.columns([1.2, 2.0])
            with cA:
                vendor = st.text_input("PO 廠商（示例）", "Default Vendor", key="rd_vendor")
            with cB:
                st.caption("流程：PR(DRAFT) → Approve → PO(OPEN)。PoC 版先用按鈕模擬。")

            approve = st.button("✅ Approve PR → Create PO", use_container_width=True, key="rd_approve_pr")
            if approve:
                # update PR status
                st.session_state.rd_pr.loc[st.session_state.rd_pr["pr_id"].astype(str) == pr_id, "status"] = "APPROVED"

                # create PO
                po_id = _next_id("PO", st.session_state.rd_po)
                st.session_state.rd_po = pd.concat([st.session_state.rd_po, pd.DataFrame([{
                    "po_id": po_id,
                    "pr_id": pr_id,
                    "created_ts": dt.datetime.now(),
                    "vendor": vendor,
                    "status": "OPEN",
                    "amount_nt": total_amt,
                }])], ignore_index=True)

                # link ticket -> PO
                st.session_state.rd_tickets.loc[st.session_state.rd_tickets["ticket_id"].astype(str) == str(sel), "po_id"] = po_id

                st.success(f"✅ 已建立 PO：{po_id}")

        st.markdown("#### PO（採購單）")
        po_id = str(st.session_state.rd_tickets.loc[st.session_state.rd_tickets["ticket_id"].astype(str) == str(sel), "po_id"].iloc[0] or "")
        if po_id:
            po_view = st.session_state.rd_po[st.session_state.rd_po["po_id"].astype(str) == po_id].copy()
            st.dataframe(po_view, use_container_width=True, hide_index=True)
        else:
            st.info("此工單尚未建立 PO（請先 Approve PR）。")

        st.markdown("### 7) ERP 匯出（Excel / PDF）")
        st.caption("將目前 RobotDog 維修工單、PR、PO 匯出成 ERP 可接收的檔案格式（PoC：Excel 多工作表 + PDF 報表）。")

        exp_c1, exp_c2, exp_c3 = st.columns([1.1, 1.1, 2.0])
        with exp_c1:
            xls_bytes = build_erp_excel(st.session_state.rd_tickets, st.session_state.rd_pr, st.session_state.rd_po)
            st.download_button(
                "📥 下載 ERP Excel（Tickets+PR+PO）",
                data=xls_bytes,
                file_name=f"YuYuan_ERP_Export_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="rd_export_excel",
            )

        with exp_c2:
            if _REPORTLAB_OK:
                pdf_bytes = build_erp_pdf(st.session_state.rd_tickets, st.session_state.rd_pr, st.session_state.rd_po)
                st.download_button(
                    "📄 下載 ERP PDF（Tickets+PR+PO）",
                    data=pdf_bytes,
                    file_name=f"YuYuan_ERP_Export_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="rd_export_pdf",
                )
            else:
                st.warning("此環境未安裝 reportlab，無法輸出 PDF（Excel 可用）。")

        with exp_c3:
            st.markdown("**ERP 欄位建議（後續串接）**")
            st.write("- Tickets：ticket_id / created_ts / line / subsystem / issue / severity / priority / status / pr_id / po_id")
            st.write("- PR：pr_id / ticket_id / item / qty / unit_cost_nt / amount_nt / status")
            st.write("- PO：po_id / pr_id / vendor / amount_nt / status")

