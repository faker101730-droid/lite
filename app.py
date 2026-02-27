import io
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
try:
    from streamlit_plotly_events import plotly_events
except Exception:
    plotly_events = None  # optional dependency


APP_TITLE = "DPC査定分析 v3.4"
REQUIRED_COLS = ["月","区分","入院種別","診療科","査定理由カテゴリ","注意項目","査定額","件数","請求額"]

LOCAL_STORE_DIR = "data_store"
LOCAL_STORE_FILE = os.path.join(LOCAL_STORE_DIR, "latest.xlsx")

@dataclass
class Settings:
    sensitivity: str = "standard"  # low|standard|high
    top_n_amount: int = 20
    top_n_increase: int = 20
    min_amount: int = 100000
    min_count: int = 3
    z_threshold: float = 2.0
    w_amount: int = 2
    w_increase: int = 1
    w_rate: int = 1
    breakdown_topn: int = 12

def fmt_month(p: pd.Period) -> str:
    try:
        return f"{int(p.year)}/{int(p.month):02d}"
    except Exception:
        return str(p)

def parse_month(x) -> pd.Period:
    if pd.isna(x):
        return pd.NaT
    s = str(x)
    try:
        return pd.Period(s[:7], freq="M")
    except Exception:
        try:
            return pd.Period(pd.to_datetime(x).strftime("%Y-%m"), freq="M")
        except Exception:
            return pd.NaT

def save_local_excel(file_bytes: bytes):
    os.makedirs(LOCAL_STORE_DIR, exist_ok=True)
    with open(LOCAL_STORE_FILE, "wb") as f:
        f.write(file_bytes)

def load_local_excel() -> bytes | None:
    if os.path.exists(LOCAL_STORE_FILE):
        with open(LOCAL_STORE_FILE, "rb") as f:
            return f.read()
    return None

def clear_local_excel():
    if os.path.exists(LOCAL_STORE_FILE):
        os.remove(LOCAL_STORE_FILE)

def load_excel(file_bytes: bytes) -> tuple[pd.DataFrame, Settings]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    if "data" not in xls.sheet_names:
        raise ValueError("Excelに 'data' シートが見つかりません。")
    df = pd.read_excel(xls, "data")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"必須列が不足しています: {missing}")

    df = df.copy()
    df["月"] = df["月"].apply(parse_month)
    df = df.dropna(subset=["月"])

    for c in ["査定額","件数","請求額"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in ["区分","入院種別","診療科","査定理由カテゴリ","注意項目"]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    s = Settings()
    if "settings" in xls.sheet_names:
        s_df = pd.read_excel(xls, "settings")
        if set(["key","value"]).issubset(s_df.columns):
            d = {str(k): str(v) for k,v in zip(s_df["key"], s_df["value"])}
            s.sensitivity = d.get("sensitivity", s.sensitivity)
            s.top_n_amount = int(float(d.get("top_n_amount", s.top_n_amount)))
            s.top_n_increase = int(float(d.get("top_n_increase", s.top_n_increase)))
            s.min_amount = int(float(d.get("min_amount", s.min_amount)))
            s.min_count = int(float(d.get("min_count", s.min_count)))
            s.z_threshold = float(d.get("z_threshold", s.z_threshold))
            s.w_amount = int(float(d.get("w_amount", s.w_amount)))
            s.w_increase = int(float(d.get("w_increase", s.w_increase)))
            s.w_rate = int(float(d.get("w_rate", s.w_rate)))
            s.breakdown_topn = int(float(d.get("breakdown_topn", s.breakdown_topn)))
    return df, s

def compute_monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["月","区分","入院種別","診療科"], as_index=False).agg(
        査定額=("査定額","sum"),
        件数=("件数","sum"),
        請求額=("請求額","max"),
    )
    g["査定率"] = np.where(g["請求額"]>0, g["査定額"]/g["請求額"], 0.0)
    return g

def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd

def segment_key(segment: str):
    if segment == "外来":
        return "外来", ""
    if segment == "入院DPC":
        return "入院", "DPC"
    if segment == "入院出来高":
        return "入院", "出来高"
    raise ValueError("unknown segment")

def fiscal_range(latest: pd.Period):
    latest_dt = latest.to_timestamp()
    fy_start = pd.Period(f"{latest_dt.year-1 if latest_dt.month<4 else latest_dt.year}-04", freq="M")
    return fy_start, latest

def build_scope(df: pd.DataFrame, monthly: pd.DataFrame, kubun: str, nyuin_type: str,
                dept_mode: str, dept: str | None):
    ddf = df[(df["区分"]==kubun) & (df["入院種別"]==nyuin_type)].copy()
    msc = monthly[(monthly["区分"]==kubun) & (monthly["入院種別"]==nyuin_type)].copy()
    if dept_mode == "診療科別" and dept:
        ddf = ddf[ddf["診療科"]==dept]
        msc = msc[msc["診療科"]==dept]
    return ddf, msc

def score_alerts(ddf: pd.DataFrame, msc: pd.DataFrame, period_mode: str, s: Settings) -> pd.DataFrame:
    all_months = msc["月"].sort_values().unique()
    if len(all_months)==0:
        return pd.DataFrame()
    latest = all_months[-1]

    if period_mode == "最新月":
        df_p = ddf[ddf["月"]==latest]
        prev = all_months[-2] if len(all_months)>=2 else None
        df_prev = ddf[ddf["月"]==prev] if prev is not None else ddf.iloc[0:0]
    else:
        fy_start, _ = fiscal_range(latest)
        df_p = ddf[(ddf["月"]>=fy_start) & (ddf["月"]<=latest)]
        df_prev = ddf.iloc[0:0]

    cur = df_p.groupby(["査定理由カテゴリ","注意項目"], as_index=False).agg(
        査定額=("査定額","sum"), 件数=("件数","sum"), 請求額=("請求額","max")
    )
    cur["査定率"] = np.where(cur["請求額"]>0, cur["査定額"]/cur["請求額"], 0.0)
    cur = cur[(cur["査定額"]>=s.min_amount) & (cur["件数"]>=s.min_count)].copy()
    if cur.empty:
        return cur

    if period_mode=="最新月" and not df_prev.empty:
        prev_tbl = df_prev.groupby(["査定理由カテゴリ","注意項目"], as_index=False).agg(査定額=("査定額","sum"))
        cur = cur.merge(prev_tbl, on=["査定理由カテゴリ","注意項目"], how="left", suffixes=("","_前月")).fillna({"査定額_前月":0})
        cur["増加額"] = cur["査定額"] - cur["査定額_前月"]
    else:
        cur["増加額"] = 0.0

    r_amt = cur["査定額"].rank(method="min", ascending=False)
    cur["p_amount"] = np.where(r_amt <= s.top_n_amount, 2, np.where(r_amt <= s.top_n_amount*2, 1, 0))

    if period_mode=="最新月":
        r_inc = cur["増加額"].rank(method="min", ascending=False)
        cur["p_increase"] = np.where(r_inc <= s.top_n_increase, 2, np.where(r_inc <= s.top_n_increase*2, 1, 0))
    else:
        cur["p_increase"] = 0

    th = s.z_threshold
    if s.sensitivity=="high":
        th = max(1.2, th-0.5)
    elif s.sensitivity=="low":
        th = th+0.5

    cur["z_rate"] = zscore(cur["査定率"])
    cur["p_rate"] = (cur["z_rate"] >= th).astype(int)*2

    cur["score"] = s.w_amount*cur["p_amount"] + s.w_increase*cur["p_increase"] + s.w_rate*cur["p_rate"]
    cur["レベル"] = np.select([cur["score"]>=6, cur["score"]>=3], ["🔴危険","🟠要注意"], default="🟡観察")
    cur = cur.sort_values(["score","査定額"], ascending=False).reset_index(drop=True)
    return cur

def monthly_scope(msc: pd.DataFrame, dept_mode: str):
    if dept_mode=="診療科別":
        return msc.sort_values("月")
    g = msc.groupby("月", as_index=False).agg(査定額=("査定額","sum"),件数=("件数","sum"),請求額=("請求額","sum"))
    g["査定率"] = np.where(g["請求額"]>0, g["査定額"]/g["請求額"], 0.0)
    return g.sort_values("月")

def build_mix_fig(chart_df: pd.DataFrame, title: str):
    """
    査定額（棒）× 査定率（折れ線）の混合グラフ。
    重要：左軸（円）と右軸（％）で「0の高さ」を揃える（0位置ズレ防止）。
    """
    x = [fmt_month(p) for p in chart_df["月"]]
    y_amt = chart_df["査定額"].astype(float).tolist()
    y_rate = (chart_df["査定率"].astype(float) * 100).tolist()

    # --- range utils: align the "0" position between y and y2 ---
    def _finite(vals):
        a = np.asarray(vals, dtype=float)
        a = a[np.isfinite(a)]
        return a

    def _pad_range(vmin, vmax, pad=0.08):
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (-1.0, 1.0)
        if vmin == vmax:
            span = max(abs(vmin), 1.0)
            return (vmin - span * 0.5, vmax + span * 0.5)
        span = vmax - vmin
        return (vmin - span * pad, vmax + span * pad)

    a = _finite(y_amt)
    r = _finite(y_rate)

    # Amount axis range (must include 0)
    if a.size == 0:
        a_min, a_max = -1.0, 1.0
    else:
        a_min = float(min(np.min(a), 0.0))
        a_max = float(max(np.max(a), 0.0))
        a_min, a_max = _pad_range(a_min, a_max, pad=0.08)

    a_span = a_max - a_min
    f = 0.0 if a_span == 0 else (0.0 - a_min) / a_span
    f = float(np.clip(f, 0.0, 1.0))

    # Rate axis range (must include 0) and align 0 position
    if r.size == 0:
        r_min_data, r_max_data = 0.0, 1.0
    else:
        r_min_data = float(min(np.min(r), 0.0))
        r_max_data = float(max(np.max(r), 0.0))

    if f <= 1e-9:
        r_min, r_max = 0.0, r_max_data if r_max_data != 0 else 1.0
        r_min, r_max = _pad_range(r_min, r_max, pad=0.10)
        r_min = 0.0  # keep bottom at 0
    elif (1.0 - f) <= 1e-9:
        r_min, r_max = r_min_data if r_min_data != 0 else -1.0, 0.0
        r_min, r_max = _pad_range(r_min, r_max, pad=0.10)
        r_max = 0.0  # keep top at 0
    else:
        need_w1 = (-r_min_data) / f if r_min_data < 0 else 0.0
        need_w2 = (r_max_data) / (1.0 - f) if r_max_data > 0 else 0.0
        w = max(need_w1, need_w2, 1.0)
        w *= 1.10  # padding
        r_min = -f * w
        r_max = (1.0 - f) * w

    # Dynamic left margin to avoid y-axis label cut off (large inpatient values)
    max_abs_amt = max([abs(v) for v in y_amt if np.isfinite(v)] + [0.0])
    digits = len(str(int(max_abs_amt))) if max_abs_amt >= 1 else 1
    l_margin = 90 + max(0, digits - 6) * 7  # grow with digits

    fig = go.Figure()
    fig.add_bar(
        x=x, y=y_amt, name="査定額",
        hovertemplate="%{x}<br>査定額：%{y:,.0f}円<extra></extra>"
    )
    fig.add_scatter(
        x=x, y=y_rate, mode="lines+markers", name="査定率(%)",
        yaxis="y2",
        hovertemplate="%{x}<br>査定率：%{y:.2f}%<extra></extra>"
    )

    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=470,
        margin=dict(l=l_margin, r=95, t=60, b=140),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.28,
            yanchor="top",
            font=dict(size=12),
            traceorder="normal"
        ),
        yaxis=dict(
            title="査定額(円)",
            tickformat=",.0f",
            automargin=True,
            title_standoff=18,
            range=[a_min, a_max],
            fixedrange=True,
        ),
        yaxis2=dict(
            title="査定率(%)",
            overlaying="y",
            side="right",
            tickformat=".2f",
            automargin=True,
            title_standoff=18,
            range=[r_min, r_max],
            fixedrange=True,
        ),
        xaxis=dict(
            title="",
            tickangle=-35,
            automargin=True,
            tickfont=dict(size=11),
            fixedrange=True,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        clickmode="event+select",
        dragmode=False,
    )

    return fig



def plotly_click_x(fig: go.Figure, key: str):
    """
    Plotly上のクリック（月）を取得する。
    - Streamlit本体の on_select が使える場合はそれを優先（ズーム/パンで壊れにくい）。
    - 使えない場合のみ streamlit-plotly-events を使う（入っていれば）。
    """
    import inspect

    # Prefer Streamlit native selection event when available
    try:
        sig = inspect.signature(st.plotly_chart)
        params = sig.parameters
        if "on_select" not in params:
            raise TypeError("plotly on_select not supported")
        kwargs = {"use_container_width": True}
        if "key" in params:
            kwargs["key"] = key
        if "on_select" in params:
            kwargs["on_select"] = "rerun"
        if "selection_mode" in params:
            # points mode enables click selection
            kwargs["selection_mode"] = "points"
        evt = st.plotly_chart(fig, **kwargs)

        # When on_select is supported, selection is returned as a dict-like object
        if isinstance(evt, dict):
            pts = None
            # Streamlit versions vary in structure
            if "selection" in evt and isinstance(evt["selection"], dict):
                pts = evt["selection"].get("points")
            if pts and len(pts) > 0:
                return pts[0].get("x")
        return None
    except TypeError:
        # older Streamlit signature: no on_select / selection_mode
        pass
    except Exception:
        # fall through to component
        pass

    # Fallback: streamlit-plotly-events (if installed)
    if plotly_events is None:
        st.plotly_chart(fig, use_container_width=True, key=key)
        return None

    try:
        clicked = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=430,
            key=key,
        )
        if clicked:
            return clicked[0].get("x")
        return None
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)
        return None
def build_pie(period_filter: pd.DataFrame, title: str, group_col: str = "査定理由カテゴリ"):
    s = (
        period_filter.groupby(group_col, as_index=False)
        .agg(査定額=("査定額","sum"))
        .sort_values("査定額", ascending=False)
    )
    if s.empty:
        return None
    fig = go.Figure(data=[
        go.Pie(labels=s[group_col], values=s["査定額"], textinfo="percent+label")
    ])
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def breakdown_tables(period_filter: pd.DataFrame, s: Settings):
    by_reason = period_filter.groupby("査定理由カテゴリ", as_index=False).agg(査定額=("査定額","sum"), 件数=("件数","sum")).sort_values("査定額", ascending=False)
    by_item = (
        period_filter.groupby(["査定理由カテゴリ","注意項目"], as_index=False)
        .agg(査定額=("査定額","sum"), 件数=("件数","sum"))
        .sort_values("査定額", ascending=False)
        .head(s.breakdown_topn)
    )
    by_dept = (
        period_filter.groupby("診療科", as_index=False)
        .agg(査定額=("査定額","sum"), 件数=("件数","sum"))
        .sort_values("査定額", ascending=False)
        .head(s.breakdown_topn)
    )

    st.markdown("**内訳（理由カテゴリ）**")
    st.dataframe(by_reason, use_container_width=True, hide_index=True)

    st.markdown(f"**内訳（注意項目 Top {s.breakdown_topn}）**")
    st.dataframe(by_item, use_container_width=True, hide_index=True)

    st.markdown(f"**内訳（診療科 Top {s.breakdown_topn}）**")
    st.dataframe(by_dept, use_container_width=True, hide_index=True)

def apply_bytes(raw: bytes):
    df, s0 = load_excel(raw)
    st.session_state["raw_bytes"] = raw
    st.session_state["df"] = df
    st.session_state["settings"] = s0

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    if "raw_bytes" not in st.session_state:
        st.session_state["raw_bytes"] = load_local_excel()
        st.session_state["df"] = None
        st.session_state["settings"] = Settings()

    with st.sidebar:
        st.subheader("データ")
        up = st.file_uploader("Excelを選択（まだ反映しません）", type=["xlsx"])
        if up is not None:
            st.session_state["pending_bytes"] = up.read()
            st.success("ファイルを読み込みました（未反映）")

        colA, colB = st.columns(2)
        with colA:
            if st.button("📌 保存して反映", use_container_width=True):
                raw = st.session_state.get("pending_bytes")
                if raw is None:
                    st.error("先にExcelを選択してね。")
                else:
                    try:
                        apply_bytes(raw)
                        save_local_excel(raw)
                        st.success("保存＆反映しました。")
                    except Exception as e:
                        st.error(str(e))

        with colB:
            if st.button("🔁 保存データを反映", use_container_width=True):
                raw = load_local_excel()
                if raw is None:
                    st.warning("保存データがありません。")
                else:
                    try:
                        apply_bytes(raw)
                        st.success("保存データを反映しました。")
                    except Exception as e:
                        st.error(str(e))

        if st.button("🧹 保存データを削除", use_container_width=True):
            clear_local_excel()
            st.session_state["raw_bytes"] = None
            st.session_state["df"] = None
            st.success("保存データを削除しました。")

        df = st.session_state.get("df")
        if df is None:
            if st.session_state.get("raw_bytes") is not None:
                try:
                    apply_bytes(st.session_state["raw_bytes"])
                    df = st.session_state.get("df")
                except Exception as e:
                    st.error(str(e))
            if df is None:
                st.info("①Excelを選択 → ②『保存して反映』を押してね。")
                st.stop()

        st.divider()
        dept_mode = st.radio("粒度", ["全体","診療科別"], horizontal=True)
        dept = None
        if dept_mode=="診療科別":
            dept = st.selectbox("診療科", sorted(df["診療科"].unique()))
        period_mode = st.radio("期間", ["最新月","累計"], horizontal=True)

        with st.expander("⚙ 判定設定（折りたたみ）", expanded=False):
            s = st.session_state["settings"]
            s.sensitivity = st.select_slider("感度", options=["low","standard","high"], value=s.sensitivity)
            s.top_n_amount = int(st.slider("金額上位N（アラート判定）", 5, 50, int(s.top_n_amount), step=5))
            s.top_n_increase = int(st.slider("増加上位N（最新月のみ）", 5, 50, int(s.top_n_increase), step=5))
            c1,c2 = st.columns(2)
            with c1:
                s.min_amount = int(st.number_input(
                    "除外：査定額（円）未満",
                    min_value=0,
                    max_value=10_000_000_000,
                    value=int(s.min_amount),
                    step=50_000
                ))
            with c2:
                s.min_count = int(st.number_input(
                    "除外：件数 未満",
                    min_value=0,
                    max_value=1_000_000,
                    value=int(s.min_count),
                    step=1
                ))
            s.z_threshold = float(st.slider("査定率Zしきい値", 1.0, 3.5, float(s.z_threshold), step=0.1))
            s.breakdown_topn = int(st.slider("内訳：注意項目/診療科 TopN", 5, 30, int(s.breakdown_topn), step=1))
            w1,w2,w3 = st.columns(3)
            with w1:
                s.w_amount = int(st.number_input("重み：金額", 0, 5, int(s.w_amount)))
            with w2:
                s.w_increase = int(st.number_input("重み：増加", 0, 5, int(s.w_increase)))
            with w3:
                s.w_rate = int(st.number_input("重み：率", 0, 5, int(s.w_rate)))

    df = st.session_state["df"]
    s = st.session_state["settings"]
    monthly = compute_monthly(df)

    tab_out, tab_in = st.tabs(["外来","入院"])

    def render_standard(segment_label: str):
        kubun, nyuin_type = segment_key(segment_label)
        ddf, msc = build_scope(df, monthly, kubun, nyuin_type, dept_mode, dept)
        if msc.empty:
            st.info("この区分のデータがありません。")
            return

        msc2 = monthly_scope(msc, dept_mode)
        latest = msc2["月"].sort_values().unique()[-1]

        if period_mode=="最新月":
            period_label = f"最新月：{fmt_month(latest)}"
            period_ddf = ddf[ddf["月"]==latest]
        else:
            fy_start, _ = fiscal_range(latest)
            period_label = f"累計：{fmt_month(fy_start)}〜{fmt_month(latest)}"
            period_ddf = ddf[(ddf["月"]>=fy_start) & (ddf["月"]<=latest)]

        cur_msc = msc2[msc2["月"]==latest] if period_mode=="最新月" else msc2[(msc2["月"]>=fiscal_range(latest)[0]) & (msc2["月"]<=latest)]
        tot_satei = float(cur_msc["査定額"].sum())
        tot_claim = float(cur_msc["請求額"].sum())
        tot_rate = (tot_satei/tot_claim) if tot_claim>0 else 0.0

        alert_tbl = score_alerts(ddf, msc, period_mode, s)

        st.subheader(f"{segment_label} / {period_label} / {dept_mode}{'' if dept is None else '：'+dept}")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("査定額", f"{tot_satei:,.0f} 円")
        c2.metric("請求額", f"{tot_claim:,.0f} 円")
        c3.metric("査定率", f"{tot_rate*100:.2f} %")
        c4.metric("アラート(🔴/🟠/🟡)",
                  f"{(alert_tbl['レベル']=='🔴危険').sum() if not alert_tbl.empty else 0}/"
                  f"{(alert_tbl['レベル']=='🟠要注意').sum() if not alert_tbl.empty else 0}/"
                  f"{(alert_tbl['レベル']=='🟡観察').sum() if not alert_tbl.empty else 0}")

        t1,t2,t3 = st.tabs(["① 推移（混合）","② 内訳（円＋詳細）","③ 注意項目（アラート）"])

        with t1:
            chart_df = msc2.sort_values("月").copy()
            fig = build_mix_fig(chart_df, title="査定額（棒）× 査定率（折れ線）")

            # クリックした月を「保持」して、詳細表示や一覧表の“ぴくつき”を抑える
            base_key = f"mix_{segment_label}_{dept_mode}_{dept}_{period_mode}"
            sel_state_key = f"{base_key}__sel_month"

            x_clicked = plotly_click_x(fig, key=base_key)
            if x_clicked is not None and str(x_clicked) != "":
                st.session_state[sel_state_key] = str(x_clicked)

            cols_hint = st.columns([1,1,6])
            with cols_hint[0]:
                if st.button("選択解除", key=f"clear_{base_key}"):
                    st.session_state.pop(sel_state_key, None)
            with cols_hint[2]:
                st.caption("※棒グラフをクリックすると、その月の詳細（注意項目/診療科Top）が下に出ます。")("※棒グラフをクリックすると、その月の詳細（注意項目/診療科Top）が下に出ます。")

            show_tbl = chart_df.copy()
            show_tbl["年月"] = show_tbl["月"].apply(fmt_month)
            show_tbl = show_tbl.drop(columns=["月"])
            show_tbl["査定率(%)"] = (show_tbl["査定率"]*100).round(2)
            show_tbl = show_tbl.drop(columns=["査定率"])
            show_tbl["査定額"] = show_tbl["査定額"].round(0).astype(int)
            show_tbl["請求額"] = show_tbl["請求額"].round(0).astype(int)
            show_tbl["件数"] = show_tbl["件数"].round(0).astype(int)
            st.markdown("**推移データ（一覧）**")
            st.dataframe(show_tbl, use_container_width=True, hide_index=True, height=min(520, 40 + 35*(len(show_tbl)+1)))

            # クリック月の詳細（保持した選択で表示）
            sel_x = st.session_state.get(sel_state_key)
            if sel_x:
                month_map = {fmt_month(p): p for p in chart_df["月"].tolist()}
                sel_p = month_map.get(str(sel_x))
                if sel_p is not None:
                    st.markdown(f"**クリック月の詳細：{fmt_month(sel_p)}**")
                    ddm = ddf[ddf["月"]==sel_p]
                    if ddm.empty:
                        st.info("この月の詳細データがありません。")
                    else:
                        top_items = ddm.groupby(["査定理由カテゴリ","注意項目"], as_index=False).agg(
                            査定額=("査定額","sum"), 件数=("件数","sum")
                        ).sort_values("査定額", ascending=False).head(s.breakdown_topn)
                        st.markdown(f"注意項目 Top {s.breakdown_topn}")
                        st.dataframe(top_items, use_container_width=True, hide_index=True, height=min(420, 40 + 35*(len(top_items)+1)))

                        top_dept = ddm.groupby("診療科", as_index=False).agg(
                            査定額=("査定額","sum"), 件数=("件数","sum")
                        ).sort_values("査定額", ascending=False).head(s.breakdown_topn)
                        st.markdown(f"診療科 Top {s.breakdown_topn}")
                        st.dataframe(top_dept, use_container_width=True, hide_index=True, height=min(420, 40 + 35*(len(top_dept)+1)))


        with t2:
            c_pie1, c_pie2 = st.columns(2)
            with c_pie1:
                pie = build_pie(period_ddf, title="査定内訳（理由カテゴリ）", group_col="査定理由カテゴリ")
                if pie is not None:
                    st.plotly_chart(pie, use_container_width=True)
                else:
                    st.info("内訳（理由カテゴリ）表示用のデータがありません。")

            with c_pie2:
                pie_dept = build_pie(period_ddf, title="査定内訳（診療科）", group_col="診療科")
                if pie_dept is not None:
                    st.plotly_chart(pie_dept, use_container_width=True)
                else:
                    st.info("内訳（診療科）表示用のデータがありません。")

            breakdown_tables(period_ddf, s)

        with t3:
            if alert_tbl.empty:
                st.info("条件に合う注意項目がありません（除外条件やTopNを調整してね）。")
                return
            level_filter = st.multiselect("表示レベル", ["🔴危険","🟠要注意","🟡観察"],
                                          default=["🔴危険","🟠要注意","🟡観察"],
                                          key=f"lv_{segment_label}_{dept_mode}_{period_mode}")
            show = alert_tbl[alert_tbl["レベル"].isin(level_filter)].copy()
            show["査定率(%)"] = (show["査定率"]*100).round(2)
            show = show.drop(columns=["査定率"])
            st.dataframe(
                show[["レベル","査定理由カテゴリ","注意項目","査定額","件数","査定率(%)","増加額","z_rate","score"]]
                .rename(columns={"z_rate":"査定率Z","score":"スコア"}),
                use_container_width=True,
                hide_index=True
            )

    with tab_out:
        render_standard("外来")

    with tab_in:
        sub_dpc, sub_fee = st.tabs(["DPC","出来高"])
        with sub_dpc:
            render_standard("入院DPC")
        with sub_fee:
            render_standard("入院出来高")

if __name__ == "__main__":
    main()
