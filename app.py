import io
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "DPC査定分析（外来 / 入院DPC / 入院出来高）"

REQUIRED_COLS = ["月","区分","入院種別","診療科","査定理由カテゴリ","注意項目","査定額","件数","請求額"]

LOCAL_STORE_DIR = "data_store"
LOCAL_STORE_FILE = os.path.join(LOCAL_STORE_DIR, "latest.xlsx")

@dataclass
class Settings:
    sensitivity: str = "standard" # low|standard|high
    top_n_amount: int = 15
    top_n_increase: int = 15
    min_amount: float = 100000
    min_count: int = 3
    z_threshold: float = 2.0
    w_amount: int = 2
    w_increase: int = 1
    w_rate: int = 1

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

    # normalize strings
    df["区分"] = df["区分"].astype(str).str.strip()
    df["入院種別"] = df["入院種別"].fillna("").astype(str).str.strip()
    df["診療科"] = df["診療科"].astype(str).str.strip()
    df["査定理由カテゴリ"] = df["査定理由カテゴリ"].astype(str).str.strip()
    df["注意項目"] = df["注意項目"].astype(str).str.strip()

    # load settings (optional)
    s = Settings()
    if "settings" in xls.sheet_names:
        s_df = pd.read_excel(xls, "settings")
        if set(["key","value"]).issubset(s_df.columns):
            d = {str(k): str(v) for k,v in zip(s_df["key"], s_df["value"])}
            s.sensitivity = d.get("sensitivity", s.sensitivity)
            s.top_n_amount = int(float(d.get("top_n_amount", s.top_n_amount)))
            s.top_n_increase = int(float(d.get("top_n_increase", s.top_n_increase)))
            s.min_amount = float(d.get("min_amount", s.min_amount))
            s.min_count = int(float(d.get("min_count", s.min_count)))
            s.z_threshold = float(d.get("z_threshold", s.z_threshold))
            s.w_amount = int(float(d.get("w_amount", s.w_amount)))
            s.w_increase = int(float(d.get("w_increase", s.w_increase)))
            s.w_rate = int(float(d.get("w_rate", s.w_rate)))
    return df, s

def save_local_excel(file_bytes: bytes):
    os.makedirs(LOCAL_STORE_DIR, exist_ok=True)
    with open(LOCAL_STORE_FILE, "wb") as f:
        f.write(file_bytes)

def load_local_excel() -> bytes | None:
    if os.path.exists(LOCAL_STORE_FILE):
        with open(LOCAL_STORE_FILE, "rb") as f:
            return f.read()
    return None

def compute_monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["月","区分","入院種別","診療科"], as_index=False).agg(
        査定額=("査定額","sum"),
        件数=("件数","sum"),
        請求額=("請求額","max"),
    )
    g["査定率"] = np.where(g["請求額"]>0, g["査定額"]/g["請求額"], 0.0)
    return g

def top_other(series: pd.Series, topn=5) -> pd.Series:
    s = series.sort_values(ascending=False)
    if len(s) <= topn:
        return s
    top = s.iloc[:topn]
    other = pd.Series({"その他": s.iloc[topn:].sum()})
    return pd.concat([top, other])

def zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd

def score_alerts(df_detail: pd.DataFrame, monthly: pd.DataFrame, period_mode: str,
                 dept_mode: str, dept: str | None, s: Settings,
                 kubun: str, nyuin_type: str) -> pd.DataFrame:
    # filter scope
    df = df_detail[(df_detail["区分"]==kubun) & (df_detail["入院種別"]==nyuin_type)].copy()
    msc = monthly[(monthly["区分"]==kubun) & (monthly["入院種別"]==nyuin_type)].copy()
    if dept_mode == "診療科別" and dept:
        df = df[df["診療科"] == dept]
        msc = msc[msc["診療科"] == dept]

    all_months = msc["月"].sort_values().unique()
    if len(all_months)==0:
        return pd.DataFrame()
    latest = all_months[-1]

    if period_mode == "最新月":
        df_p = df[df["月"] == latest]
        prev = all_months[-2] if len(all_months) >= 2 else None
        df_prev = df[df["月"] == prev] if prev is not None else df.iloc[0:0]
    else:
        latest_dt = latest.to_timestamp()
        fy_start = pd.Period(f"{latest_dt.year-1 if latest_dt.month<4 else latest_dt.year}-04", freq="M")
        df_p = df[(df["月"] >= fy_start) & (df["月"] <= latest)]
        df_prev = df.iloc[0:0]

    cur = df_p.groupby(["査定理由カテゴリ","注意項目"], as_index=False).agg(
        査定額=("査定額","sum"), 件数=("件数","sum"), 請求額=("請求額","max")
    )
    cur["査定率"] = np.where(cur["請求額"]>0, cur["査定額"]/cur["請求額"], 0.0)

    cur = cur[(cur["査定額"] >= s.min_amount) & (cur["件数"] >= s.min_count)].copy()
    if cur.empty:
        return cur

    if period_mode == "最新月" and not df_prev.empty:
        prev_tbl = df_prev.groupby(["査定理由カテゴリ","注意項目"], as_index=False).agg(査定額=("査定額","sum"))
        cur = cur.merge(prev_tbl, on=["査定理由カテゴリ","注意項目"], how="left", suffixes=("","_前月")).fillna({"査定額_前月":0})
        cur["増加額"] = cur["査定額"] - cur["査定額_前月"]
    else:
        cur["増加額"] = 0.0

    cur["p_amount"] = 0
    r_amt = cur["査定額"].rank(method="min", ascending=False)
    cur.loc[r_amt <= s.top_n_amount, "p_amount"] = 2
    cur.loc[(r_amt <= s.top_n_amount*2) & (cur["p_amount"]==0), "p_amount"] = 1

    cur["p_increase"] = 0
    if period_mode == "最新月":
        r_inc = cur["増加額"].rank(method="min", ascending=False)
        cur.loc[r_inc <= s.top_n_increase, "p_increase"] = 2
        cur.loc[(r_inc <= s.top_n_increase*2) & (cur["p_increase"]==0), "p_increase"] = 1

    th = s.z_threshold
    if s.sensitivity == "high":
        th = max(1.2, th - 0.5)
    elif s.sensitivity == "low":
        th = th + 0.5
    cur["z_rate"] = zscore(cur["査定率"])
    cur["p_rate"] = (cur["z_rate"] >= th).astype(int) * 2

    cur["score"] = s.w_amount*cur["p_amount"] + s.w_increase*cur["p_increase"] + s.w_rate*cur["p_rate"]
    cur["レベル"] = np.select([cur["score"] >= 6, cur["score"] >= 3], ["🔴危険","🟠要注意"], default="🟡観察")
    cur = cur.sort_values(["score","査定額"], ascending=False).reset_index(drop=True)
    return cur

def pie_chart(ddf: pd.DataFrame):
    s = ddf.groupby("査定理由カテゴリ")["査定額"].sum()
    s = top_other(s, topn=5)
    st.plotly_chart({
        "data":[{"type":"pie","labels":s.index.tolist(),"values":s.values.tolist(),"textinfo":"percent+label"}],
        "layout":{"margin":{"l":0,"r":0,"t":10,"b":0}, "height":340}
    }, use_container_width=True)

def segment_key(segment: str):
    if segment == "外来":
        return "外来", ""
    if segment == "入院DPC":
        return "入院", "DPC"
    if segment == "入院出来高":
        return "入院", "出来高"
    raise ValueError("unknown segment")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    if "raw_bytes" not in st.session_state:
        st.session_state["raw_bytes"] = load_local_excel()
        st.session_state["df"] = None
        st.session_state["settings"] = Settings()

    with st.sidebar:
        st.subheader("データ")
        up = st.file_uploader("Excelをアップロード（dataシート）", type=["xlsx"])
        if up is not None:
            raw = up.read()
            st.session_state["raw_bytes"] = raw
            save_local_excel(raw)
            st.success("データを保存しました（ローカル自動反映）")

        if st.session_state["raw_bytes"] is not None and st.session_state["df"] is None:
            try:
                df, s0 = load_excel(st.session_state["raw_bytes"])
                st.session_state["df"] = df
                st.session_state["settings"] = s0
            except Exception as e:
                st.error(str(e))

        df = st.session_state.get("df")
        if df is None:
            st.info("左のアップロードからデータを入れてね（デモ用Excelも同梱）。")
            st.stop()

        st.divider()
        dept_mode = st.radio("粒度", ["全体","診療科別"], horizontal=True)
        dept = None
        if dept_mode == "診療科別":
            dept = st.selectbox("診療科", sorted(df["診療科"].unique()))
        period_mode = st.radio("期間", ["最新月","累計"], horizontal=True)

        with st.expander("⚙ 判定設定（折りたたみ）", expanded=False):
            s = st.session_state["settings"]
            s.sensitivity = st.select_slider("感度", options=["low","standard","high"], value=s.sensitivity,
                                             help="high=拾いやすい / low=絞り込み強め")
            s.top_n_amount = int(st.slider("金額インパクト 上位N", 5, 50, int(s.top_n_amount), step=5))
            s.top_n_increase = int(st.slider("増加インパクト 上位N（最新月のみ）", 5, 50, int(s.top_n_increase), step=5))
            c1,c2 = st.columns(2)
            with c1:
                s.min_amount = float(st.number_input("除外：査定額（円）未満", min_value=0, value=float(s.min_amount), step=50000.0))
            with c2:
                s.min_count = int(st.number_input("除外：件数 未満", min_value=0, value=int(s.min_count), step=1))
            s.z_threshold = float(st.slider("査定率 異常度(Z)しきい値", 1.0, 3.5, float(s.z_threshold), step=0.1))
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

    # Top-level tabs: 外来 / 入院
    tab_out, tab_in = st.tabs(["外来","入院"])

    def render_segment(segment_label: str):
        kubun, nyuin_type = segment_key(segment_label)
        # scope monthly
        msc = monthly[(monthly["区分"]==kubun) & (monthly["入院種別"]==nyuin_type)].copy()
        if msc.empty:
            st.info("この区分のデータがありません。")
            return

        if dept_mode == "診療科別" and dept:
            msc2 = msc[msc["診療科"]==dept].copy()
        else:
            msc2 = msc.groupby("月", as_index=False).agg(査定額=("査定額","sum"),件数=("件数","sum"),請求額=("請求額","sum"))
            msc2["査定率"] = np.where(msc2["請求額"]>0, msc2["査定額"]/msc2["請求額"], 0.0)

        latest = msc2["月"].sort_values().unique()[-1]
        if period_mode == "最新月":
            cur_months = [latest]
            period_label = f"最新月：{latest}"
        else:
            latest_dt = latest.to_timestamp()
            fy_start = pd.Period(f"{latest_dt.year-1 if latest_dt.month<4 else latest_dt.year}-04", freq="M")
            cur_months = [m for m in msc2["月"].unique() if (m>=fy_start and m<=latest)]
            period_label = f"累計：{fy_start}〜{latest}"

        # summary cards
        if period_mode == "最新月":
            cur = msc2[msc2["月"]==latest]
        else:
            cur = msc2[msc2["月"].isin(cur_months)]
        tot_satei = float(cur["査定額"].sum())
        tot_claim = float(cur["請求額"].sum())
        tot_rate = (tot_satei/tot_claim) if tot_claim>0 else 0.0

        alert_tbl = score_alerts(df, monthly, period_mode, dept_mode, dept, s, kubun, nyuin_type)

        st.subheader(f"{segment_label} / {period_label} / {dept_mode}{'' if dept is None else '：'+dept}")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("査定額", f"{tot_satei:,.0f} 円")
        c2.metric("請求額", f"{tot_claim:,.0f} 円")
        c3.metric("査定率", f"{tot_rate*100:.2f} %")
        c4.metric("アラート(🔴/🟠/🟡)", 
                  f"{(alert_tbl['レベル']=='🔴危険').sum() if not alert_tbl.empty else 0}/"
                  f"{(alert_tbl['レベル']=='🟠要注意').sum() if not alert_tbl.empty else 0}/"
                  f"{(alert_tbl['レベル']=='🟡観察').sum() if not alert_tbl.empty else 0}")

        t1,t2,t3 = st.tabs(["① 推移（混合）","② 内訳（円）","③ 注意項目（アラート）"])

        with t1:
            st.markdown("棒：査定額 / 折れ線：査定率(%)")
            chart_df = msc2.sort_values("月").copy()
            st.plotly_chart({
                "data":[
                    {"type":"bar","x":chart_df["月"].astype(str).tolist(),"y":chart_df["査定額"].tolist(),"name":"査定額"},
                    {"type":"scatter","mode":"lines+markers","x":chart_df["月"].astype(str).tolist(),"y":(chart_df["査定率"]*100).tolist(),
                     "name":"査定率(%)","yaxis":"y2"},
                ],
                "layout":{
                    "height":420,
                    "margin":{"l":40,"r":40,"t":20,"b":40},
                    "yaxis":{"title":"査定額(円)"},
                    "yaxis2":{"title":"査定率(%)","overlaying":"y","side":"right"},
                    "legend":{"orientation":"h"},
                }
            }, use_container_width=True)

        with t2:
            ddf = df[(df["区分"]==kubun) & (df["入院種別"]==nyuin_type)].copy()
            if dept_mode == "診療科別" and dept:
                ddf = ddf[ddf["診療科"]==dept]
            if period_mode == "最新月":
                ddf = ddf[ddf["月"]==latest]
            else:
                latest_dt = latest.to_timestamp()
                fy_start = pd.Period(f"{latest_dt.year-1 if latest_dt.month<4 else latest_dt.year}-04", freq="M")
                ddf = ddf[(ddf["月"]>=fy_start) & (ddf["月"]<=latest)]
            pie_chart(ddf)
            tbl = ddf.groupby("査定理由カテゴリ", as_index=False).agg(査定額=("査定額","sum"),件数=("件数","sum"))
            tot = tbl["査定額"].sum()
            tbl["割合"] = np.where(tot>0, tbl["査定額"]/tot, 0.0)
            tbl = tbl.sort_values("査定額", ascending=False)
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        with t3:
            if alert_tbl.empty:
                st.info("条件に合う注意項目がありません（除外条件やTopNを調整してね）。")
                return
            level_filter = st.multiselect("表示レベル", ["🔴危険","🟠要注意","🟡観察"], default=["🔴危険","🟠要注意","🟡観察"], key=f"lv_{segment_label}")
            show = alert_tbl[alert_tbl["レベル"].isin(level_filter)].copy()
            st.dataframe(
                show[["レベル","査定理由カテゴリ","注意項目","査定額","件数","査定率","増加額","z_rate","score"]]
                .rename(columns={"z_rate":"査定率Z","score":"スコア"}),
                use_container_width=True,
                hide_index=True
            )
            sel = st.selectbox("ドリルダウン（注意項目）", show["注意項目"].tolist(), key=f"sel_{segment_label}")
            dd = df[(df["区分"]==kubun) & (df["入院種別"]==nyuin_type) & (df["注意項目"]==sel)].copy()
            if dept_mode == "診療科別" and dept:
                dd = dd[dd["診療科"]==dept]
            if period_mode == "最新月":
                dd = dd[dd["月"]==latest]
            else:
                latest_dt = latest.to_timestamp()
                fy_start = pd.Period(f"{latest_dt.year-1 if latest_dt.month<4 else latest_dt.year}-04", freq="M")
                dd = dd[(dd["月"]>=fy_start) & (dd["月"]<=latest)]
            st.markdown("**選択項目の内訳（診療科/理由）**")
            c1,c2 = st.columns(2)
            with c1:
                st.write("診療科別")
                t1 = dd.groupby("診療科", as_index=False).agg(査定額=("査定額","sum"),件数=("件数","sum")).sort_values("査定額", ascending=False)
                st.dataframe(t1, use_container_width=True, hide_index=True)
            with c2:
                st.write("理由カテゴリ別")
                t2 = dd.groupby("査定理由カテゴリ", as_index=False).agg(査定額=("査定額","sum"),件数=("件数","sum")).sort_values("査定額", ascending=False)
                st.dataframe(t2, use_container_width=True, hide_index=True)

    with tab_out:
        render_segment("外来")

    with tab_in:
        sub1, sub2 = st.tabs(["DPC","出来高"])
        with sub1:
            render_segment("入院DPC")
        with sub2:
            render_segment("入院出来高")

if __name__ == "__main__":
    main()