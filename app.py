import os
import sys
import io
import urllib.parse
import warnings
warnings.filterwarnings("ignore")

for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(proxy_var, None)
os.environ["NO_PROXY"] = "*"

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import yfinance as yf

if get_script_run_ctx() is None:
    os.system(f'"{sys.executable}" -m streamlit run "{__file__}"')
    sys.exit()

# 创建带有浏览器级 User-Agent 和自动重试机制的稳健 Session
DIRECT_SESSION = requests.Session()
DIRECT_SESSION.trust_env = False
DIRECT_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
})
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
DIRECT_SESSION.mount('https://', HTTPAdapter(max_retries=retries))
DIRECT_SESSION.mount('http://', HTTPAdapter(max_retries=retries))

FAST_CIK_MAP = {
    "AAPL": "0000320193", "MSFT": "0000789019", "GOOGL": "0001652044", "GOOG": "0001652044",
    "AMZN": "0001018724", "META": "0001326801", "TSLA": "0001318605", "NVDA": "0001045810",
    "KO": "0000021344", "PG": "0000080424", "PEP": "0000077476", "WMT": "0000104169",
    "COST": "0000909832", "JNJ": "0000200406", "MCD": "0000063908", "PM": "0001413329",
    "TSM": "0001046179", "AVGO": "0001730168", "AMD": "0000002488", "MU": "0000723125",
    "ASML": "0000937966", "BRK-B": "0001067983", "JPM": "0000019617", "V": "0001403161",
    "MA": "0001141391", "BAC": "0000070858", "GS": "0000886982", "BLK": "0001364742", "C": "0000831001"
}

# ================= 1. 页面基本配置 =================
st.set_page_config(page_title="客观估值与财报追踪终端", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    div[data-testid="stMetricValue"] {font-size: 1.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================= 2. 细分板块字典 =================
SECTOR_DICT = {
    "🚀 科技巨头 (Mag 7 - 纳指核心池)": {
        "【板块整体合成】(大盘加权算法验证)": "__SECTOR_AGG__",
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Alphabet (GOOGL)": "GOOGL",
        "Amazon (AMZN)": "AMZN",
        "Meta (META)": "META",
        "Tesla (TSLA)": "TSLA",
        "NVIDIA (NVDA)": "NVDA",
    },
    "🛒 必需消费品与快消巨头 (FMCG)": {
        "【板块整体合成】(大盘加权算法验证)": "__SECTOR_AGG__",
        "可口可乐 (KO)": "KO",
        "宝洁 (PG)": "PG",
        "百事可乐 (PEP)": "PEP",
        "沃尔玛 (WMT)": "WMT",
        "好市多 (COST)": "COST",
    },
    "💾 半导体与AI算力": {
        "【板块整体合成】(大盘加权算法验证)": "__SECTOR_AGG__",
        "NVIDIA (NVDA)": "NVDA",
        "TSMC 台积电 (TSM)": "TSM",
        "Broadcom 博通 (AVGO)": "AVGO",
        "AMD (AMD)": "AMD",
        "ASML 阿斯麦 (ASML)": "ASML",
    },
    "🌐 核心大盘 ETF (注:无底层财报,请用上方合成)": {
        "标普500 ETF (SPY)": "SPY",
        "纳斯达克100 ETF (QQQ)": "QQQ",
        "罗素2000 ETF (IWM)": "IWM",
    }
}

# ================= 3. 侧边栏控制区 =================
with st.sidebar:
    st.markdown("### ⚙️ 终端控制面板")

    selected_sector = st.selectbox("1. 选择板块", list(SECTOR_DICT.keys()))
    sector_map = SECTOR_DICT[selected_sector]
    default_idx = 1 if len(sector_map) > 1 else 0
    selected_label = st.selectbox("2. 选择板块内标的", list(sector_map.keys()), index=default_idx)

    custom_code = st.text_input("或直接输入股票代码", value="").strip().upper()
    target_symbol = custom_code if custom_code else sector_map[selected_label]

    st.markdown("---")
    
    val_metric = st.radio(
        "3. 估值倍数口径",
        ["P/E (市盈率)", "P/S (市销率)", "P/B (市净率)"],
        index=0,
    )

    weight_method = st.selectbox(
        "3.1 指数/板块加权方式 (仅对合成有效)",
        ["市值加权平均 (Cap-Weighted)", "等权平均 (Equal-Weighted)"],
        index=0
    )

    calc_mode = st.selectbox(
        "3.2 财报折算算法",
        ["原版初始算法 (保留10-K累计)", "TTM 滚动四季 (主流券商标准)"],
        index=1,
    )

    k_interval = st.selectbox(
        "4. K 线周期级别",
        ["日K (1d)", "周K (1wk)", "月K (1mo)", "季K (3mo)", "年K (1y)"],
        index=0,
    )

    time_span = st.selectbox(
        "5. 历史时间跨度",
        ["1 year (1y)", "2 years (2y)", "5 years (5y)", "10 years (10y)", "20 years (20y)", "全部历史 (max)"],
        index=3,
    )

    macro_choice = st.selectbox(
        "6. 附加宏观图表",
        [
            "10年期美债收益率 (^TNX)", 
            "布伦特原油 (Brent, BZ=F)",
            "美国 CPI (消费者物价指数 YoY%)",
            "美国核心 PCE (美联储最爱 YoY%)",
            "美国 PPI (生产者物价指数 YoY%)"
        ],
        index=0
    )

    st.markdown("---")
    col_a, _ = st.columns(2)
    color_style = col_a.radio("K线配色", ["红涨", "绿涨"], index=0)

    if st.button("🔄 刷新最新数据", width="stretch"):
        st.cache_data.clear()

# ================= 4. 基础工具函数 =================
def parse_period_and_interval(k_str, span_str):
    itv = k_str.split("(")[1].replace(")", "")
    prd = span_str.split("(")[1].replace(")", "").strip()
    return itv, prd

def format_money_str(x):
    if pd.isna(x): return "N/A"
    return f"{x / 1e7:.2f} 千万"

def safe_resample(df, itv_raw):
    mapping_new = {"1y": "YE", "3mo": "QE", "1mo": "ME", "1wk": "W"}
    mapping_old = {"1y": "Y", "3mo": "Q", "1mo": "M", "1wk": "W"}
    if itv_raw not in mapping_new: return df.copy()
    try:
        return df.resample(mapping_new[itv_raw]).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    except ValueError:
        return df.resample(mapping_old[itv_raw]).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()

def extract_gaap_data(units_list, col_name, mode_str, is_pb=False):
    if not units_list: return pd.DataFrame()
    if "原版" in mode_str:
        rows = [{"Period_End": u.get("end"), "Filed_Date": u.get("filed"), col_name: float(u.get("val"))} 
                for u in units_list if u.get("form") in ["10-Q", "10-K"] and u.get("val") is not None and u.get("end") and u.get("filed")]
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["Period_End"] = pd.to_datetime(df["Period_End"])
        df["Filed_Date"] = pd.to_datetime(df["Filed_Date"])
        df = df.sort_values(["Filed_Date", "Period_End"]).drop_duplicates(subset=["Filed_Date"], keep="last")
        df.set_index("Filed_Date", inplace=True)
        return df[[col_name, "Period_End"]]
    
    q_rows, fy_rows = [], []
    for u in units_list:
        form = u.get("form")
        if form not in ["10-Q", "10-K"] or u.get("val") is None or not u.get("end") or not u.get("filed"): continue
        if is_pb:
            q_rows.append({"Period_End": u.get("end"), "Filed_Date": u.get("filed"), col_name: float(u.get("val"))})
        else:
            start = u.get("start")
            if not start: continue
            try:
                delta = (pd.to_datetime(u.get("end")) - pd.to_datetime(start)).days
                if 70 <= delta <= 115:
                    q_rows.append({"Period_End": u.get("end"), "Filed_Date": u.get("filed"), col_name: float(u.get("val"))})
                elif 340 <= delta <= 385 and form == "10-K":
                    fy_rows.append({"Period_End": u.get("end"), "Filed_Date": u.get("filed"), col_name: float(u.get("val"))})
            except: continue
            
    if not q_rows and not fy_rows: return pd.DataFrame()

    df_q = pd.DataFrame(q_rows) if q_rows else pd.DataFrame(columns=["Period_End", "Filed_Date", col_name])
    if not df_q.empty:
        df_q["Period_End"] = pd.to_datetime(df_q["Period_End"])
        df_q["Filed_Date"] = pd.to_datetime(df_q["Filed_Date"])
        df_q = df_q.sort_values(["Period_End", "Filed_Date"]).drop_duplicates(subset=["Period_End"], keep="first")

    if fy_rows and not is_pb:
        df_fy = pd.DataFrame(fy_rows)
        df_fy["Period_End"] = pd.to_datetime(df_fy["Period_End"])
        df_fy["Filed_Date"] = pd.to_datetime(df_fy["Filed_Date"])
        df_fy = df_fy.sort_values(["Period_End", "Filed_Date"]).drop_duplicates(subset=["Period_End"], keep="first")
        derived_q4 = []
        for _, fy_row in df_fy.iterrows():
            fy_end = fy_row["Period_End"]
            if df_q.empty or not ((df_q["Period_End"] - fy_end).abs() <= pd.Timedelta(days=15)).any():
                prev_3q = df_q[(df_q["Period_End"] < fy_end - pd.Timedelta(days=15)) & (df_q["Period_End"] >= fy_end - pd.Timedelta(days=315))] if not df_q.empty else pd.DataFrame()
                q4_val = fy_row[col_name] - prev_3q[col_name].sum() if len(prev_3q) == 3 else fy_row[col_name] / 4.0
                derived_q4.append({"Period_End": fy_end, "Filed_Date": fy_row["Filed_Date"], col_name: float(q4_val)})
        if derived_q4: df_q = pd.concat([df_q, pd.DataFrame(derived_q4)], ignore_index=True)

    df_q = df_q.sort_values(["Period_End", "Filed_Date"]).drop_duplicates(subset=["Period_End"], keep="first")
    if "TTM" in mode_str and not is_pb: df_q[col_name] = df_q[col_name].rolling(4, min_periods=1).mean()
    df_q = df_q.sort_values(["Filed_Date", "Period_End"]).drop_duplicates(subset=["Filed_Date"], keep="last")
    df_q.set_index("Filed_Date", inplace=True)
    return df_q[[col_name, "Period_End"]]

# ================= 5. 数据获取引擎 =================
@st.cache_data(ttl=86400)
def get_cik_for_ticker(ticker: str):
    t_up = ticker.upper()
    if t_up in FAST_CIK_MAP:
        return FAST_CIK_MAP[t_up]
    headers = {"User-Agent": "ResearchTerminal/2.0 (admin@finance-research.com)"}
    try:
        r = DIRECT_SESSION.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=8)
        if r.status_code == 200:
            for item in r.json().values():
                if item["ticker"].upper() == t_up:
                    return str(item["cik_str"]).zfill(10)
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def get_sec_quarterly_financials(ticker: str, metric_choice: str, mode_str: str):
    headers = {"User-Agent": "ResearchTerminal/2.0 (admin@finance-research.com)"}
    is_ps = "P/S" in metric_choice
    is_pb = "P/B" in metric_choice
    
    try:
        cik = get_cik_for_ticker(ticker)
        if not cik: return pd.DataFrame()

        facts_resp = DIRECT_SESSION.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=headers, timeout=12)
        if facts_resp.status_code != 200: return pd.DataFrame()
        us_gaap = facts_resp.json().get("facts", {}).get("us-gaap", {})
        
        target_units = []
        if is_pb: tags = ["StockholdersEquity", "AssetsMinusLiabilities", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
        elif is_ps: tags = ["Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet", "RevenueFromContractWithCustomerExcludingAssessedTax"]
        else: tags = ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"]
            
        for tag in tags:
            if tag in us_gaap: target_units.extend(us_gaap[tag].get("units", {}).get("USD", []))
        return extract_gaap_data(target_units, "GAAP_Val", mode_str, is_pb=is_pb)
    except:
        return pd.DataFrame()


def parse_csv_by_position(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text), na_values=[".", "", "NA", "null"])
    if df.shape[1] < 2:
        return pd.DataFrame()
    date_col, val_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col, val_col]).set_index(date_col).sort_index()
    return df[[val_col]].rename(columns={val_col: "Value"})


@st.cache_data(ttl=3600)
def fetch_macro_data(indicator: str, prd_code: str):
    yf_prd = "max" if prd_code == "20y" else prd_code
    cutoff_date = pd.Timestamp.today().normalize()
    
    if "美债" in indicator or "原油" in indicator:
        try:
            tk = yf.Ticker("^TNX" if "美债" in indicator else "BZ=F", session=DIRECT_SESSION)
            df = tk.history(period=yf_prd)
            if not df.empty:
                df = df[["Close"]].rename(columns={"Close": "Value"})
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                if prd_code == "20y": df = df[df.index >= cutoff_date - pd.DateOffset(years=20)]
            return df, "Yahoo Finance API", ""
        except Exception as e:
            return pd.DataFrame(), "", str(e)
        
    fred_id_map = {
        "美国 CPI (消费者物价指数 YoY%)": "CPIAUCSL",
        "美国核心 PCE (美联储最爱 YoY%)": "PCEPILFE",
        "美国 PPI (生产者物价指数 YoY%)": "WPSFD49207"
    }
    fred_id = fred_id_map.get(indicator)
    if not fred_id:
        return pd.DataFrame(), "", "未知指标映射"
    
    raw_df = pd.DataFrame()
    source_used = ""
    error_logs = []
    
    fred_csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    try:
        resp = DIRECT_SESSION.get(fred_csv_url, headers=headers, timeout=10)
        if resp.status_code == 200 and "<html" not in resp.text[:200].lower():
            raw_df = parse_csv_by_position(resp.text)
            if not raw_df.empty:
                source_used = "FRED Official CSV (Direct)"
        else:
            error_logs.append(f"FRED状态码:{resp.status_code}")
    except Exception as e:
        error_logs.append(f"FRED异常:{type(e).__name__}")

    if raw_df.empty:
        try:
            db_url = f"https://api.db.nomics.world/v22/series/FRED/{fred_id}?observations=1"
            resp = DIRECT_SESSION.get(db_url, timeout=8)
            if resp.status_code == 200:
                docs = resp.json().get("series", {}).get("docs", [])
                if docs:
                    period = docs[0].get("period", [])
                    value = docs[0].get("value", [])
                    if period and value:
                        temp_df = pd.DataFrame({"Date": period, "Value": value})
                        temp_df["Date"] = pd.to_datetime(temp_df["Date"], errors="coerce")
                        temp_df["Value"] = pd.to_numeric(temp_df["Value"], errors="coerce")
                        raw_df = temp_df.dropna().set_index("Date").sort_index()
                        source_used = "DB.nomics Open API"
            else:
                error_logs.append(f"DB.nomics状态码:{resp.status_code}")
        except Exception as e:
            error_logs.append(f"DB.nomics异常:{type(e).__name__}")

    if raw_df.empty:
        eco_map = {"CPIAUCSL": "USCPI", "WPSFD49207": "USPPI"}
        if fred_id in eco_map:
            try:
                eco_url = f"https://www.econdb.com/api/series/{eco_map[fred_id]}/?format=json"
                resp = DIRECT_SESSION.get(eco_url, headers=headers, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    dates = data.get("data", {}).get("dates", [])
                    vals = data.get("data", {}).get("values", [])
                    if dates and vals:
                        raw_df = pd.DataFrame({"Value": vals}, index=pd.to_datetime(dates)).sort_index()
                        source_used = "Econdb Quant API"
            except Exception as e:
                error_logs.append(f"Econdb异常:{type(e).__name__}")

    if not raw_df.empty:
        try:
            raw_df["Value"] = raw_df["Value"].pct_change(12) * 100 
            raw_df = raw_df.dropna()
            
            if raw_df.index.tz is not None:
                raw_df.index = raw_df.index.tz_localize(None)
            
            if prd_code != "max":
                if "y" in prd_code: cutoff = cutoff_date - pd.DateOffset(years=int(prd_code.replace("y", "")))
                elif "mo" in prd_code: cutoff = cutoff_date - pd.DateOffset(months=int(prd_code.replace("mo", "")))
                else: cutoff = cutoff_date - pd.DateOffset(years=1)
                raw_df = raw_df[raw_df.index >= cutoff]
            
            return raw_df, source_used, ""
        except Exception as e:
            error_logs.append(f"YoY计算异常:{str(e)}")
        
    return pd.DataFrame(), "", " | ".join(error_logs)


@st.cache_data(ttl=300)
def fetch_single_security(symbol, itv_raw, prd, metric_choice, mode_str):
    yf_prd = "max" if prd == "20y" else prd
    
    tk = yf.Ticker(symbol, session=DIRECT_SESSION)
    raw_daily = tk.history(period=yf_prd, interval="1d", auto_adjust=True)
        
    if raw_daily.empty: return pd.DataFrame(), {}
    
    if isinstance(raw_daily.columns, pd.MultiIndex): raw_daily.columns = raw_daily.columns.get_level_values(0)
    raw_daily.index = raw_daily.index.tz_localize(None)
    if prd == "20y": raw_daily = raw_daily[raw_daily.index >= pd.Timestamp.now() - pd.DateOffset(years=20)]

    df = safe_resample(raw_daily, itv_raw)
    latest_close = float(df["Close"].iloc[-1])
    
    shares, mcap_now = None, None
    try:
        f_info = tk.fast_info
        shares = f_info.get("shares")
        mcap_now = f_info.get("market_cap")
    except:
        pass
        
    if not shares and not mcap_now:
        try:
            info = tk.info or {}
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            mcap_now = info.get("marketCap")
        except:
            pass
            
    if not shares and mcap_now and latest_close > 0:
        shares = mcap_now / latest_close

    df["MarketCap"] = df["Close"] * shares if shares else np.nan
    df["GAAP_Val"] = np.nan
    df["Val_Metric_Annualized"] = np.nan

    is_pb = "P/B" in metric_choice
    sec_fin_df = get_sec_quarterly_financials(symbol, metric_choice, mode_str)
    
    try:
        if not sec_fin_df.empty and shares:
            sec_fin_df = sec_fin_df.dropna(subset=["GAAP_Val"]).sort_index()
            sec_fin_df["Period_End_Str"] = sec_fin_df["Period_End"].dt.strftime('%Y-%m-%d')
            sec_fin_df["Filed_Str"] = sec_fin_df.index.strftime('%Y-%m-%d')
            
            merged = pd.merge_asof(
                df[["MarketCap"]], sec_fin_df[["GAAP_Val", "Filed_Str", "Period_End_Str"]],
                left_index=True, right_index=True, direction="backward"
            )
            df["GAAP_Val"] = merged["GAAP_Val"]
            df["Filed_Str"] = merged["Filed_Str"]
            df["Period_End_Str"] = merged["Period_End_Str"]
            
            annualized_factor = 1 if is_pb else 4
            annualized_val = df["GAAP_Val"] * annualized_factor
            df["Val_Metric_Annualized"] = np.where(annualized_val > 0, df["MarketCap"] / annualized_val, np.nan)
    except:
        pass

    if df["Val_Metric_Annualized"].isna().all():
        df["Hover_Val"] = "缺少 SEC 底层数据 (可能为ETF)"
    else:
        df["Val_Metric_Annualized"] = df["Val_Metric_Annualized"].ffill()
    
    metric_display = metric_choice.split(" ")[0]
    fundamental_display = "GAAP 权益" if is_pb else ("GAAP 营收" if "P/S" in metric_choice else "GAAP 净利")

    if "Period_End_Str" in df.columns:
        df["Hover_Val"] = ("<b>财报锚定:</b> " + df["Period_End_Str"].fillna("N/A").astype(str) + 
                           f"<br><b>当季基准 {fundamental_display}:</b> $" + df["GAAP_Val"].apply(format_money_str))

    return df, {
        "symbol": symbol,
        "price": latest_close,
        "pct_change": ((latest_close - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2]) * 100 if len(df) > 1 else 0.0),
        "mcap_b": (df["MarketCap"].iloc[-1] / 1e9) if not np.isnan(df["MarketCap"].iloc[-1]) else None,
        "val_multiple": float(df["Val_Metric_Annualized"].dropna().iloc[-1]) if not df["Val_Metric_Annualized"].dropna().empty else None,
        "metric_name": metric_display,
        "fundamental_name": fundamental_display
    }

def fetch_sector_aggregate(stock_dict, itv_raw, prd, metric_choice, mode_str, weight_mode):
    mcap_df = pd.DataFrame()
    fund_df = pd.DataFrame()
    val_df = pd.DataFrame()
    open_frames, high_frames, low_frames, close_frames = [], [], [], []
    table_rows = []
    is_pb = "P/B" in metric_choice

    valid_items = [(label, sym) for label, sym in stock_dict.items() if sym != "__SECTOR_AGG__"]
    prog_bar = st.progress(0, text="正在合成板块底层财报与行情数据...")

    for idx, (label, sym) in enumerate(valid_items):
        prog_bar.progress((idx + 1) / len(valid_items), text=f"正在拉取并合成成分股 ({idx+1}/{len(valid_items)}): {label}...")
        s_df, summ = fetch_single_security(sym, itv_raw, prd, metric_choice, mode_str)
        if s_df.empty: continue
        table_rows.append({"名称": label, **summ})
        
        mcap_df[sym] = s_df["MarketCap"]
        fund_df[sym] = s_df["GAAP_Val"] * (1 if is_pb else 4)
        val_df[sym] = s_df["Val_Metric_Annualized"]
        
        w_factor = (summ.get("mcap_b") or 1.0) / s_df["Close"].iloc[-1]
        open_frames.append(s_df["Open"] * w_factor)
        high_frames.append(s_df["High"] * w_factor)
        low_frames.append(s_df["Low"] * w_factor)
        close_frames.append(s_df["Close"] * w_factor)

    prog_bar.empty()

    if mcap_df.empty or not close_frames:
        return pd.DataFrame(), {}, pd.DataFrame()

    mcap_df = mcap_df.ffill()
    fund_df = fund_df.ffill()
    val_df = val_df.ffill()
    
    agg_open = pd.concat(open_frames, axis=1).ffill().bfill().sum(axis=1)
    agg_high = pd.concat(high_frames, axis=1).ffill().bfill().sum(axis=1)
    agg_low = pd.concat(low_frames, axis=1).ffill().bfill().sum(axis=1)
    agg_close = pd.concat(close_frames, axis=1).ffill().bfill().sum(axis=1)

    agg_df = pd.DataFrame({
        "Open": agg_open,
        "High": np.maximum.reduce([agg_high, agg_open, agg_close]),
        "Low": np.minimum.reduce([agg_low, agg_open, agg_close]),
        "Close": agg_close,
    })
    
    if "等权" in weight_mode:
        agg_df["Val_Metric_Annualized"] = val_df.mean(axis=1)
        agg_df["GAAP_Val"] = fund_df.mean(axis=1) / (1 if is_pb else 4)
    else:
        agg_df["Val_Metric_Annualized"] = mcap_df.sum(axis=1, min_count=1) / fund_df.sum(axis=1, min_count=1)
        agg_df["GAAP_Val"] = fund_df.sum(axis=1, min_count=1) / (1 if is_pb else 4)

    agg_df["Hover_Val"] = f"<b>合成大盘 ({weight_mode.split(' ')[0]})</b>"
    latest_c = float(agg_df["Close"].iloc[-1])
    return agg_df, {
        "symbol": "合成大盘",
        "price": latest_c,
        "pct_change": (latest_c - float(agg_df["Close"].iloc[-2])) / float(agg_df["Close"].iloc[-2]) * 100 if len(agg_df) > 1 else 0.0,
        "mcap_b": mcap_df.sum(axis=1).iloc[-1] / 1e9,
        "val_multiple": float(agg_df["Val_Metric_Annualized"].iloc[-1]) if not pd.isna(agg_df["Val_Metric_Annualized"].iloc[-1]) else None,
        "metric_name": metric_choice.split(" ")[0],
        "fundamental_name": "加权权益" if is_pb else ("加权营收" if "P/S" in metric_choice else "加权净利")
    }, pd.DataFrame(table_rows)

# ================= 6. 执行数据加载 =================
itv_code, prd_code = parse_period_and_interval(k_interval, time_span)

if target_symbol == "__SECTOR_AGG__":
    df, summary, sector_table_df = fetch_sector_aggregate(sector_map, itv_code, prd_code, val_metric, calc_mode, weight_method)
    display_title = f"{selected_sector} - 真实大盘估值 ({weight_method.split(' ')[0]})"
else:
    with st.spinner(f"正在加载 {target_symbol} 财报与行情数据..."):
        df, summary = fetch_single_security(target_symbol, itv_code, prd_code, val_metric, calc_mode)
    display_title = f"{target_symbol} ({k_interval})"

if df.empty or "Close" not in df.columns:
    st.error("当前标的暂无数据，请尝试点击左侧『🔄 刷新最新数据』。")
    st.stop()

df["MA5"] = df["Close"].rolling(5).mean()
df["MA20"] = df["Close"].rolling(20).mean()
df["MA60"] = df["Close"].rolling(60).mean()

m_name = summary.get("metric_name", "P/E")
f_name = summary.get("fundamental_name", "净利润")

# ================= 7. 顶部指标栏 =================
k1, k2, k3 = st.columns(3)
k1.metric("最新价格/合成点数", f"{summary['price']:.2f}", f"{summary['pct_change']:+.2f}%")
k2.metric("覆盖总市值 (Market Cap)", f"${summary['mcap_b']:.2f} B" if summary.get("mcap_b") else "N/A")
k3.metric(f"当前 {m_name} ({calc_mode.split(' ')[0]})", f"{summary['val_multiple']:.2f}x" if summary.get("val_multiple") else "N/A")

# ================= 8. 独立联动图表 =================
up_c, dn_c = ("#ff3b30", "#00c805") if color_style == "红涨" else ("#00c805", "#ff3b30")

fig_kline = go.Figure()
fig_kline.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K线", increasing_line_color=up_c, decreasing_line_color=dn_c))
for ma, col_hex in [("MA5", "#f5a623"), ("MA20", "#00e5ff"), ("MA60", "#e040fb")]:
    fig_kline.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=col_hex, width=1.2)))
fig_kline.update_layout(title=f"📊 {display_title} K线系统", height=340, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=20, r=20, t=45, b=25), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig_kline.update_xaxes(matches='x')

fig_val = go.Figure()
if "Val_Metric_Annualized" in df:
    valid_val = df["Val_Metric_Annualized"].dropna()
    if not valid_val.empty:
        v_mean = valid_val.mean()
        v_std = valid_val.std()
        
        fig_val.add_trace(go.Scatter(x=valid_val.index, y=[v_mean+v_std]*len(valid_val), name="+1 标准差 (高估区)", line=dict(color="#ff3b30", width=1, dash="dot")))
        fig_val.add_trace(go.Scatter(x=valid_val.index, y=[v_mean]*len(valid_val), name="历史平均值", line=dict(color="#f5a623", width=1.5, dash="dash")))
        fig_val.add_trace(go.Scatter(x=valid_val.index, y=[v_mean-v_std]*len(valid_val), name="-1 标准差 (低估区)", line=dict(color="#00c805", width=1, dash="dot")))
        
        fig_val.add_trace(go.Scatter(
            x=valid_val.index, y=valid_val, name=f"动态 {m_name}",
            text=df.loc[valid_val.index, "Hover_Val"], hovertemplate="%{text}<extra></extra>",
            line=dict(color="#00ffcc", width=2.5)
        ))
    else:
        fig_val.add_annotation(text="⚠️ 暂无底层财务数据，无法绘制估值折线<br>如果是单一指数(如SPY)请使用『【板块整体合成】』",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#f5a623"))
else:
    fig_val.add_annotation(text="⚠️ 暂无估值指标计算结果", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
fig_val.update_layout(title=f"📈 估值追踪带: {m_name} [{calc_mode}] 与 统计标准差", height=320, template="plotly_dark", hovermode="x", margin=dict(l=20, r=20, t=45, b=25), xaxis=dict(matches="x"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

fig_profit = go.Figure()
if "GAAP_Val" in df.columns:
    valid_fund = df["GAAP_Val"].dropna()
    if not valid_fund.empty:
        fig_profit.add_trace(go.Scatter(x=valid_fund.index, y=valid_fund / 1e7, name=f"季报实际 {f_name}", line=dict(color="#e040fb", width=2.5)))
        fig_profit.update_yaxes(rangemode="tozero") 
    else:
        fig_profit.add_annotation(text="⚠️ 暂无底层财报数据",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#f5a623"))
else:
    fig_profit.add_annotation(text="⚠️ 基本面数据缺失", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

fig_profit.update_layout(title=f"💰 基本面追踪: 季报 {f_name} (千万美元)", height=250, template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=45, b=25), xaxis=dict(matches="x"))

st.plotly_chart(fig_kline, use_container_width=True)
st.plotly_chart(fig_val, use_container_width=True)
st.plotly_chart(fig_profit, use_container_width=True)

# ================= 9. 宏观与明细表 =================
st.markdown("---")
st.markdown(f"#### 🌍 宏观经济追踪: {macro_choice}")

with st.spinner(f"正在获取 {macro_choice} 数据..."):
    macro_df, node_used, err_msg = fetch_macro_data(macro_choice, prd_code)

if not macro_df.empty:
    line_color = "#f5a623" if "美债" in macro_choice else ("#00e5ff" if "CPI" in macro_choice or "PCE" in macro_choice or "PPI" in macro_choice else "#ff3b30")
    fig_macro = go.Figure(go.Scatter(x=macro_df.index, y=macro_df["Value"], name=macro_choice.split(" ")[0], line=dict(color=line_color, width=2.5)))
    
    y_title = "同比增速 (YoY %)" if "YoY" in macro_choice else ("收益率 (%)" if "美债" in macro_choice else "价格 (USD)")
    
    fig_macro.update_layout(
        height=280, 
        template="plotly_dark", 
        hovermode="x unified", 
        margin=dict(l=20, r=20, t=45, b=25), 
        xaxis=dict(matches="x"),
        yaxis_title=y_title
    )
    st.plotly_chart(fig_macro, use_container_width=True)
    if node_used: st.caption(f"✅ 获取成功。数据源: `{node_used}`")
else:
    st.error(f"⚠️ 无法获取宏观数据。底层诊断日志: `{err_msg}`")

if target_symbol == "__SECTOR_AGG__":
    st.markdown("---")
    st.markdown(f"##### 📋 {selected_sector} - 估值明细表")
    st.dataframe(sector_table_df, use_container_width=True, hide_index=True)
