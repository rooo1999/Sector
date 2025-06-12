# ==========================================================
#  FINAL DIAGNOSTIC SCRIPT
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import openpyxl 
from io import StringIO # Needed for the debug info

# --- Page Configuration ---
st.set_page_config(
    page_title="Momentum Sector Trading Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

#======================================================================
# --- CORE BACKTESTING & DATA PROCESSING FUNCTIONS ---
#======================================================================

def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        if date_col is None:
            st.error("Error: 'Date' column not found in the uploaded file.")
            return None
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

def calculate_indicators(df, sectors):
    # This function should now work if data types are correct
    indicators_df = pd.DataFrame(index=df.index)
    volatility_window = 21
    for sector in sectors:
        indicators_df[f'{sector}_mom1m'] = df[sector].pct_change(periods=21) 
        indicators_df[f'{sector}_mom3m'] = df[sector].pct_change(periods=63)
        indicators_df[f'{sector}_mom6m'] = df[sector].pct_change(periods=126)
        sma_val = df[sector].ta.sma(length=30)
        indicators_df[f'{sector}_sma_norm'] = (df[sector] - sma_val) / sma_val
        indicators_df[f'{sector}_rsi'] = df[sector].ta.rsi(length=14)
        macd = df[sector].ta.macd(fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            indicators_df[f'{sector}_macd_hist'] = macd['MACDh_12_26_9']
        else:
            indicators_df[f'{sector}_macd_hist'] = 0
        daily_returns = df[sector].pct_change()
        volatility = daily_returns.rolling(window=volatility_window).std()
        indicators_df[f'{sector}_inv_vol'] = 1 / volatility
        indicators_df[f'{sector}_inv_vol'].replace([np.inf, -np.inf], 0, inplace=True)
    return indicators_df.dropna()

def run_backtest(price_df, indicators_df, sectors, benchmark_col, weights, top_n=2):
    # This is your original backtest function, it remains unchanged
    rebalancing_dates = price_df.resample('MS').first().index
    rebalancing_dates = rebalancing_dates[rebalancing_dates >= indicators_df.index[0]]
    trades = []
    portfolio_values = []
    initial_capital = 100000
    current_cash = initial_capital
    for i in range(len(rebalancing_dates) - 1):
        start_date = rebalancing_dates[i]
        end_date = rebalancing_dates[i+1]
        month_df = price_df.loc[start_date:end_date].iloc[:-1]
        if month_df.empty: continue
        entry_price_date, exit_price_date = month_df.index[0], month_df.index[-1]
        signal_date_loc = price_df.index.get_loc(start_date) - 1
        if signal_date_loc < 0: continue
        signal_date = price_df.index[signal_date_loc]
        if signal_date not in indicators_df.index: continue
        latest_indicators = indicators_df.loc[signal_date]
        sector_scores = pd.Series(index=sectors, dtype=float)
        for sector in sectors:
            score = 0
            score += latest_indicators.get(f'{sector}_mom1m', 0) * weights['mom1m']
            score += latest_indicators.get(f'{sector}_mom3m', 0) * weights['mom3m']
            score += latest_indicators.get(f'{sector}_mom6m', 0) * weights['mom6m']
            score += latest_indicators.get(f'{sector}_sma_norm', 0) * weights['sma']
            score += latest_indicators.get(f'{sector}_rsi', 0) * weights['rsi']
            score += latest_indicators.get(f'{sector}_macd_hist', 0) * weights['macd_hist']
            score += latest_indicators.get(f'{sector}_inv_vol', 0) * weights['inv_vol']
            sector_scores[sector] = score
        top_sectors = sector_scores.nlargest(top_n).index.tolist()
        monthly_return = 0
        trade_info = {'Start Date': start_date.date(), 'End Date': exit_price_date.date()}
        for sector in top_sectors:
            entry_price, exit_price = price_df.loc[entry_price_date, sector], price_df.loc[exit_price_date, sector]
            sector_return = (exit_price - entry_price) / entry_price
            monthly_return += sector_return
            trade_info[f'Selected Sector: {sector}'] = f"{sector_return:.2%}"
        avg_monthly_return = monthly_return / top_n
        current_cash *= (1 + avg_monthly_return)
        portfolio_values.append({'Date': exit_price_date, 'Portfolio_Value': current_cash})
        trades.append(trade_info)
    if not portfolio_values: return None
    portfolio_df = pd.DataFrame(portfolio_values).set_index('Date')
    benchmark_series = price_df[benchmark_col].loc[portfolio_df.index]
    benchmark_norm = (benchmark_series / benchmark_series.iloc[0]) * initial_capital
    total_return = (portfolio_df['Portfolio_Value'].iloc[-1] / initial_capital) - 1
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    cagr = ((portfolio_df['Portfolio_Value'].iloc[-1] / initial_capital) ** (1/years)) - 1 if years > 0 else 0
    monthly_returns = portfolio_df['Portfolio_Value'].pct_change().dropna()
    sharpe_ratio = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12) if monthly_returns.std() != 0 else 0
    trades_df = pd.DataFrame(trades).fillna('-')
    return { "portfolio_df": portfolio_df, "benchmark_series": benchmark_norm, "trades_df": trades_df, "total_return": total_return, "cagr": cagr, "sharpe_ratio": sharpe_ratio, "latest_scores": sector_scores.sort_values(ascending=False) }

#======================================================================
# --- STREAMLIT UI APPLICATION ---
#======================================================================

st.title("Momentum-Based Sector Rotation Strategy")

st.sidebar.header("⚙️ Strategy Parameters")
uploaded_file = st.sidebar.file_uploader( "Upload your Excel data file", type=["xlsx"] )

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        
        # =================================================================
        # ==           CRUCIAL DEBUGGING BLOCK HERE                    ==
        # =================================================================
        st.subheader("🕵️‍♀️ Debugging: Raw Data Structure Check")
        st.warning("""
        **Action Required:** Look at the 'Dtype' column below. All your sector columns **MUST** be a number type (`float64` or `int64`).
        If any sector column shows up as **`object`**, that is the source of the error.
        It means that column contains text (like commas, $, or 'N/A') in your Excel file.
        You must open your Excel file, find that column, and format it purely as a number.
        """)
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        # =================================================================

        all_columns = df.columns.tolist()
        benchmark_col = st.sidebar.selectbox("Select Benchmark Column", options=all_columns, index=len(all_columns)-1)
        available_sectors = [col for col in all_columns if col != benchmark_col]
        sectors_to_run = st.sidebar.multiselect("Select Sectors to Include", options=available_sectors, default=available_sectors)
        top_n = st.sidebar.slider("Number of Top Sectors to Select (N)", min_value=1, max_value=len(sectors_to_run) if sectors_to_run else 1, value=min(2, len(sectors_to_run)) if sectors_to_run else 1)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚖️ Indicator Weights")
        weights = {}
        weights['mom1m'] = st.sidebar.slider("1M Mom", 0.0, 1.0, 0.2, 0.05)
        weights['mom3m'] = st.sidebar.slider("3M Mom", 0.0, 1.0, 0.2, 0.05)
        weights['mom6m'] = st.sidebar.slider("6M Mom", 0.0, 1.0, 0.2, 0.05)
        weights['sma'] = st.sidebar.slider("SMA Pos", 0.0, 1.0, 0.1, 0.05)
        weights['rsi'] = st.sidebar.slider("RSI", 0.0, 1.0, 0.1, 0.05)
        weights['macd_hist'] = st.sidebar.slider("MACD Hist", 0.0, 1.0, 0.1, 0.05)
        weights['inv_vol'] = st.sidebar.slider("Inv Volatility", 0.0, 1.0, 0.1, 0.05)
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        if st.sidebar.button("🚀 Run Backtest"):
            # The app will likely error after this button press if a Dtype is 'object'
            with st.spinner("Running..."):
                results = None
                try:
                    indicators_df = calculate_indicators(df, sectors_to_run)
                    results = run_backtest(df, indicators_df, sectors_to_run, benchmark_col, weights, top_n)
                except Exception as e:
                    st.error(f"An error occurred during calculation. This is likely due to an 'object' data type. Please check the debug info above.")
                    st.exception(e)
                
                if results:
                    st.success("✅ Backtest Complete!")
                    # ... (rest of the display logic) ...

else:
    st.info("👋 Welcome! Please upload an Excel data file to begin.")