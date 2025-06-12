# Final, clean code.
import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import openpyxl 

st.set_page_config(page_title="Momentum Sector Trading Strategy", layout="wide", initial_sidebar_state="expanded")

def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        if date_col is None:
            st.error("Error: 'Date' column not found.")
            return None
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        for col in df.columns:
            if col.lower() != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        df = pd.DataFrame(df.values, index=df.index, columns=df.columns) # The "laundering" fix
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_indicators(df, sectors):
    # ... (Your original, working calculate_indicators function) ...
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
    # ... (Your original, working run_backtest function) ...
    rebalancing_dates = price_df.resample('MS').first().index
    rebalancing_dates = rebalancing_dates[rebalancing_dates >= indicators_df.index[0]]
    trades, portfolio_values = [], []
    initial_capital = 100000
    current_cash = initial_capital
    latest_scores = pd.Series(dtype=float)
    for i in range(len(rebalancing_dates) - 1):
        start_date, end_date = rebalancing_dates[i], rebalancing_dates[i+1]
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
        latest_scores = sector_scores
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
    return { "portfolio_df": portfolio_df, "benchmark_series": benchmark_norm, "trades_df": trades_df, "total_return": total_return, "cagr": cagr, "sharpe_ratio": sharpe_ratio, "latest_scores": latest_scores.sort_values(ascending=False) }

# ... (Your original, working UI section) ...
st.title("Momentum-Based Sector Rotation Strategy")
st.sidebar.header("⚙️ Strategy Parameters")
uploaded_file = st.sidebar.file_uploader("Upload your Excel data file", type=["xlsx"])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None and not df.empty:
        all_columns = df.columns.tolist()
        benchmark_col = st.sidebar.selectbox("Select Benchmark Column", options=all_columns, index=len(all_columns)-1)
        available_sectors = [col for col in all_columns if col != benchmark_col]
        sectors_to_run = st.sidebar.multiselect("Select Sectors to Include", options=available_sectors, default=available_sectors)
        top_n = st.sidebar.slider("Number of Top Sectors to Select (N)", 1, len(sectors_to_run) if sectors_to_run else 1, min(2, len(sectors_to_run)) if sectors_to_run else 1)
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚖️ Indicator Weights")
        weights = {}
        weights['mom1m'] = st.sidebar.slider("1M Mom", 0.0, 1.0, 0.2, 0.05)
        weights['mom3m'] = st.sidebar.slider("3M Mom", 0.0, 1.0, 0.2, 0.05)
        weights['mom6m'] = st.sidebar.slider("6M Mom", 0.0, 1.0, 0.2, 0.05)
        weights['sma'] = st.sidebar.slider("SMA Pos", 0.0, 1.0, _0.1, 0.05)
        weights['rsi'] = st.sidebar.slider("RSI", 0.0, 1.0, 0.1, 0.05)
        weights['macd_hist'] = st.sidebar.slider("MACD Hist", 0.0, 1.0, 0.1, 0.05)
        weights['inv_vol'] = st.sidebar.slider("Inv Volatility", 0.0, 1.0, 0.1, 0.05)
        total_weight = sum(weights.values())
        if total_weight > 0: weights = {k: v / total_weight for k, v in weights.items()}
        if st.sidebar.button("🚀 Run Backtest"):
            if not sectors_to_run:
                st.warning("Please select at least one sector.")
            else:
                with st.spinner("Calculating indicators and running backtest..."):
                    indicators_df = calculate_indicators(df, sectors_to_run)
                    results = run_backtest(df, indicators_df, sectors_to_run, benchmark_col, weights, top_n)
                if results:
                    st.success("✅ Backtest Complete!")
                    st.subheader("📊 Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Return", f"{results['total_return']:.2%}")
                    col2.metric("CAGR (Annualized)", f"{results['cagr']:.2%}")
                    col3.metric("Sharpe Ratio (Annualized)", f"{results['sharpe_ratio']:.2f}")
                    st.subheader("📈 Equity Curve")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=results['portfolio_df'].index, y=results['portfolio_df']['Portfolio_Value'], name='Strategy Portfolio', line=dict(color='royalblue', width=2)))
                    fig.add_trace(go.Scatter(x=results['benchmark_series'].index, y=results['benchmark_series'].values, name=f'Benchmark ({benchmark_col})', line=dict(color='grey', dash='dash')))
                    fig.update_layout(title='Strategy vs. Benchmark Performance', xaxis_title='Date', yaxis_title='Portfolio Value (Indexed to 100k)', legend=dict(x=0.01, y=0.99))
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("🥇 View Latest Sector Scores"):
                        st.dataframe(results['latest_scores'].to_frame(name='Final Score').style.format("{:.4f}"))
                    with st.expander("📜 View Monthly Trades Log"):
                        st.dataframe(results['trades_df'])
                else:
                    st.error("Backtest could not be completed.")
else:
    st.info("👋 Welcome! Please upload an Excel data file to begin.")