import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import openpyxl 

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Momentum Sector Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL CONSTANT ---
INITIAL_CAPITAL = 100000

#======================================================================
# --- CORE BACKTESTING & DATA PROCESSING FUNCTIONS ---
#======================================================================

def load_data(uploaded_file):
    """Loads and robustly preprocesses the data from the Excel file."""
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
        df = pd.DataFrame(df.values, index=df.index, columns=df.columns) # "Laundering" fix
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_indicators(df, sectors, lookbacks):
    """Calculates indicators using DIRECT FUNCTION CALLS and editable lookbacks."""
    indicators_df = pd.DataFrame(index=df.index)
    
    for sector in sectors:
        sector_series = df[sector]
        
        indicators_df[f'{sector}_mom1m'] = sector_series.pct_change(periods=lookbacks['mom1'])
        indicators_df[f'{sector}_mom3m'] = sector_series.pct_change(periods=lookbacks['mom3'])
        indicators_df[f'{sector}_mom6m'] = sector_series.pct_change(periods=lookbacks['mom6'])
        
        sma_val = ta.sma(sector_series, length=lookbacks['sma'])
        indicators_df[f'{sector}_rsi'] = ta.rsi(sector_series, length=lookbacks['rsi'])
        macd = ta.macd(sector_series, fast=12, slow=26, signal=9)

        indicators_df[f'{sector}_sma_norm'] = (sector_series - sma_val) / sma_val if sma_val is not None and not sma_val.empty else 0
        
        if macd is not None and not macd.empty:
            indicators_df[f'{sector}_macd_hist'] = macd['MACDh_12_26_9']
        else:
            indicators_df[f'{sector}_macd_hist'] = 0
            
        daily_returns = sector_series.pct_change()
        volatility = daily_returns.rolling(window=lookbacks['volatility']).std()
        indicators_df[f'{sector}_inv_vol'] = 1 / volatility
        indicators_df[f'{sector}_inv_vol'].replace([np.inf, -np.inf], 0, inplace=True)

    return indicators_df.dropna()

def calculate_performance_metrics(series, initial_value):
    """Calculates all key performance metrics for a given returns series."""
    if series.empty or len(series) < 2:
        return {metric: 0 for metric in ["Total Return", "CAGR", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"]}

    total_return = (series.iloc[-1] / initial_value) - 1
    
    days = (series.index[-1] - series.index[0]).days
    cagr = ((series.iloc[-1] / initial_value) ** (365.25 / days)) - 1 if days > 0 else 0
    
    daily_returns = series.pct_change().dropna()
    if daily_returns.empty:
        return { "Total Return": total_return, "CAGR": cagr, "Annualized Volatility": 0, "Sharpe Ratio": 0, "Max Drawdown": 0, "Calmar Ratio": 0}

    volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility != 0 else 0
    
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        "Total Return": total_return, "CAGR": cagr, "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_drawdown, "Calmar Ratio": calmar_ratio
    }

def run_backtest(price_df, indicators_df, sectors, benchmark_col, weights, top_n=2):
    """Runs the full backtest and calculates all performance metrics."""
    rebalancing_dates = price_df.groupby([price_df.index.year, price_df.index.month]).head(1).index
    rebalancing_dates = rebalancing_dates[rebalancing_dates >= indicators_df.index[0]]

    trades, portfolio_values = [], []
    current_cash = INITIAL_CAPITAL
    latest_scores = pd.Series(dtype=float)
    
    previous_month_sectors, total_churned_positions = set(), 0
    
    for i in range(len(rebalancing_dates) - 1):
        start_date, end_date = rebalancing_dates[i], rebalancing_dates[i+1]
        month_df = price_df.loc[start_date:end_date].iloc[:-1]
        if month_df.empty: continue
            
        entry_price_date, exit_price_date = month_df.index[0], month_df.index[-1]
        
        signal_date_loc = price_df.index.get_loc(start_date)
        if signal_date_loc == 0: continue
        signal_date = price_df.index[signal_date_loc - 1]
        if signal_date not in indicators_df.index: continue

        sector_scores = pd.Series({ s: ( indicators_df.loc[signal_date].get(f'{s}_mom1m', 0) * weights['mom1m'] + indicators_df.loc[signal_date].get(f'{s}_mom3m', 0) * weights['mom3m'] + indicators_df.loc[signal_date].get(f'{s}_mom6m', 0) * weights['mom6m'] + indicators_df.loc[signal_date].get(f'{s}_sma_norm', 0) * weights['sma'] + indicators_df.loc[signal_date].get(f'{s}_rsi', 0) * weights['rsi'] + indicators_df.loc[signal_date].get(f'{s}_macd_hist', 0) * weights['macd_hist'] + indicators_df.loc[signal_date].get(f'{s}_inv_vol', 0) * weights['inv_vol'] ) for s in sectors })
        
        latest_scores = sector_scores 
        top_sectors = set(sector_scores.nlargest(top_n).index)
        
        total_churned_positions += len(previous_month_sectors - top_sectors)
        previous_month_sectors = top_sectors

        monthly_return = sum((price_df.loc[exit_price_date, s] - price_df.loc[entry_price_date, s]) / price_df.loc[entry_price_date, s] for s in top_sectors)
        avg_monthly_return = monthly_return / top_n if top_n > 0 else 0
        current_cash *= (1 + avg_monthly_return)
        
        portfolio_values.append({'Date': exit_price_date, 'Portfolio_Value': current_cash})
        trades.append({'Start Date': start_date.date(), 'End Date': exit_price_date.date(), 'Sectors Selected': ', '.join(sorted(list(top_sectors)))})

    if not portfolio_values: return None

    portfolio_df = pd.DataFrame(portfolio_values).set_index('Date')
    daily_portfolio = portfolio_df['Portfolio_Value'].resample('D').last().ffill()
    strategy_metrics = calculate_performance_metrics(daily_portfolio, INITIAL_CAPITAL)
    
    benchmark_data = price_df[benchmark_col]
    benchmark_series = benchmark_data.reindex(daily_portfolio.index, method='ffill').dropna()
    benchmark_metrics = calculate_performance_metrics(benchmark_series, benchmark_series.iloc[0])
    
    churn_ratio = total_churned_positions / (len(trades) * top_n) if (len(trades) * top_n) > 0 else 0

    return {
        "portfolio_df": portfolio_df, "benchmark_series": benchmark_series, "trades_df": pd.DataFrame(trades),
        "monthly_returns": portfolio_df['Portfolio_Value'].pct_change(), "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics, "latest_scores": latest_scores.sort_values(ascending=False),
        "churn_ratio": churn_ratio
    }

#======================================================================
# --- STREAMLIT UI APPLICATION ---
#======================================================================

st.title("📈 Advanced Momentum Sector Rotation Strategy")
st.sidebar.header("⚙️ Strategy Parameters")
uploaded_file = st.sidebar.file_uploader("Upload your Excel data file", type=["xlsx"])

if uploaded_file:
    df_full = load_data(uploaded_file)
    if df_full is not None and not df_full.empty:
        
        start_date = st.sidebar.date_input("Start Date", df_full.index.min(), min_value=df_full.index.min(), max_value=df_full.index.max())
        end_date = st.sidebar.date_input("End Date", df_full.index.max(), min_value=df_full.index.min(), max_value=df_full.index.max())
        df = df_full.loc[start_date:end_date]
        
        all_columns = df.columns.tolist()
        benchmark_col = st.sidebar.selectbox("Select Benchmark", options=all_columns, index=len(all_columns)-1)
        available_sectors = [col for col in all_columns if col != benchmark_col]
        sectors_to_run = st.sidebar.multiselect("Select Sectors", options=available_sectors, default=available_sectors)
        top_n = st.sidebar.slider("Sectors to Invest In (N)", 1, len(sectors_to_run) if sectors_to_run else 1, min(2, len(sectors_to_run)) if sectors_to_run else 1)

        st.sidebar.subheader("⚖️ Indicator Weights")
        weights = {}
        c1, c2 = st.sidebar.columns(2)
        weights['mom1m'] = c1.slider("1M Mom", 0.0, 1.0, 0.2, 0.05)
        weights['mom3m'] = c2.slider("3M Mom", 0.0, 1.0, 0.2, 0.05)
        weights['mom6m'] = c1.slider("6M Mom", 0.0, 1.0, 0.2, 0.05)
        weights['sma'] = c2.slider("SMA Pos", 0.0, 1.0, 0.1, 0.05)
        weights['rsi'] = c1.slider("RSI", 0.0, 1.0, 0.1, 0.05)
        weights['macd_hist'] = c2.slider("MACD Hist", 0.0, 1.0, 0.1, 0.05)
        weights['inv_vol'] = st.sidebar.slider("Inverse Volatility", 0.0, 1.0, 0.1, 0.05, help="Higher weight prefers less volatile sectors.")
        total_weight = sum(weights.values())
        if total_weight > 0: weights = {k: v / total_weight for k, v in weights.items()}
        
        with st.sidebar.expander("🔧 Advanced: Edit Lookback Periods"):
            lookbacks = {}
            lookbacks['mom1'] = st.number_input("1M Momentum Days", 1, 252, 21)
            lookbacks['mom3'] = st.number_input("3M Momentum Days", 1, 252, 63)
            lookbacks['mom6'] = st.number_input("6M Momentum Days", 1, 504, 126)
            lookbacks['sma'] = st.number_input("SMA Length", 1, 200, 30)
            lookbacks['rsi'] = st.number_input("RSI Length", 1, 200, 14)
            lookbacks['volatility'] = st.number_input("Volatility Window", 1, 200, 21)

        if st.sidebar.button("🚀 Run Backtest"):
            if not sectors_to_run:
                st.warning("Please select at least one sector.")
            else:
                with st.spinner("Calculating..."):
                    indicators_df = calculate_indicators(df, sectors_to_run, lookbacks)
                    results = run_backtest(df, indicators_df, sectors_to_run, benchmark_col, weights, top_n)
                
                if results:
                    st.success("✅ Backtest Complete!")
                    
                    st.subheader("📊 Key Performance Indicators")
                    perf_data = {
                        'Metric': list(results['strategy_metrics'].keys()) + ["Churn Ratio"],
                        'Strategy': list(results['strategy_metrics'].values()) + [results['churn_ratio']],
                        'Benchmark': list(results['benchmark_metrics'].values()) + ["-"]
                    }
                    perf_df = pd.DataFrame(perf_data).set_index('Metric')
                    
                    format_mapping = {
                        "Total Return": "{:,.2%}", "CAGR": "{:,.2%}", "Annualized Volatility": "{:,.2%}",
                        "Max Drawdown": "{:,.2%}", "Churn Ratio": "{:,.2%}", "Sharpe Ratio": "{:.2f}", "Calmar Ratio": "{:.2f}"
                    }
                    st.dataframe(perf_df.style.format(formatter=format_mapping))

                    st.subheader("📈 Equity Curve & Drawdowns")
                    portfolio_series = results['portfolio_df']['Portfolio_Value'].resample('D').last().ffill()
                    benchmark_series_norm = (results['benchmark_series'] / results['benchmark_series'].iloc[0]) * INITIAL_CAPITAL if not results['benchmark_series'].empty else pd.Series()
                    
                    if not portfolio_series.empty:
                        strat_dd = (portfolio_series - portfolio_series.cummax()) / portfolio_series.cummax()
                        bench_dd = (benchmark_series_norm - benchmark_series_norm.cummax()) / benchmark_series_norm.cummax() if not benchmark_series_norm.empty else pd.Series()

                        from plotly.subplots import make_subplots
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                        fig.add_trace(go.Scatter(x=portfolio_series.index, y=portfolio_series, name='Strategy', line=dict(color='royalblue')), row=1, col=1)
                        if not benchmark_series_norm.empty:
                            fig.add_trace(go.Scatter(x=benchmark_series_norm.index, y=benchmark_series_norm, name='Benchmark', line=dict(color='grey', dash='dash')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd, name='Strategy Drawdown', fill='tozeroy', line=dict(color='rgba(65, 105, 225, 0.5)')), row=2, col=1)
                        if not bench_dd.empty:
                            fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd, name='Benchmark Drawdown', fill='tozeroy', line=dict(color='rgba(128, 128, 128, 0.5)')), row=2, col=1)
                        
                        fig.update_layout(height=600, title_text="Performance and Underwater Equity Curve", legend_tracegroupgap=180)
                        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
                        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("🥇 View Latest Sector Scores"):
                        st.dataframe(results['latest_scores'].to_frame(name='Final Score').style.format("{:.4f}"))

                    with st.expander("📜 View Historical Monthly Trades"):
                        st.dataframe(results['trades_df'].set_index('Start Date'))

                    with st.expander("📅 View Historical Monthly Returns"):
                        st.dataframe(results['monthly_returns'].to_frame(name="Return").style.format("{:.2%}"))
                else:
                    st.error("Backtest could not be completed. The selected date range might be too short for the given lookback periods.")
else:
    st.info("👋 Welcome! Please upload an Excel data file to begin.")