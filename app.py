import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import openpyxl  # Required by pandas for reading .xlsx files

# --- Page Configuration ---
st.set_page_config(
    page_title="Momentum Sector Trading Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

#======================================================================
# --- CORE BACKTESTING & DATA PROCESSING FUNCTIONS ---
# (Previously in backtester.py)
#======================================================================

def load_data(uploaded_file):
    """Loads and preprocesses the data from an uploaded Excel file."""
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        # Find the date column (robustly handles different names like 'Date' or 'date')
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
    """Calculates all technical indicators and momentum for each sector."""
    indicators_df = pd.DataFrame(index=df.index)
    volatility_window = 21  # Approx 1 month of trading days

    for sector in sectors:
        # 1. Momentum (Rate of Change) - periods are trading days
        indicators_df[f'{sector}_mom1m'] = df[sector].pct_change(periods=21) 
        indicators_df[f'{sector}_mom3m'] = df[sector].pct_change(periods=63)
        indicators_df[f'{sector}_mom6m'] = df[sector].pct_change(periods=126)

        # 2. SMA (Normalized) - How far is the price from its 30-day SMA?
        sma_val = df[sector].ta.sma(length=30)
        indicators_df[f'{sector}_sma_norm'] = (df[sector] - sma_val) / sma_val

        # 3. RSI (14-day)
        indicators_df[f'{sector}_rsi'] = df[sector].ta.rsi(length=14)

        # 4. MACD Histogram - A measure of momentum acceleration
        macd = df[sector].ta.macd(fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            indicators_df[f'{sector}_macd_hist'] = macd['MACDh_12_26_9']
        else:
            indicators_df[f'{sector}_macd_hist'] = 0


        # 5. Inverse Volatility (Suggested New Parameter)
        daily_returns = df[sector].pct_change()
        volatility = daily_returns.rolling(window=volatility_window).std()
        indicators_df[f'{sector}_inv_vol'] = 1 / volatility
        indicators_df[f'{sector}_inv_vol'].replace([np.inf, -np.inf], 0, inplace=True)

    return indicators_df.dropna()


def run_backtest(price_df, indicators_df, sectors, benchmark_col, weights, top_n=2):
    """Runs the monthly rebalancing backtest."""
    # Identify rebalancing dates (first trading day of each month)
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
        if month_df.empty:
            continue
            
        entry_price_date = month_df.index[0]
        exit_price_date = month_df.index[-1]
        
        # --- Ranking Logic ---
        # Use data from the day *before* rebalancing to avoid lookahead bias
        signal_date_loc = price_df.index.get_loc(start_date) - 1
        if signal_date_loc < 0: continue # Skip if it's the first day
        signal_date = price_df.index[signal_date_loc]
        
        if signal_date not in indicators_df.index:
            continue

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
        
        # --- Execute Trades & Calculate Monthly PnL ---
        monthly_return = 0
        trade_info = {'Start Date': start_date.date(), 'End Date': exit_price_date.date()}
        
        for sector in top_sectors:
            entry_price = price_df.loc[entry_price_date, sector]
            exit_price = price_df.loc[exit_price_date, sector]
            sector_return = (exit_price - entry_price) / entry_price
            monthly_return += sector_return
            trade_info[f'Selected Sector: {sector}'] = f"{sector_return:.2%}"

        avg_monthly_return = monthly_return / top_n
        current_cash *= (1 + avg_monthly_return)
        portfolio_values.append({'Date': exit_price_date, 'Portfolio_Value': current_cash})
        trades.append(trade_info)

    # --- Performance Calculation ---
    if not portfolio_values:
        return None # Return None if no trades were made

    portfolio_df = pd.DataFrame(portfolio_values).set_index('Date')
    
    benchmark_series = price_df[benchmark_col].loc[portfolio_df.index]
    benchmark_norm = (benchmark_series / benchmark_series.iloc[0]) * initial_capital
    
    total_return = (portfolio_df['Portfolio_Value'].iloc[-1] / initial_capital) - 1
    years = (portfolio_df.index[-1] - portfolio_df.index[0]).days / 365.25
    cagr = ((portfolio_df['Portfolio_Value'].iloc[-1] / initial_capital) ** (1/years)) - 1 if years > 0 else 0
    
    monthly_returns = portfolio_df['Portfolio_Value'].pct_change().dropna()
    sharpe_ratio = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12) if monthly_returns.std() != 0 else 0
    
    trades_df = pd.DataFrame(trades).fillna('-')

    return {
        "portfolio_df": portfolio_df,
        "benchmark_series": benchmark_norm,
        "trades_df": trades_df,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_ratio": sharpe_ratio,
        "latest_scores": sector_scores.sort_values(ascending=False)
    }

#======================================================================
# --- STREAMLIT UI APPLICATION ---
# (Previously in app.py)
#======================================================================

# --- App Title ---
st.title("Momentum-Based Sector Rotation Strategy")
st.markdown("""
This dashboard backtests a trading strategy that invests in the top-performing sectors based on a weighted combination of momentum and technical indicators.
The portfolio is rebalanced on the first trading day of every month.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Strategy Parameters")

uploaded_file = st.sidebar.file_uploader(
    "Upload your Excel data file", 
    type=["xlsx"],
    help="The Excel file should have 'Date' in the first column and sector/benchmark prices in subsequent columns."
)

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        all_columns = df.columns.tolist()
        
        benchmark_col = st.sidebar.selectbox(
            "Select Benchmark Column", 
            options=all_columns, 
            index=len(all_columns)-1
        )
        
        available_sectors = [col for col in all_columns if col != benchmark_col]
        
        sectors_to_run = st.sidebar.multiselect(
            "Select Sectors to Include",
            options=available_sectors,
            default=available_sectors
        )

        top_n = st.sidebar.slider(
            "Number of Top Sectors to Select (N)", 
            min_value=1, 
            max_value=len(sectors_to_run) if sectors_to_run else 1, 
            value=min(2, len(sectors_to_run)) if sectors_to_run else 1
        )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚖️ Indicator Weights")
        st.sidebar.markdown("Define the importance of each indicator in the ranking score. Weights are automatically normalized to sum to 1.")

        weights = {}
        weights['mom1m'] = st.sidebar.slider("1-Month Momentum Weight", 0.0, 1.0, 0.2, 0.05)
        weights['mom3m'] = st.sidebar.slider("3-Month Momentum Weight", 0.0, 1.0, 0.2, 0.05)
        weights['mom6m'] = st.sidebar.slider("6-Month Momentum Weight", 0.0, 1.0, 0.2, 0.05)
        weights['sma'] = st.sidebar.slider("SMA (30d) Position Weight", 0.0, 1.0, 0.1, 0.05)
        weights['rsi'] = st.sidebar.slider("RSI (14d) Weight", 0.0, 1.0, 0.1, 0.05)
        weights['macd_hist'] = st.sidebar.slider("MACD Histogram Weight", 0.0, 1.0, 0.1, 0.05)
        weights['inv_vol'] = st.sidebar.slider("Inverse Volatility Weight", 0.0, 1.0, 0.1, 0.05, help="Higher weight prefers less volatile sectors.")
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        if st.sidebar.button("🚀 Run Backtest"):
            if not sectors_to_run:
                st.warning("Please select at least one sector to run the backtest.")
            else:
                with st.spinner("Calculating indicators and running backtest... This may take a moment."):
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
                    
                    # --- Detailed Results in Expanders ---
                    with st.expander("🥇 View Latest Sector Scores"):
                        st.markdown("These are the final scores calculated on the last available day, which would determine the *next* month's investment decision.")
                        st.dataframe(results['latest_scores'].to_frame(name='Final Score').style.format("{:.4f}"))

                    with st.expander("📜 View Monthly Trades Log"):
                        st.markdown("This table shows the top sectors chosen each month and their performance during that holding period.")
                        st.dataframe(results['trades_df'])
                else:
                    st.error("Backtest could not be completed. This might be due to insufficient data for the lookback periods. Please try a larger dataset.")

else:
    st.info("👋 Welcome! Please upload an Excel data file using the sidebar to begin.")
    st.image("https://i.imgur.com/v14A446.png", caption="Ensure your Excel file has a 'Date' column and price columns for sectors and a benchmark.", width=600)
    