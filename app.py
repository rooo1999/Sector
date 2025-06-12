# ==========================================================
#  FINAL BRUTE-FORCE DIAGNOSTIC SCRIPT
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import openpyxl 

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("Momentum Sector Trading Strategy")

#======================================================================
# --- CORE BACKTESTING & DATA PROCESSING FUNCTIONS ---
#======================================================================

def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        # Convert all potential sector columns to numeric, coercing errors to NaN, then drop rows with any NaN
        cols_to_convert = [col for col in df.columns if col != date_col]
        for col in cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error in load_data: {e}")
        return None

def calculate_indicators(df, sectors):
    st.info("Attempting to calculate indicators...")
    st.write("Data types passed into `calculate_indicators` function:")
    st.write(df.dtypes.to_frame('Dtype'))

    indicators_df = pd.DataFrame(index=df.index)
    
    for i, sector in enumerate(sectors):
        st.markdown(f"--- \n ### Processing Sector {i+1}: `{sector}`")
        
        # Isolate the series
        sector_series = df[sector]
        
        st.write(f"Type of `df['{sector}']` is: `{type(sector_series)}`")
        st.write(f"Data type of the series' contents is: `{sector_series.dtype}`")

        # The most important check
        has_ta = hasattr(sector_series, 'ta')
        if not has_ta:
            st.error(f"**CRITICAL FAILURE:** The Series for sector `{sector}` does NOT have the `.ta` attribute. This is the point of failure. Stopping execution.")
            st.stop()
        else:
            st.success(f"The Series for `{sector}` **successfully** has the `.ta` attribute.")

        try:
            st.write(f"Attempting `df['{sector}'].ta.sma(length=10)`...")
            # Use a simple indicator first
            sma_val = sector_series.ta.sma(length=10)
            if sma_val is None:
                 st.warning(f"SMA calculation for `{sector}` returned None.")
            else:
                 st.success(f"SMA calculation for `{sector}` was successful.")
                 # Only add to df if successful
                 indicators_df[f'{sector}_sma_test'] = sma_val

        except Exception as e:
            st.error(f"An exception occurred while calculating SMA for `{sector}`.")
            st.exception(e)
            st.stop()

    st.success("All sectors processed without fatal errors. If you see this, the rest of the app should work.")
    # The rest of your original function would go here, but we are stopping for debug.
    return indicators_df.dropna()


def run_backtest(price_df, indicators_df, sectors, benchmark_col, weights, top_n=2):
    # This function won't be called if the above fails.
    st.info("Backtest function called. This is a good sign.")
    # In a real run, you would put your original backtest logic here.
    # For now, just return something to avoid more errors.
    return {"portfolio_df": pd.DataFrame(), "benchmark_series": pd.Series(), "trades_df": pd.DataFrame(),
            "total_return": 0, "cagr": 0, "sharpe_ratio": 0, "latest_scores": pd.Series()}

#======================================================================
# --- STREAMLIT UI APPLICATION ---
#======================================================================

st.sidebar.header("⚙️ Strategy Parameters")
uploaded_file = st.sidebar.file_uploader( "Upload your Excel data file", type=["xlsx"] )

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None and not df.empty:
        st.success("✅ Data loaded successfully. Displaying initial data types:")
        st.write(df.dtypes.to_frame('Dtype'))

        all_columns = df.columns.tolist()
        benchmark_col = st.sidebar.selectbox("Select Benchmark", all_columns, index=len(all_columns)-1)
        sectors_to_run = st.sidebar.multiselect("Select Sectors", all_columns, default=[c for c in all_columns if c != benchmark_col])
        
        if st.sidebar.button("🚀 Run Calculation"):
            if not sectors_to_run:
                st.warning("Please select at least one sector.")
            else:
                with st.spinner("Running verbose calculation..."):
                    # This is the line that calls our heavily modified function
                    indicators_df = calculate_indicators(df, sectors_to_run)
                    
                    if not indicators_df.empty:
                        st.balloons()
                        st.header("Success!")
                        st.write("Indicators DataFrame created:")
                        st.dataframe(indicators_df)
    else:
        st.warning("Data could not be loaded or is empty after cleaning.")