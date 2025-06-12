# ===================================================================
# ==                 MINIMAL DEBUGGING SCRIPT                    ==
# == If this works, the environment is fixed.                      ==
# ===================================================================
import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- Page Config & Title (CHANGED FOR VERIFICATION) ---
st.set_page_config(layout="wide")
st.title("DEBUGGING: Pandas-TA Environment Test")
st.warning("If you see this title, the new code has been deployed successfully.")

# --- Display Package Versions (The most important check) ---
st.header("1. Verifying Installed Package Versions")
st.code(f"""
- Pandas Version:      {pd.__version__}
- NumPy Version:       {np.__version__}
- Pandas-TA Version:   {ta.__version__}
""")

# --- Create a simple DataFrame ---
st.header("2. Creating a Test DataFrame")
data = {'close': np.random.randn(50).cumsum() + 100}
df = pd.DataFrame(data)
st.write("Sample of the test data:", df.head())

# --- The CRITICAL Test ---
st.header("3. The Test: Attempting to use the '.ta' accessor")
try:
    # This is the line that fails in your app.
    # We are calling it directly on our test DataFrame.
    df['RSI'] = df.ta.rsi(length=14)
    
    st.success("SUCCESS! The '.ta' accessor is working correctly.")
    st.write("DataFrame with RSI calculated:", df)

except Exception as e:
    st.error("TEST FAILED. The '.ta' accessor is still not attached.")
    st.exception(e)