# # app.py
# import streamlit as st
# import pandas as pd
# import plotly.graph_objs as go
# from data_utils import load_and_preprocess
# from model_utils import load_resources, create_sequences, predict

# st.set_page_config(page_title="Stock Predictor", layout="wide")

# # Load model & scaler once
# model, scaler = load_resources()

# # Apply custom CSS
# with open("assets/style.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# # Navbar
# st.markdown("""
# <nav class="navbar">
#   <span class="nav-title">StockPredict</span>
#   <div class="nav-links">
#     <a href="#">Dashboard</a>
#     <a href="#">Docs</a>
#     <a href="#">About</a>
#   </div>
# </nav>
# """, unsafe_allow_html=True)

# # Sidebar inputs
# st.sidebar.header("Controls")
# ticker = st.sidebar.text_input("Ticker", "RELIANCE.NS")
# start  = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
# end    = st.sidebar.date_input("End Date",   pd.to_datetime("2024-12-31"))
# lookback = st.sidebar.number_input("Look-back (days)", min_value=5, max_value=60, value=20)

# if st.sidebar.button("Run"):
#     df = load_and_preprocess(ticker, start, end)
#     if df.shape[0] <= lookback:
#         st.error(f"Need ≥ {lookback+1} rows; got {df.shape[0]}. Try a wider date range.")
#         st.stop()


#     with st.spinner("Computing predictions…"):
#         Xs, Xl, y_true, dates = create_sequences(df, lookback=lookback)
#         preds = predict(model, scaler, Xs, Xl)

#     # Layout: metrics + chart
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st.subheader("🔮 Next-Day Prediction")
#         st.metric("Predicted Price", f"₹{preds[-1]:.2f}")
#         st.write(f"{len(preds)} total predictions")
#     with col2:
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=dates, y=y_true,   name="Actual"))
#         fig.add_trace(go.Scatter(x=dates, y=preds,    name="Predicted", line=dict(dash="dash")))
#         fig.update_layout(title="Actual vs Predicted Close Price", template="plotly_white")
#         st.plotly_chart(fig, use_container_width=True)


# # app.py
# import streamlit as st
# import pandas as pd
# import plotly.graph_objs as go

# from data_utils import load_and_preprocess
# from model_utils import load_resources, create_sequences, predict, multi_day_predict

# # --- Page config & CSS ---
# st.set_page_config(page_title="StockPredict", layout="wide")
# with open("assets/style.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# # --- Navbar ---
# st.markdown("""
# <nav class="navbar">
#   <span class="nav-title">StockPredict</span>
#   <div class="nav-links">
#     <a href="#">Dashboard</a>
#     <a href="#">Analytics</a>
#     <a href="#">About</a>
#   </div>
# </nav>
# """, unsafe_allow_html=True)

# # --- Sidebar Controls ---
# st.sidebar.header("Controls")
# ticker           = st.sidebar.text_input("Ticker", "RELIANCE.NS")
# start            = st.sidebar.date_input("Start Date", pd.to_datetime("2024-11-01"))
# end              = st.sidebar.date_input("End Date",   pd.to_datetime("2025-03-22"))
# lookback         = st.sidebar.number_input("Look-back (days)", min_value=5, max_value=60, value=20)
# forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 60, 7)
# run_button       = st.sidebar.button("Run")

# # --- Load model & scaler once ---
# model, scaler = load_resources()

# if run_button:
#     # 1) Fetch & validate data
#     df = load_and_preprocess(ticker, start, end)
#     if df.shape[0] <= lookback:
#         st.error(f"Need at least {lookback+1} rows; got {df.shape[0]}.")
#         st.stop()

#     # 2) Build sequences & predict
#     Xs, Xl, y_true, dates = create_sequences(df, lookback)
#     y_next = predict(model, scaler, Xs, Xl)

#     # 3) Multi-day forecast
#     seed_seq, seed_last = Xs[-1], Xl[-1]
#     future_preds = multi_day_predict(model, scaler, seed_seq, seed_last, forecast_horizon)
#     future_dates = pd.bdate_range(dates[-1], periods=forecast_horizon+1, freq="B")[1:]

#     # --- Tabs for organized display ---
#     tab1, tab2, tab3 = st.tabs(["📈 Price", "📊 Returns", "🔊 Volume"])

#     with tab1:
#         st.subheader("🔮 Price Forecast")
#         # Metrics row
#         c1, c2, c3, c4 = st.columns(4)
#         # Extract scalar last close
#         # last_close = df["Close"].iloc[-1].item()
#         close_prices = df["Close"].squeeze()
#         last_close = close_prices.iloc[-1]
#         c1.metric("Last Close",       f"₹{last_close:.2f}")
#         c2.metric("Next-Day Pred",    f"₹{y_next[-1]:.2f}")
#         c3.metric("Horizon End Pred", f"₹{future_preds[-1]:.2f}")
#         c4.metric("Total Predictions", f"{len(future_preds)} days")

#         # Combined plot
#         fig = go.Figure([
#             go.Scatter(x=dates,       y=y_true,      name="Actual"),
#             go.Scatter(x=dates,       y=y_next,      name="Next-Day",   line=dict(dash="dash")),
#             go.Scatter(x=future_dates,y=future_preds,name="Future",     line=dict(color="green", dash="dot")),
#         ])
#         fig.update_layout(title="Actual & Predicted Prices", template="plotly_white")
#         st.plotly_chart(fig, use_container_width=True)

#     with tab2:
#         st.subheader("📊 Daily Returns")
#         # Convert to DataFrame so st.line_chart won’t misinterpret rename
#         returns = close_prices.pct_change().dropna()
#         st.line_chart(returns.rename("Returns"))


#     with tab3:
#         st.subheader("🔊 Volume History")
#         st.bar_chart(df["Volume"])


import streamlit as st
st.set_page_config(page_title="StockPredict", layout="wide")
import pandas as pd
import plotly.graph_objs as go
from streamlit_option_menu import option_menu
import datetime

# … your existing sidebar code …
# start = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
# end   = st.sidebar.date_input("End Date",   datetime.date.today())

# Clamp end so it’s never in the future
# if end > datetime.date.today():
#     st.warning(f"End date {end} is in the future—clamping to today.")
#     end = datetime.date.today()

from data_utils import load_and_preprocess
from model_utils import load_resources, create_sequences, predict, multi_day_predict

# --- Page config & CSS ---
# st.set_page_config(page_title="StockPredict", layout="wide")
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Top horizontal menu ---
selected = option_menu(
    menu_title=None,                      # no title
    options=["Dashboard", "Analytics", "About"],
    icons=["house", "bar-chart-line", "info-circle"],
    menu_icon="cast",                    # unused
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#2563EB"},
        "icon": {"color": "white", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "color": "white", "font-weight": "500"},
        "nav-link-selected": {"background-color": "#1E40AF"},
    }
)

# --- Load model & scaler once ---
model, scaler = load_resources()

# --- DASHBOARD ---
if selected == "Dashboard":
    st.title("🏠 Dashboard")
    # Sidebar controls
    st.sidebar.header("Controls")
    ticker           = st.sidebar.text_input("Ticker", "RELIANCE.NS")
    start = st.sidebar.date_input("Start Date", pd.to_datetime("2024-11-01"))
    end   = st.sidebar.date_input("End Date",   datetime.date.today())
    lookback         = 20   # fixed
    forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 15, 7)
    if st.sidebar.button("Run"):
        df = load_and_preprocess(ticker, start, end)
        if df.shape[0] <= lookback:
            st.error(f"Need ≥ {lookback+1} rows; got {df.shape[0]}.")
            st.stop()

        # build and predict
        Xs, Xl, y_true, dates = create_sequences(df, lookback)
        y_next  = predict(model, scaler, Xs, Xl)
        future_preds = multi_day_predict(model, scaler, Xs[-1], Xl[-1], forecast_horizon)
        future_dates = pd.bdate_range(dates[-1], periods=forecast_horizon+1, freq="B")[1:]

        # metrics
        st.subheader("🔮 Price Forecast")
        c1, c2, c3, c4 = st.columns(4)
        cp = df["Close"].squeeze()
        c1.metric("Last Close",       f"₹{cp.iloc[-1]:.2f}")
        c2.metric("Next-Day Pred",    f"₹{y_next[-1]:.2f}")
        c3.metric("Horizon End Pred", f"₹{future_preds[-1]:.2f}")
        c4.metric("Total Predictions",f"{len(future_preds)} days")

        # plot
        fig = go.Figure([
            go.Scatter(x=dates,        y=y_true,      name="Actual"),
            go.Scatter(x=dates,        y=y_next,      name="Next-Day", line=dict(dash="dash")),
            go.Scatter(x=future_dates, y=future_preds,name="Future",   line=dict(color="green", dash="dot")),
        ])
        fig.update_layout(title="Actual & Predicted Prices", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # table
        table = pd.DataFrame({
            "Date": future_dates.date,
            "Predicted Close": [f"₹{x:.2f}" for x in future_preds]
        })
        st.subheader("📋 Future Predictions")
        st.table(table)

# --- ANALYTICS ---
elif selected == "Analytics":
    st.title("📊 Analytics")
    st.markdown("""
    **Technical Indicator Explorer**  
    - Overlay Close price with SMA, EMA, RSI, MACD, Bollinger Bands  
    - Interactive sliders to adjust window lengths  

    **Model Diagnostics**  
    - Training vs. validation loss & MAE curves  
    - Residual error histograms & KDE  

    **Scenario Testing**  
    - What-if sliders on key features  
    - Batch CSV upload for bulk forecasts
    """)

# --- ABOUT ---
elif selected == "About":
    st.title("ℹ️ About StockPredict")
    st.markdown("""
**StockPredict** is a hybrid LSTM + Dense neural network wrapped in a sleek dashboard.

- **Data Engineering:** `yfinance` + `pandas` for OHLC and indicators (SMA/EMA/RSI/MACD/Bollinger).  
- **Model:**  
  - LSTM branch for temporal patterns  
  - Dense branch for snapshot features  
  - Merged, normalized layers for final prediction  
- **Features:**  
  - Next‐day & multi‐day forecasts (up to 15 days)  
  - Interactive controls, metrics cards, rich Plotly visuals  
  - Analytics & scenario‐testing modules  
    """)

# import streamlit as st
# import pandas as pd
# import plotly.graph_objs as go
# from streamlit_option_menu import option_menu

# from data_utils import load_and_preprocess
# from model_utils import load_resources, create_sequences, predict, multi_day_predict

# # --- Page config & CSS ---
# st.set_page_config(page_title="StockPredict", layout="wide")
# with open("assets/style.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# # --- Top horizontal menu ---
# selected = option_menu(
#     menu_title=None,                      # no title
#     options=["Dashboard", "Analytics", "About"],
#     icons=["house", "bar-chart-line", "info-circle"],
#     menu_icon="cast",                    # unused
#     default_index=0,
#     orientation="horizontal",
#     styles={
#         "container": {"padding": "0!important", "background-color": "#2563EB"},
#         "icon": {"color": "white", "font-size": "20px"}, 
#         "nav-link": {"font-size": "16px", "color": "white", "font-weight": "500"},
#         "nav-link-selected": {"background-color": "#1E40AF"},
#     }
# )

# # --- Load model & scaler once ---
# model, scaler = load_resources()

# # --- DASHBOARD ---
# if selected == "Dashboard":
#     st.title("🏠 Dashboard")
#     # Sidebar controls
#     st.sidebar.header("Controls")
#     ticker           = st.sidebar.text_input("Ticker", "RELIANCE.NS")
#     start            = st.sidebar.date_input("Start Date", pd.to_datetime("2024-11-01"))
#     end              = st.sidebar.date_input("End Date",   pd.to_datetime("2025-03-22"))
#     lookback         = 20   # fixed
#     forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 15, 7)
#     if st.sidebar.button("Run"):
#         raw_df, df_scaled = load_and_preprocess(ticker, start, end)

#         # operate on the *raw* DataFrame for plotting & sequence creation
#         df = raw_df
#         if df.shape[0] <= lookback:
#             st.error(f"Need ≥ {lookback+1} rows; got {df.shape[0]}.")
#             st.stop()

#         # build and predict
#         Xs, Xl, y_true, dates = create_sequences(df, lookback)
#         y_next  = predict(model, scaler, Xs, Xl)
#         future_preds = multi_day_predict(model, scaler, Xs[-1], Xl[-1], forecast_horizon)
#         future_dates = pd.bdate_range(dates[-1], periods=forecast_horizon+1, freq="B")[1:]

#         # metrics
#         st.subheader("🔮 Price Forecast")
#         c1, c2, c3, c4 = st.columns(4)
#         cp = df["Close"].squeeze()
#         c1.metric("Last Close",       f"₹{cp.iloc[-1]:.2f}")
#         c2.metric("Next-Day Pred",    f"₹{y_next[-1]:.2f}")
#         c3.metric("Horizon End Pred", f"₹{future_preds[-1]:.2f}")
#         c4.metric("Total Predictions",f"{len(future_preds)} days")

#         # plot
#         fig = go.Figure([
#             go.Scatter(x=dates,        y=y_true,      name="Actual"),
#             go.Scatter(x=dates,        y=y_next,      name="Next-Day", line=dict(dash="dash")),
#             go.Scatter(x=future_dates, y=future_preds,name="Future",   line=dict(color="green", dash="dot")),
#         ])
#         fig.update_layout(title="Actual & Predicted Prices", template="plotly_white")
#         st.plotly_chart(fig, use_container_width=True)

#         # table
#         table = pd.DataFrame({
#             "Date": future_dates.date,
#             "Predicted Close": [f"₹{x:.2f}" for x in future_preds]
#         })
#         st.subheader("📋 Future Predictions")
#         st.table(table)

# # --- ANALYTICS ---
# elif selected == "Analytics":
#     st.title("📊 Analytics")
#     st.markdown("""
#     **Technical Indicator Explorer**  
#     - Overlay Close price with SMA, EMA, RSI, MACD, Bollinger Bands  
#     - Interactive sliders to adjust window lengths  

#     **Model Diagnostics**  
#     - Training vs. validation loss & MAE curves  
#     - Residual error histograms & KDE  

#     **Scenario Testing**  
#     - What-if sliders on key features  
#     - Batch CSV upload for bulk forecasts
#     """)

# # --- ABOUT ---
# elif selected == "About":
#     st.title("ℹ️ About StockPredict")
#     st.markdown("""
# **StockPredict** is a hybrid LSTM + Dense neural network wrapped in a sleek dashboard.

# - **Data Engineering:** `yfinance` + `pandas` for OHLC and indicators (SMA/EMA/RSI/MACD/Bollinger).  
# - **Model:**  
#   - LSTM branch for temporal patterns  
#   - Dense branch for snapshot features  
#   - Merged, normalized layers for final prediction  
# - **Features:**  
#   - Next‐day & multi‐day forecasts (up to 15 days)  
#   - Interactive controls, metrics cards, rich Plotly visuals  
#   - Analytics & scenario‐testing modules  
#     """)

