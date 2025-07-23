
# data_utils.py
import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime

@st.cache_data(ttl=3600)
def load_and_preprocess(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
      st.error(f"No data for symbol {ticker!r} in {start}–{end}.")
      st.stop()

    # Reuse the same feature engineering:
    # SMA, EMA, RSI, MACD, Bollinger Bands
    df["SMA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["Diff_SMA"] = df["SMA_5"] - df["SMA_20"]
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + rs))
    ema_fast = df["Close"].ewm(span=12).mean()
    ema_slow = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["STD_20"] = df["Close"].rolling(20).std()
    df["Upper_BB"] = df["SMA_20"] + 2 * df["STD_20"]
    df["Lower_BB"] = df["SMA_20"] - 2 * df["STD_20"]
    df.dropna(inplace=True)
    return df

# # data_utils.py
# import yfinance as yf
# import pandas as pd
# import streamlit as st
# from datetime import datetime,timedelta

# @st.cache_data(ttl=3600)
# def load_and_preprocess(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
#     # 1) Convert to strings
#     start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
#     # 2) Bump end by one day so it's inclusive
#     end_str   = (pd.to_datetime(end) + timedelta(days=1)).strftime("%Y-%m-%d")

#     # 3) Download raw
#     df = yf.download(ticker,
#                      start=start_str,
#                      end=end_str,
#                      progress=False,
#                      auto_adjust=False)
#     # Immediately bail if empty
#     if df.empty:
#         return df
    

#     # Compute indicators with min_periods=1 so we don't drop everything immediately
#     df["SMA_5"]    = df["Close"].rolling(5,  min_periods=1).mean()
#     df["SMA_20"]   = df["Close"].rolling(20, min_periods=1).mean()
#     df["EMA_5"]    = df["Close"].ewm(span=5,  adjust=False).mean()
#     df["EMA_20"]   = df["Close"].ewm(span=20, adjust=False).mean()
#     df["Diff_SMA"] = df["SMA_5"] - df["SMA_20"]

#     delta = df["Close"].diff()
#     gain  = delta.clip(lower=0)
#     loss  = -delta.clip(upper=0)
#     rs    = gain.rolling(14, min_periods=1).mean() / loss.rolling(14, min_periods=1).mean()
#     df["RSI_14"] = 100 - (100 / (1 + rs))

#     macd_line = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
#     df["MACD"]        = macd_line
#     df["MACD_Signal"] = macd_line.ewm(span=9).mean()

#     df["STD_20"]   = df["Close"].rolling(20, min_periods=1).std()
#     df["Upper_BB"] = df["SMA_20"] + 2 * df["STD_20"]
#     df["Lower_BB"] = df["SMA_20"] - 2 * df["STD_20"]

#     df.dropna(inplace=True)
#     return df

# #data_utils.py

# import yfinance as yf
# import pandas as pd
# import streamlit as st
# import joblib
# from datetime import datetime,timedelta 

# @st.cache_data(ttl=3600)
# def load_and_preprocess(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
#     # start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
#     # end_str   = (pd.to_datetime(end) + timedelta(days=1)).strftime("%Y-%m-%d")

#     # df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=False)

#     feature_columns = [
#         "Open", "High", "Low", "Close", "Adj Close", "Volume",
#         "SMA_5", "SMA_20", "EMA_5", "EMA_20", "Diff_SMA",
#         "RSI_14", "MACD", "MACD_Signal", "STD_20", "Upper_BB", "Lower_BB"
#     ]

#     start = "2024-11-01"
# # bump end by one day so it’s inclusive
#     end   = (pd.to_datetime("2025-03-22") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

#     print("Requesting:", start, "→", end)
#     df = yf.download("RELIANCE.NS",
#                  start=start,
#                  end=end,
#                  progress=False,
#                  auto_adjust=False)

#     if df.empty:
#         st.error(f"No data for {ticker!r} between {start} and {end}.")
#         st.stop()

#     df["SMA_5"] = df["Close"].rolling(5, min_periods=1).mean()
#     df["SMA_20"] = df["Close"].rolling(20, min_periods=1).mean()
#     df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
#     df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
#     df["Diff_SMA"] = df["SMA_5"] - df["SMA_20"]

#     delta = df["Close"].diff()
#     gain = delta.clip(lower=0)
#     loss = -delta.clip(upper=0)
#     rs = gain.rolling(14, min_periods=1).mean() / loss.rolling(14, min_periods=1).mean()
#     df["RSI_14"] = 100 - (100 / (1 + rs))

#     ema_fast = df["Close"].ewm(span=12).mean()
#     ema_slow = df["Close"].ewm(span=26).mean()
#     df["MACD"] = ema_fast - ema_slow
#     df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

#     df["STD_20"] = df["Close"].rolling(20, min_periods=1).std()
#     df["Upper_BB"] = df["SMA_20"] + 2 * df["STD_20"]
#     df["Lower_BB"] = df["SMA_20"] - 2 * df["STD_20"]

#     df.dropna(inplace=True)

#     # keep raw for display & slicing
#     raw_df = df[feature_columns].copy()

#     # scale
#     scaler = joblib.load("scaler.pkl")
#     arr = scaler.transform(raw_df)
#     df_scaled = pd.DataFrame(arr, index=raw_df.index, columns=feature_columns)

#     return raw_df, df_scaled


