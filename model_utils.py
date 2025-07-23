# # model_utils.py
# import joblib
# import numpy as np
# import streamlit as st
# import tensorflow
# # from tensorflow.keras.models import load_model

# @st.cache_resource
# def load_resources():
#     # Load without re-compiling, then compile manually
#     model = tensorflow.keras.models.load_model("hybrid_model.h5", compile=False)
#     model.compile(optimizer="adam", loss="mse", metrics=["mae"])
#     scaler = joblib.load("scaler.pkl")
#     return model, scaler

# def create_sequences(df, lookback=20):
#     df = df.copy().sort_index()
#     df["Target"] = df["Close"].shift(-1)
#     df.dropna(inplace=True)
#     feature_cols = [c for c in df.columns if c != "Target"]
#     X_seq, X_last, y, dates = [], [], [], []
#     for i in range(lookback, len(df)):
#         window = df.iloc[i - lookback : i]
#         X_seq.append(window[feature_cols].values)
#         X_last.append(window[feature_cols].values[-1])
#         y.append(df["Target"].iloc[i])
#         dates.append(df.index[i])
#     return (
#         np.array(X_seq),
#         np.array(X_last),
#         np.array(y),
#         np.array(dates)
#     )

# def predict(model, scaler, X_seq, X_last):
#     # Scale both branches
#     n_samples, lookback, n_f = X_seq.shape
#     flat = X_seq.reshape(n_samples * lookback, n_f)
#     flat_scaled = scaler.transform(flat)
#     X_seq_scaled = flat_scaled.reshape(n_samples, lookback, n_f)
#     X_last_scaled = scaler.transform(X_last)
#     preds = model.predict([X_seq_scaled, X_last_scaled], verbose=0)
#     return preds.flatten()

# model_utils.py
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("hybrid_model.h5", compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def create_sequences(df, lookback: int = 20):
    df = df.sort_index().copy()
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    feature_cols = [c for c in df.columns if c != "Target"]

    X_seq, X_last, y_true, dates = [], [], [], []
    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback : i]
        X_seq.append(window[feature_cols].values)
        X_last.append(window[feature_cols].values[-1])
        y_true.append(df["Target"].iloc[i])
        dates.append(df.index[i])

    return (
        np.array(X_seq, dtype=float),
        np.array(X_last, dtype=float),
        np.array(y_true, dtype=float),
        np.array(dates),
    )

def predict(model, scaler, X_seq, X_last):
    n_samples, lookback, n_f = X_seq.shape
    flat = X_seq.reshape(n_samples * lookback, n_f)
    flat_scaled = scaler.transform(flat)
    X_seq_scaled = flat_scaled.reshape(n_samples, lookback, n_f)
    X_last_scaled = scaler.transform(X_last)
    return model.predict([X_seq_scaled, X_last_scaled], verbose=0).flatten()

def multi_day_predict(model, scaler, seed_seq, seed_last, horizon: int):
    preds = []
    seq, last = seed_seq.copy(), seed_last.copy()
    for _ in range(horizon):
        last_scaled = scaler.transform(last.reshape(1, -1))
        seq_scaled = scaler.transform(seq).reshape(1, *seq.shape)
        p = model.predict([seq_scaled, last_scaled], verbose=0)[0, 0]
        preds.append(p)

        # Roll window: drop oldest, append new day with only 'Close' updated
        new_feat = last.copy()
        new_feat[0] = p  # assuming 'Close' is the first feature
        seq = np.vstack([seq[1:], new_feat.reshape(1, -1)])
        last = new_feat
    return np.array(preds, dtype=float)


# # model_utils.py
# import joblib
# import numpy as np
# import streamlit as st
# import tensorflow as tf
# import pandas as pd

# @st.cache_resource
# def load_resources():
#     model = tf.keras.models.load_model("hybrid_model.h5", compile=False)
#     model.compile(optimizer="adam", loss="mse", metrics=["mae"])
#     scaler = joblib.load("scaler.pkl")
#     return model, scaler

# def create_sequences(df, lookback: int = 20):
#     if isinstance(df, np.ndarray):
#         df = pd.DataFrame(df)

#     df = df.sort_index().copy()
#     df["Target"] = df["Close"].shift(-1)
#     df.dropna(inplace=True)
#     feature_cols = [c for c in df.columns if c != "Target"]

#     X_seq, X_last, y_true, dates = [], [], [], []
#     for i in range(lookback, len(df)):
#         window = df.iloc[i - lookback : i]
#         X_seq.append(window[feature_cols].values)
#         X_last.append(window[feature_cols].values[-1])
#         y_true.append(df["Target"].iloc[i])
#         dates.append(df.index[i])

#     return (
#         np.array(X_seq, dtype=float),
#         np.array(X_last, dtype=float),
#         np.array(y_true, dtype=float),
#         np.array(dates),
#     )

# def predict(model, scaler, X_seq, X_last):
#     n_samples, lookback, n_f = X_seq.shape
#     flat = X_seq.reshape(n_samples * lookback, n_f)
#     flat_scaled = scaler.transform(flat)
#     X_seq_scaled = flat_scaled.reshape(n_samples, lookback, n_f)
#     X_last_scaled = scaler.transform(X_last)
#     return model.predict([X_seq_scaled, X_last_scaled], verbose=0).flatten()

# def multi_day_predict(model, scaler, seed_seq, seed_last, horizon: int):
#     preds = []
#     seq, last = seed_seq.copy(), seed_last.copy()
#     for _ in range(horizon):
#         last_scaled = scaler.transform(last.reshape(1, -1))
#         seq_scaled = scaler.transform(seq).reshape(1, *seq.shape)
#         p = model.predict([seq_scaled, last_scaled], verbose=0)[0, 0]
#         preds.append(p)

#         # Roll window: drop oldest, append new day with only 'Close' updated
#         new_feat = last.copy()
#         new_feat[0] = p  # assuming 'Close' is the first feature
#         seq = np.vstack([seq[1:], new_feat.reshape(1, -1)])
#         last = new_feat
#     return np.array(preds, dtype=float)


