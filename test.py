import yfinance as yf
import pandas as pd

start = "2024-11-01"
# bump end by one day so it’s inclusive
end   = (pd.to_datetime("2025-03-22") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

print("Requesting:", start, "→", end)
df = yf.download("RELIANCE.NS",
                 start=start,
                 end=end,
                 progress=False,
                 auto_adjust=False)
print("Shape:", df.shape)
print(df.head(), "\n…\n", df.tail())
