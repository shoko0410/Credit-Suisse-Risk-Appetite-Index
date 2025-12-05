import yfinance as yf
import pandas as pd

COMMODITIES = [
    'GC=F',   # Gold Futures
    'SI=F',   # Silver Futures
    'CL=F',   # WTI Crude Oil Futures
    'HG=F',   # Copper Futures
    'NG=F',   # Natural Gas Futures
    'VNQ'     # Vanguard Real Estate ETF
]

print(f"Testing data fetch for: {COMMODITIES}")

try:
    # Using the same parameters as in csgrai.py
    df = yf.download(COMMODITIES, start="2023-01-01", progress=False, auto_adjust=False)
    
    print("\nDownload complete.")
    print(f"Data shape: {df.shape}")
    
    if df.empty:
        print("ERROR: Downloaded DataFrame is empty.")
    else:
        print("\nColumns:")
        print(df.columns)
        
        # Check if we have data for all tickers
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance returns MultiIndex columns if multiple tickers are downloaded
            # Level 0 is usually 'Price' (Adj Close, Close, etc.), Level 1 is Ticker
            # But with auto_adjust=False, it might be different.
            # Let's inspect the 'Adj Close' or 'Close' level.
            
            if 'Adj Close' in df.columns.get_level_values(0):
                data = df['Adj Close']
                print("\n'Adj Close' data found.")
            elif 'Close' in df.columns.get_level_values(0):
                data = df['Close']
                print("\n'Close' data found (Adj Close not available).")
            else:
                print("\nWARNING: Neither 'Adj Close' nor 'Close' found in top level columns.")
                data = df
                
            print("\nMissing values per ticker:")
            print(data.isnull().sum())
            
            print("\nFirst 5 rows:")
            print(data.head())
            
            print("\nLast 5 rows:")
            print(data.tail())
            
        else:
            # Single ticker or flat index (less likely with multiple tickers)
            print("\nFlat column index (unexpected for multiple tickers).")
            print(df.head())

except Exception as e:
    print(f"\nAn error occurred: {e}")
