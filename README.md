# Credit Suisse Global Risk Appetite Index (CS GRAI) Calculator

This project calculates and visualizes a **Credit Suisse Global Risk Appetite Index (CS GRAI)**. It analyzes market Panic and Euphoria phases based on historical data, incorporating advanced quantitative methodologies to ensure accuracy and stability.

> **Note**: This version includes a hybrid data approach for Bitcoin, using long-term `BTC-USD` price data but **masking it prior to Jan 10, 2024** (Launch of IBIT ETF). This allows the index to reflect the modern inclusion of crypto assets without introducing historical bias before they were accessible via major ETFs.

## 🚀 Key Features

- **Excess Return per Unit of Risk**: Calculates Excess Return by strictly matching the duration of the Risk-Free Rate (6-month accumulated) with asset returns, ensuring precise "Duration Matching" as per CS Whitepaper.
- **Institutional-Grade Data (1990~)**: Replaces short-history ETFs with original Global Indices and long-standing Vanguard Mutual Funds (`VUSTX`, `VWEHX`).
- **Real-Time FX Adjustment**: Automatically fetches currency data (JPY, GBP, EUR, etc.) and converts all local indices (Nikkei 225, FTSE 100) into **USD-denominated returns** for accurate cross-asset comparison.
- **Dynamic Universe Selection**: Eliminates **Survivorship Bias** by allowing assets to enter the index naturally as they get listed.
- **Market Price of Risk (Beta)**: Calculates the regression slope (Beta) of Return vs. Risk to directly measure the market's price of risk, consistent with the Core CS GRAI methodology.
- **Adaptive Normalization (5-Year)**: Uses a **5-Year Rolling Window** for Z-Score normalization to fix **Upward Bias**. By re-centering the index against the recent business cycle, it accurately detects Panic/Euphoria even during prolonged secular bull markets.
- **Robust Outlier Cleaning**: Automatically filters out unrealistic daily price spikes (>50%) to maintain data integrity.
- **Optimized Performance**: Replaces `sklearn` with vectorized `numpy` operations for lightning-fast calculation.
- **Official Methodology**: Aligned with Credit Suisse's official whitepaper specifications (126-day Return / 252-day Volatility windows) to prevent overfitting.

## 🌍 Asset Universe

The index is constructed using a diverse set of global assets:

### Global Indices (FX Adjusted to USD)
- **Developed Majors**: ^GSPC, ^IXIC, ^RUT (US), ^N225 (JP), ^GDAXI (DE), ^FTSE (UK), ^FCHI (FR), ^AORD (AU)
- **Developed Minors**: ^SSMI (Swiss), ^GSPTSE (Canada), ^STI (Singapore), ^OMX (Sweden) - *New*
- **Emerging**: ^KS11 (Korea), ^TWII (Taiwan), ^BVSP (Brazil), ^HSI (Hong Kong/China)

### Fixed Income Funds (Vanguard + Global)
- **Credit Signals**: VWEHX (High Yield - Crisis Indicator), VWESX (Inv Grade)
- **Rates**: VUSTX (Long-Term), VFISX (Short-Term)
- **International**: RPIBX (Intl Bond Fund - 1986~) - *New*
- **EM Debt**: FNMIX (Fidelity EM), PREMX (T. Rowe Price EM)

### Sectors (Cyclical) - *New*
- **Technology**: ^SOX (Semiconductors)
- **Financials**: XLF (Banking Risk)
- **Discretionary**: XLY (Consumer Sentiment)
- **Real Estate**: VGSIX (REITs)

### Commodities & Macro
- **Precious Metals**: GC=F (Gold), SI=F (Silver)
- **Cyclical**: CL=F (WTI Oil), HG=F (Copper)
- **Currencies**: DX-Y.NYB (Dollar Index), AUDUSD=X (Aussie Dollar)

### Emerging Market ETFs (Strategic Detail)
- **Specific Risks**: TUR (Turkey), INDA (India), EZA (South Africa), EWW (Mexico)
46: 
47: ### Crypto (Digital Assets) - *New*
48: - **Bitcoin**: BTC-USD (Masked pre-2024 to simulate ETF launch)

*(Benchmark: ^GSPC - S&P 500)*
*(Risk-Free: ^IRX - 13 Week Treasury Bill)*

## 🛠️ Installation

Python 3.x is required. Install the necessary libraries:

```bash
pip install -r requirements.txt
```

### Required Libraries
- `yfinance`: Financial data download
- `pandas`: Data processing and analysis
- `numpy`: Numerical calculations & Vectorized regression
- `matplotlib`: Chart visualization
*(Note: `scikit-learn` is no longer required due to numpy optimization)*

## 🚀 Usage

Run the script to download data, calculate the index, and visualize results:

```bash
python csgrai.py
```

### Execution Process
1.  **Data Download**: Fetches data for the defined asset universe.
2.  **Data Cleaning**: Applies outlier filtering (>50% spikes) and dynamic universe adjustments.
3.  **Index Calculation**:
    *   Calculates 6-month Excess Returns and 12-month Volatility.
    *   Computes the regression slope (Risk Appetite).
    *   Standardizes the Risk Appetite Index (Z-Score).
4.  **Visualization**: Displays the CS GRAI Z-Score chart with Panic/Euphoria zones.
5.  **Event Analysis**: Identifies market regimes and calculates forward returns, saving results to `GRAI_Event_Analysis_YYYYMMDD.csv`.

## 📊 Output

- **Chart**: A graph showing the trend of Risk Appetite and market phases (Panic/Euphoria).
- **CSV File**: Detailed analysis data for each event occurrence.

## ⚠️ Methodology Notes

- **Time Windows**: Uses 126-day (6-month) return and 252-day (12-month) volatility windows for long-term robustness.
- **Thresholds**: ±2.0 Z-Score indicates extreme Panic (Buy signal) or Euphoria (Sell signal).
