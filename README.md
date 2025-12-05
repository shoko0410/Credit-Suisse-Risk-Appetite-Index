# Credit Suisse Global Risk Appetite Index (CS GRAI) Calculator

This project calculates and visualizes a **Robust, Institutional-Grade Credit Suisse Global Risk Appetite Index (CS GRAI)**. It analyzes market Panic and Euphoria phases based on historical data, incorporating advanced quantitative methodologies to ensure accuracy and stability.

## 🚀 Key Features (Refactored)

- **Excess Return Calculation**: Incorporates the Risk-Free Rate (`^IRX`) to measure *real* risk premiums, crucial for high-interest environments.
- **Dynamic Universe Selection**: Eliminates **Survivorship Bias** by allowing assets to enter the index dynamically as data becomes available, rather than requiring a full history.
- **Market Price of Risk (Beta)**: Calculates the regression slope (Beta) of Return vs. Risk to directly measure the market's price of risk, consistent with the Core CS GRAI methodology.
- **Risk Standardization (Z-Score)**: Applies Z-Score normalization to the final index to ensure comparable signals across different market regimes.
- **Robust Outlier Cleaning**: Automatically filters out unrealistic daily price spikes (>50%) to maintain data integrity.
- **Optimized Performance**: Replaces `sklearn` with vectorized `numpy` operations for lightning-fast calculation.
- **Official Methodology**: Aligned with Credit Suisse's official whitepaper specifications (126-day Return / 252-day Volatility windows) to prevent overfitting.

## 🌍 Asset Universe

The index is constructed using a diverse set of global assets:

### Developed Market Equities
- **US**: SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000)
- **Europe & Others**: EWJ (Japan), EWG (Germany), EWU (UK), EWQ (France), EWL (Switzerland), EWC (Canada), EWA (Australia), EWD (Sweden), EWH (Hong Kong), EWS (Singapore)

### Developed Market Bonds
- SHY (1-3 Year Treasury), IEF (7-10 Year Treasury), TLT (20+ Year Treasury)
- BWX (Intl Treasury ex-US), IGOV (Intl Treasury)

### Emerging Market Equities
- EEM (Emerging Markets), FXI (China Large-Cap), EWY (South Korea), EWT (Taiwan)
- INDA (India), EWZ (Brazil), EWW (Mexico), EZA (South Africa), TUR (Turkey)

### Emerging Market Bonds
- EMB (USD Emerging Markets Bond), EMLC (Local Currency Emerging Markets Bond)

### Commodities & REITs
- **Precious Metals**: GC=F (Gold Futures), SI=F (Silver Futures)
- **Energy & Others**: CL=F (WTI Crude Oil), HG=F (Copper), NG=F (Natural Gas)
- **Real Estate**: VNQ (US REITs)

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
