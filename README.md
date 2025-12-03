# Credit Suisse Risk Appetite Index (CS GRAI) Calculator

This project calculates and visualizes the **Credit Suisse Risk Appetite Index (CS GRAI)**, analyzing market Panic and Euphoria phases based on historical data.

## 📋 Features

- **Data Collection**: Automatically collects data for various asset classes including global equities, bonds, and commodities using `yfinance`.
- **GRAI Calculation**: Calculates the Risk Appetite Index through regression analysis between Risk (Volatility) and Return of assets.
- **Visualization**: Visualizes the calculated index by normalizing it as a Z-Score and intuitively shows market conditions through reference lines (±2.0).
- **Event Backtesting**: Identifies Panic and Euphoria phases and analyzes returns (1 week, 1 month, 3 months, etc.) following these events, saving the results as a CSV file.

## 🌍 Asset Universe

The index is constructed using the following assets:

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
- **Precious Metals**: GLD (Gold), SLV (Silver)
- **Energy & Others**: USO (Oil), CPER (Copper), DBC (Commodity Index)
- **Real Estate**: VNQ (US REITs)

*(Benchmark: ^GSPC - S&P 500)*

## 🛠️ Installation

Python 3.x is required to run this project. Install the necessary libraries using the command below:

```bash
pip install -r requirements.txt
```

### Required Libraries
- `yfinance`: Financial data download
- `pandas`: Data processing and analysis
- `numpy`: Numerical calculations
- `scikit-learn`: Linear regression analysis
- `matplotlib`: Chart visualization

## 🚀 Usage

Run the command below in your terminal to proceed with data download, index calculation, visualization, and analysis sequentially.

```bash
python csrai.py
```

### Execution Process
1. **Data Download**: Fetches data for the defined asset universe (Developed/Emerging market equities & bonds, commodities, etc.).
2. **Index Calculation**: Calculates GRAI by analyzing the daily Risk-Return relationship.
3. **Chart Output**: A graph visualizing the CS GRAI index and S&P 500 index will pop up.
4. **Result Analysis**: Analyzes the timing of events (Panic/Euphoria) and subsequent returns, printing them to the console and saving them as `GRAI_Event_Analysis_YYYYMMDD.csv`.

## 📊 Output

- **Chart**: A graph showing the trend of Risk Appetite and market phases (Panic/Euphoria).
- **CSV File**: Detailed analysis data for each event occurrence.

## ⚠️ Notes

- Initial execution may take some time for data download.
- Depending on the status of `yfinance`, data for some tickers may be missing.
