import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional, Union

# ==============================================================================
# 0. 설정 및 상수 (Configuration) - "The Kitchen Sink"

# ==============================================================================
RET_WINDOW = 126  # 6개월
VOL_WINDOW = 252  # 12개월
MIN_ASSETS = 15   
THRESHOLD = 2.0   

# ------------------------------------------------------------------------------
# 1. 환율 매핑 (FX Mapping) - 전 세계 자산을 달러(USD) 기준으로 통일
# ------------------------------------------------------------------------------
FX_MAPPING = {
    # [선진국]
    '^N225':  ('JPY=X', 'divide'),    # 일본 (엔화)
    '^FTSE':  ('GBPUSD=X', 'multiply'), # 영국 (파운드)
    '^GDAXI': ('EURUSD=X', 'multiply'), # 독일 (유로)
    '^FCHI':  ('EURUSD=X', 'multiply'), # 프랑스 (유로)
    '^AORD':  ('AUDUSD=X', 'multiply'), # 호주 (호주달러)
    '^SSMI':   ('CHF=X', 'divide'),    # 스위스 (프랑)
    '^GSPTSE': ('CAD=X', 'divide'),    # 캐나다 (달러)
    '^STI':    ('SGD=X', 'divide'),    # 싱가포르 (달러)
    '^OMX':    ('SEK=X', 'divide'),    # 스웨덴 (크로나)
    
    # [신흥국] 
    '^HSI':   ('HKD=X', 'divide'),    # 홍콩/중국 (홍콩달러)
    '^KS11':  ('KRW=X', 'divide'),    # 한국 (원화)
    '^TWII':  ('TWD=X', 'divide'),    # 대만 (대만달러)
    '^BVSP':  ('BRL=X', 'divide')     # 브라질 (헤알화)
}

# ------------------------------------------------------------------------------
# 2. 자산 유니버스 (Asset Universe) - All-In Strategy
# ------------------------------------------------------------------------------

# [1] Global Indices (역사적 데이터 확보용 원본 지수)
MAJOR_INDICES = [
    '^GSPC', '^IXIC', '^RUT', # 미국 (S&P500, 나스닥, 러셀2000-소형주)
    '^N225', '^GDAXI', '^FTSE', '^FCHI', '^AORD', # Major DM
    '^SSMI', '^GSPTSE', '^STI', '^OMX', # Minor DM (스위스, 캐나다, 싱가포르, 스웨덴)
    '^HSI', '^KS11', '^TWII', '^BVSP' # Major EM
]

# [2] Bond & Credit Funds (채권 & 신용 - 뮤추얼 펀드 사용)
FIXED_INCOME = [
    'VUSTX', # [장기국채] Vanguard Long-Term Treasury (1986~)
    'VFISX', # [단기국채] Vanguard Short-Term Treasury (1991~)
    'VWESX', # [우량회사채] Vanguard Long-Term Inv Grade (1973~)
    'VWEHX', # [하이일드] Vanguard High-Yield Corp (1978~) 
    'RPIBX', # [해외국채] T. Rowe Price Intl Bond (1986~)
    'FNMIX', # [이머징채권] Fidelity New Markets Income (1993~)
    'PREMX'  # [이머징채권2] T. Rowe Price EM Bond (1994~)
]

# [3] Sectors & Real Assets (경기 민감형 자산)
# 반도체 지수 + 리츠 펀드 + 주요 경기 민감 ETF(금융, 소비재)
SECTORS = [
    '^SOX',   # [반도체] 필라델피아 반도체 지수 (1994~)
    'VGSIX',  # [부동산] Vanguard Real Estate Fund (1996~)
    'XLF',    # [금융] Financial Select Sector SPDR (1998~)
    'XLY'     # [소비재] Consumer Discretionary SPDR (1998~)
]

# [4] Commodities & Currencies (원자재 & 통화)
MACRO = [
    'GC=F', 'SI=F', # 금, 은 (인플레 헤지)
    'CL=F', 'HG=F', # 원유, 구리 (실물 경기)
    'DX-Y.NYB',     # 달러 인덱스 (안전자산)
    'AUDUSD=X'      # 호주 달러 (위험자산 통화)
]

# [5] Emerging Markets ETFs (지수로 커버 안 되는 나머지 국가들)
# 터키, 인도, 멕시코, 남아공 등 '약한 고리' 감지용
EM_SPECIFIC = [
    'EEM',  # 이머징 전체 (2003~)
    'INDA', # 인도 (2012~)
    'TUR',  # 터키 (2008~)
    'EWW',  # 멕시코 (1996~)
    'EZA'   # 남아공 (2003~)
]

# [6] Crypto (Digital Assets)
CRYPTO = [
    'BTC-USD'  # [비트코인] Bitcoin (Using underlying asset but masked as ETF)
]

# [6] Benchmarks
BENCHMARK = ['^GSPC']
RISK_FREE = ['^IRX']

TICKERS = (
    MAJOR_INDICES + 
    FIXED_INCOME + 
    SECTORS + 
    MACRO + 
    EM_SPECIFIC + 
    CRYPTO +
    RISK_FREE
)

# 중복 제거
UNIQUE_TICKERS = list(set(TICKERS))

# ==============================================================================
# 1. 데이터 수집 함수
# ==============================================================================
def clean_outliers(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """일간 수익률이 threshold(50%)를 초과하는 이상치 제거 (Data Cleaning)"""
    print("  [Debug] clean_outliers 시작", flush=True)
    # 수익률 계산
    returns = df.pct_change(fill_method=None)
    
    # 1. 일반 자산: 이상치 마스크 생성 (절대값 50% 이상 변동)
    mask = returns.abs() > threshold
    
    # 2. ^IRX (금리) 예외 처리: 수익률(%)이 아닌 절대 변화폭(bp) 기준
    # 금리가 0.1% -> 0.2% 되면 수익률은 100%지만, 실제 변화는 10bp에 불과함.
    # 따라서 IRX는 전일 대비 1.0%p (100bp) 이상 급변했을 때만 이상치로 간주
    if '^IRX' in df.columns:
        irx_diff = df['^IRX'].diff().abs()
        irx_mask = irx_diff > 1.0  # 100bp
        
        # 기존 mask에서 IRX 부분 덮어쓰기
        mask['^IRX'] = irx_mask

    # 3. Whitelist 처리 (역사적 사실 보존)
    # 특정 날짜의 극단적 변동이 실제 사건인 경우 이상치에서 제외
    whitelist = {
        'CL=F': ['2020-04-20', '2020-04-21'] # 마이너스 유가 사태
    }
    
    # Whitelist 적용
    for asset, dates in whitelist.items():
        if asset in df.columns:
            for d_str in dates:
                try:
                    d = pd.Timestamp(d_str)
                    if d in mask.index and mask.loc[d, asset]:
                        mask.loc[d, asset] = False
                        # 보존 로그 출력
                        val = df.loc[d, asset]
                        print(f"  [Whitelisted] Date: {d_str} | Asset: {asset} | Val: {val:.2f} (Historical Event Confirmed)")
                except Exception as e:
                    print(f"  [Warning] Whitelist check failed for {asset} on {d_str}: {e}")

    outlier_count = mask.sum().sum()
    
    if outlier_count > 0:
        print(f"데이터 정제: 총 {outlier_count}개의 이상치(일반 >50%, 금리 >100bp)를 감지하여 제거합니다.", flush=True)
        
        # 상세 내역 출력
        stack = mask.stack()
        outliers_indices = stack[stack].index
        for date, asset in outliers_indices:
            if asset == '^IRX':
                # 금리는 절대 변화폭 출력
                diff_val = df['^IRX'].diff().loc[date]
                print(f"  [Outlier Removed] Date: {date.date()} | Asset: {asset} | Diff: {diff_val:+.2f} bp (Price: {df.loc[date, asset]:.2f})")
            else:
                # 일반 자산은 수익률 출력
                ret_val = returns.loc[date, asset]
                print(f"  [Outlier Removed] Date: {date.date()} | Asset: {asset} | Return: {ret_val*100:.2f}%")

        # 이상치를 NaN으로 처리하고 이전 값으로 채움
        df_clean = df.copy()
        df_clean[mask] = np.nan
        df_clean = df_clean.ffill()
        print("  [Debug] clean_outliers 완료 (이상치 제거됨)", flush=True)
        return df_clean
    
    print("  [Debug] clean_outliers 완료 (이상치 없음)", flush=True)
    return df

def fetch_data(ticker_list: list, start_date: str = "1990-01-01") -> pd.DataFrame:
    """
    데이터 다운로드 및 환율 보정 (USD Adjustment) 수행
    """
    print("데이터 다운로드 및 통화 보정(USD Adjustment) 시작...", flush=True)
    
    # 1. 환율 데이터에 필요한 티커들 추출
    fx_tickers = set()
    for idx, (fx, op) in FX_MAPPING.items():
        if idx in ticker_list:
            fx_tickers.add(fx)
    
    # 2. 전체 데이터 다운로드 (자산 + 환율)
    all_tickers = list(set(ticker_list) | fx_tickers)
    
    try:
        # auto_adjust=False로 원본 가격 확보 (수정주가 사용을 위함)
        df = yf.download(all_tickers, start=start_date, progress=False, auto_adjust=False)
        
        # MultiIndex 처리 ('Adj Close' 우선 사용)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                data = df['Adj Close']
            except KeyError:
                data = df['Close']
        else:
            data = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
            
    except Exception as e:
        print(f"다운로드 오류: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    # 3. 환율 보정 (Currency Conversion)
    adjusted_data = data.copy()
    print("  [Processing] 현지 통화 자산을 USD로 변환 중...", flush=True)
    
    for asset in ticker_list:
        # FX 매핑이 존재하는 자산인지 확인
        if asset in FX_MAPPING and asset in data.columns:
            fx_ticker, operation = FX_MAPPING[asset]
            
            if fx_ticker in data.columns:
                # 환율 데이터 결측치 보정 (휴일 등으로 비는 경우 전일 값 사용)
                fx_series = data[fx_ticker].ffill()
                
                if operation == 'divide':
                    # 예: Nikkei(JPY) / (JPY/USD) = Nikkei(USD)
                    adjusted_data[asset] = data[asset] / fx_series
                    # print(f"    - {asset}: JPY -> USD 변환 (Divide)")
                elif operation == 'multiply':
                    # 예: FTSE(GBP) * (USD/GBP) = FTSE(USD)
                    adjusted_data[asset] = data[asset] * fx_series
                    # print(f"    - {asset}: Local -> USD 변환 (Multiply)")
            else:
                print(f"    [Warning] 환율 데이터({fx_ticker}) 누락으로 {asset} 변환 실패.")

    # 4. 분석에 필요한 자산만 남기고 환율 티커 제거
    final_cols = [t for t in ticker_list if t in adjusted_data.columns]
    final_data = adjusted_data[final_cols]
    
    # 5. 전처리 (이상치 제거 및 결측치 처리)
    final_data = clean_outliers(final_data)
    final_data = final_data.dropna(axis=0, how='all').ffill()
    
    # S&P500 데이터가 시작되는 시점부터 자름 (너무 먼 과거 NaN 방지)
    if '^GSPC' in final_data.columns:
        first_valid = final_data['^GSPC'].first_valid_index()
        if first_valid:
            final_data = final_data.loc[first_valid:]
            
    print(f"데이터 준비 완료: {final_data.shape[1]}개 자산 (USD 기준 통일)")
    return final_data

# ==============================================================================
# 2. CS GRAI 계산 엔진
# ==============================================================================
def calculate_cs_grai(price_data: pd.DataFrame) -> pd.Series:
    grai_results = {}
    
    # 자산군과 무위험자산 분리
    rf_ticker = '^IRX'
    if rf_ticker not in price_data.columns:
        print("경고: 무위험자산(^IRX) 데이터가 없습니다. 무위험 수익률을 0으로 가정합니다.")
        rf_series = pd.Series(0, index=price_data.index)
    else:
        # ^IRX는 연율(%)로 제공되므로 일간 수익률로 변환 필요 (근사치: 연율/252)
        rf_series = price_data[rf_ticker] / 100 / 252
        rf_series = rf_series.ffill() # [추가] 금리 데이터가 빈 날은 전날 금리 사용
        
    valid_assets = [t for t in price_data.columns if t != '^GSPC' and t != rf_ticker]

    # [2] Pre-calculate Risk (Volatility)
    # (Vectorized implementation)
    print("  [Step 1] Risk Matrix(Volatility) 벡터화 계산 중...", flush=True)
    # Calculate daily log returns for all assets
    log_rets_daily = np.log(price_data[valid_assets] / price_data[valid_assets].shift(1))
    
    # Rolling Volatility (Annualized)
    # DataFrame.rolling().std() computes sample standard deviation (N-1) by default, matching the previous logic.
    risk_matrix = log_rets_daily.rolling(window=VOL_WINDOW).std() * np.sqrt(252)
    
    # [3] Pre-calculate Excess Returns & Duration Matching
    # (Vectorized implementation for speed and accuracy)
    print("  [Step 2] Excess Return Matrix & Duration Matching 계산 중...", flush=True)
    
    # 6-month Returns (Discrete) -> pct_change(126) matches p_current/p_past - 1
    return_matrix = price_data[valid_assets].pct_change(periods=RET_WINDOW)
    
    # Risk-Free Rate Duration Matching (Sum over same window)
    rf_rolling = rf_series.rolling(window=RET_WINDOW).sum()
    
    # Excess Returns: Subtract RF from each asset's return
    # align axes automatically
    excess_return_matrix = return_matrix.sub(rf_rolling, axis=0)
    
    start_idx = max(RET_WINDOW, VOL_WINDOW)
    total_days = len(price_data)
    dates = price_data.index
    
    print(f"지수 산출 시작... 총 {total_days - start_idx}일 처리 예정")
    
    for i in range(start_idx, total_days):
        current_date = dates[i]
        
        # Retrieve pre-calculated values
        volatility = risk_matrix.iloc[i].copy()
        excess_returns = excess_return_matrix.iloc[i].copy()
        
        # [Critical] BTC-USD Masking (ETF Launch Simulation)
        # IBIT 상장일(2024-01-10) 이전에는 비트코인이 인덱스에 포함되지 않도록 강제 마스킹
        if 'BTC-USD' in volatility.index and current_date < pd.Timestamp('2024-01-10'):
             volatility['BTC-USD'] = np.nan
             excess_returns['BTC-USD'] = np.nan
        
        daily_df = pd.DataFrame({'Risk': volatility, 'Return': excess_returns}).dropna()
        
        if len(daily_df) < MIN_ASSETS:
            grai_results[current_date] = np.nan
            continue
            
        # 회귀분석 (Regression Analysis)
        # 수정: Correlation이 아닌 Slope(Beta)를 계산하도록 변경
        # CS GRAI는 "Market Price of Risk"를 측정해야 하므로, Risk(X)에 대한 Return(Y)의 기울기를 구함
        
        x = daily_df['Risk'].values
        y = daily_df['Return'].values
        
        if np.std(x) == 0: # Risk가 없으면 기울기 계산 불가
            grai_results[current_date] = 0
            continue

        # Linear Regression (Slope = Beta)
        # deg=1: 1차 회귀분석 (y = ax + b)
        slope, intercept = np.polyfit(x, y, 1)
        
        grai_results[current_date] = slope
        
        if i % 100 == 0:
            print(f"Processing: {current_date.date()} | Assets: {len(daily_df)} | Progress: {i}/{total_days}")

    return pd.Series(grai_results)

# ==============================================================================
# 3. 심화 백테스팅: 이벤트(Episode)별 진입/극점/회복 수익률 분석
# ==============================================================================
def get_forward_returns(date: datetime, price_series: pd.Series, periods: Dict[str, int]) -> Dict[str, float]:
    """특정 날짜 기준 미래 수익률 계산"""
    returns = {}
    try:
        current_idx = price_series.index.get_loc(date)
        current_price = price_series.iloc[current_idx]
        
        for label, days in periods.items():
            future_idx = current_idx + days
            if future_idx < len(price_series):
                future_price = price_series.iloc[future_idx]
                ret = (future_price / current_price) - 1
                returns[label] = ret
            else:
                returns[label] = np.nan
    except KeyError:
        for label in periods.keys():
            returns[label] = np.nan
    return returns

def analyze_episodes(grai_series: pd.Series, price_series: pd.Series, threshold: float = 2.0) -> pd.DataFrame:
    """
    이벤트 기반 분석:
    1. 진입 (Entry): 기준선 돌파
    2. 극점 (Extremum): 기간 중 최대/최소값
    3. 회복 (Exit): 기준선 복귀
    """
    periods = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '12M': 252}
    events = []
    
    # --- Panic Analysis (Below -threshold) ---
    in_panic = False
    panic_start_date = None
    panic_records = [] # (date, value) 저장용
    
    # --- Euphoria Analysis (Above +threshold) ---
    in_euphoria = False
    euphoria_start_date = None
    euphoria_records = []
    
    dates = grai_series.index
    
    for i in range(len(dates)):
        date = dates[i]
        val = grai_series.iloc[i]
        
        # 1. Panic Logic (침체)
        if val <= -threshold:
            if not in_panic:
                # [진입] Panic 시작
                in_panic = True
                panic_start_date = date
                panic_records = [(date, val)]
                
                # 진입 시점 기록
                row = {'Date': date, 'Type': 'Entry', 'Signal': 'Panic (Buy)', 'GRAI_Value': val}
                row.update(get_forward_returns(date, price_series, periods))
                events.append(row)
            else:
                # Panic 진행 중
                panic_records.append((date, val))
                
        elif in_panic and val > -threshold:
            # [회복] Panic 종료 (기준선 복귀)
            in_panic = False
            
            # 1) 회복(Exit) 시점 기록
            row_exit = {'Date': date, 'Type': 'Exit', 'Signal': 'Panic (Buy)', 'GRAI_Value': val}
            row_exit.update(get_forward_returns(date, price_series, periods))
            events.append(row_exit)
            
            # 2) 극점(Bottom) 시점 찾기 (지난 에피소드 중 최저점)
            if panic_records:
                min_record = min(panic_records, key=lambda x: x[1])
                min_date, min_val = min_record
                
                # 진입일과 겹치지 않을 때만 기록 (중복 방지)
                if min_date != panic_start_date and min_date != date:
                    row_min = {'Date': min_date, 'Type': 'Bottom', 'Signal': 'Panic (Buy)', 'GRAI_Value': min_val}
                    row_min.update(get_forward_returns(min_date, price_series, periods))
                    events.append(row_min)

        # 2. Euphoria Logic (과열)
        if val >= threshold:
            if not in_euphoria:
                # [진입] Euphoria 시작
                in_euphoria = True
                euphoria_start_date = date
                euphoria_records = [(date, val)]
                
                # 진입 시점 기록
                row = {'Date': date, 'Type': 'Entry', 'Signal': 'Euphoria (Sell)', 'GRAI_Value': val}
                row.update(get_forward_returns(date, price_series, periods))
                events.append(row)
            else:
                euphoria_records.append((date, val))
                
        elif in_euphoria and val < threshold:
            # [회복] Euphoria 종료
            in_euphoria = False
            
            # 1) 회복(Exit) 시점
            row_exit = {'Date': date, 'Type': 'Exit', 'Signal': 'Euphoria (Sell)', 'GRAI_Value': val}
            row_exit.update(get_forward_returns(date, price_series, periods))
            events.append(row_exit)
            
            # 2) 극점(Peak) 시점 찾기 (지난 에피소드 중 최고점)
            if euphoria_records:
                max_record = max(euphoria_records, key=lambda x: x[1])
                max_date, max_val = max_record
                
                if max_date != euphoria_start_date and max_date != date:
                    row_max = {'Date': max_date, 'Type': 'Peak', 'Signal': 'Euphoria (Sell)', 'GRAI_Value': max_val}
                    row_max.update(get_forward_returns(max_date, price_series, periods))
                    events.append(row_max)
                    
    return pd.DataFrame(events)

# ==============================================================================
# 4. 메인 실행 함수 (Main Execution)
# ==============================================================================
def main():
    print(f"총 {len(UNIQUE_TICKERS)}개의 자산으로 인덱스를 구성합니다.")
    
    # 1. 실행 (데이터 수집)
    # Dynamic Universe: 1990년부터 데이터를 받아오되, 각 자산별 상장일에 따라 데이터가 없는 기간은 NaN으로 남겨둠
    raw_data = fetch_data(UNIQUE_TICKERS, start_date="1990-01-01")

    if raw_data.empty:
        print("데이터 다운로드에 실패했습니다.")
        return

    # 2. 계산
    cs_grai_raw = calculate_cs_grai(raw_data)

    # 3. 정규화 (Z-Score)
    # 수정: Adaptive Normalization (5-Year Rolling Window)
    # Expanding Window의 문제점(Upward Bias)을 해결하기 위해, 최근 5년(경기 사이클)을 기준으로
    # 0점을 재설정(Re-centering)하여 현재 시장의 과열/침체를 상대적으로 평가함.
    rolling_window_z = 252 * 5
    grai_mean = cs_grai_raw.rolling(window=rolling_window_z, min_periods=252).mean()
    grai_std = cs_grai_raw.rolling(window=rolling_window_z, min_periods=252).std()
    
    cs_grai_z = (cs_grai_raw - grai_mean) / grai_std
    cs_grai_smooth = cs_grai_z.rolling(window=5).mean() # 스무딩

    # 4. 그리기
    fig, ax1 = plt.subplots(figsize=(15, 8))
    plt.style.use('bmh')

    # 메인 차트
    line1 = ax1.plot(cs_grai_smooth.index, cs_grai_smooth, label='Global Risk Appetite (Proxy)', color='#1f77b4', linewidth=1.5)
    
    # 기준선
    ax1.axhline(y=THRESHOLD, color='r', linestyle='--', alpha=0.6)
    ax1.axhline(y=-THRESHOLD, color='g', linestyle='--', alpha=0.6)
    ax1.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    
    ax1.fill_between(cs_grai_smooth.index, THRESHOLD, cs_grai_smooth, where=(cs_grai_smooth >= THRESHOLD), facecolor='red', alpha=0.3)
    ax1.fill_between(cs_grai_smooth.index, -THRESHOLD, cs_grai_smooth, where=(cs_grai_smooth <= -THRESHOLD), facecolor='green', alpha=0.3)

    # 현재값 표시
    if not cs_grai_smooth.empty and not np.isnan(cs_grai_smooth.iloc[-1]):
        last_date = cs_grai_smooth.index[-1]
        last_val = cs_grai_smooth.iloc[-1]
        ax1.plot(last_date, last_val, marker='o', color='red', markersize=8)
        ax1.annotate(f'{last_val:.2f}', xy=(last_date, last_val), xytext=(10, 5), 
                     textcoords='offset points', color='red', fontweight='bold', fontsize=12)

    ax1.set_ylabel('Risk Appetite (Z-Score)', fontsize=12, color='#1f77b4')
    ax1.set_title('Credit Suisse Global Risk Appetite Index (CS GRAI)', fontsize=18, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)

    # 보조 차트 (S&P 500)
    if '^GSPC' in raw_data.columns:
        ax2 = ax1.twinx()
        sp500 = raw_data['^GSPC'].loc[cs_grai_smooth.index[0]:]
        line2 = ax2.plot(sp500.index, sp500, color='gray', alpha=0.4, linewidth=1, label='S&P 500')
        ax2.set_ylabel('S&P 500 Level', color='gray')
        ax2.grid(False)
        
        # 범례 통합
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    # plt.show() # Blocking prevention
    plt.savefig('csgrai_result.png')
    print("그래프가 'csgrai_result.png'로 저장되었습니다.")

    # 결과 텍스트
    print(f"\n현재 CS GRAI 지수: {cs_grai_smooth.iloc[-1]:.2f}")

    # 5. 이벤트 분석 실행
    if '^GSPC' in raw_data.columns:
        print("\n이벤트 기반 수익률 분석 중...")
        sp500 = raw_data['^GSPC']
        
        # 분석 수행
        results_df = analyze_episodes(cs_grai_smooth, sp500, threshold=THRESHOLD)
        
        if not results_df.empty:
            # 날짜순 정렬
            results_df = results_df.sort_values(by='Date')
            
            # 컬럼 순서 재정렬 (보기 좋게)
            cols_order = ['Date', 'Signal', 'Type', 'GRAI_Value', '1W', '1M', '3M', '6M', '12M']
            results_df = results_df[cols_order]
            
            # CSV 저장
            # CSV 저장
            filename = f"GRAI_Event_Analysis_{datetime.now().strftime('%Y%m%d')}.csv"
            try:
                results_df.to_csv(filename, index=False)
                print(f"결과 파일 저장됨: {filename}")
            except PermissionError:
                # 파일이 열려있어서 저장 실패시, 시간 포함한 새 이름으로 저장
                new_filename = filename.replace(".csv", f"_{datetime.now().strftime('%H%M%S')}.csv")
                results_df.to_csv(new_filename, index=False)
                print(f"[알림] 기존 파일이 열려있어 다른 이름으로 저장되었습니다: {new_filename}")
            
            # 결과 출력 포맷팅
            print_df = results_df.copy()
            time_cols = ['1W', '1M', '3M', '6M', '12M']
            for col in time_cols:
                print_df[col] = print_df[col].apply(lambda x: f"{x*100:6.2f}%" if pd.notnull(x) else "-")
                
            print(f"\n[분석 완료] 총 {len(results_df)}개의 시그널 포인트 발견")
            print(f"결과 파일 저장됨: {filename}")
            
            print("\n=== 최근 10개 주요 시점 분석 결과 ===")
            print(print_df.tail(10).to_string(index=False))
            
        else:
            print("지정된 기간 내에 기준선을 넘는 이벤트가 발생하지 않았습니다.")
    else:
        print("S&P 500 데이터가 없어 수익률 분석이 불가능합니다.")

if __name__ == "__main__":
    main()