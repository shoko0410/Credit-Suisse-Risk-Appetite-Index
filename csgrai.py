import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional, Union

# ==============================================================================
# 0. 설정 및 상수 (Configuration & Constants)
# ==============================================================================
RET_WINDOW = 126  # 6개월 (Official CS Whitepaper)
VOL_WINDOW = 252  # 12개월 (Official CS Whitepaper)
MIN_ASSETS = 15
THRESHOLD = 2.0   # 이벤트 분석 임계값

# 자산 유니버스 정의 (Data Universe)
# 선진국 주식
DM_EQUITIES = [
    'SPY', 'QQQ', 'IWM', 'EWJ', 'EWG', 'EWU', 'EWQ', 
    'EWL', 'EWC', 'EWA', 'EWD', 'EWH', 'EWS'
]
# 선진국 채권
DM_BONDS = ['SHY', 'IEF', 'TLT', 'BWX', 'IGOV']
# 신흥국 주식
EM_EQUITIES = ['EEM', 'FXI', 'EWY', 'EWT', 'INDA', 'EWZ', 'EWW', 'EZA', 'TUR']
# 신흥국 채권
EM_BONDS = ['EMB', 'EMLC']
# 원자재 (선물 직접 사용 - 추적오차 및 롤오버 비용 제거)
COMMODITIES = [
    'GC=F',   # Gold Futures (금 선물)
    'SI=F',   # Silver Futures (은 선물)
    'CL=F',   # WTI Crude Oil Futures (원유 선물)
    'HG=F',   # Copper Futures (구리 선물)
    'NG=F',   # Natural Gas Futures (천연가스 선물)
    'VNQ'     # Vanguard Real Estate ETF (리츠 - 선물 대체재 없음)
]
# 벤치마크 (S&P 500)
BENCHMARK = ['^GSPC']

RISK_FREE = ['^IRX'] # 13 Week Treasury Bill
TICKERS = DM_EQUITIES + DM_BONDS + EM_EQUITIES + EM_BONDS + COMMODITIES + BENCHMARK + RISK_FREE
UNIQUE_TICKERS = list(set(TICKERS))

# ==============================================================================
# 1. 데이터 수집 함수
# ==============================================================================
def clean_outliers(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """일간 수익률이 threshold(50%)를 초과하는 이상치 제거 (Data Cleaning)"""
    print("  [Debug] clean_outliers 시작", flush=True)
    # 수익률 계산
    returns = df.pct_change()
    
    # 이상치 마스크 생성 (절대값 50% 이상 변동)
    mask = returns.abs() > threshold
    outlier_count = mask.sum().sum()
    
    if outlier_count > 0:
        print(f"데이터 정제: 총 {outlier_count}개의 이상치(>50% 변동)를 감지하여 제거합니다.", flush=True)
        # 이상치를 NaN으로 처리하고 이전 값으로 채움
        df_clean = df.copy()
        df_clean[mask] = np.nan
        df_clean = df_clean.ffill()
        print("  [Debug] clean_outliers 완료 (이상치 제거됨)", flush=True)
        return df_clean
    
    print("  [Debug] clean_outliers 완료 (이상치 없음)", flush=True)
    return df

def fetch_data(ticker_list: List[str], start_date: str = "1990-01-01") -> pd.DataFrame:
    print("데이터 다운로드 시작... (yf.download 호출)", flush=True)
    try:
        # auto_adjust=False 옵션 추가 (Adj Close 확보용)
        df = yf.download(ticker_list, start=start_date, progress=False, auto_adjust=False)
        print("데이터 다운로드 완료 (yf.download 반환)", flush=True)
        
        if 'Adj Close' in df.columns:
            data = df['Adj Close']
        elif 'Close' in df.columns:
            print("알림: 'Adj Close' 대신 'Close' 데이터를 사용합니다.", flush=True)
            data = df['Close']
        else:
            print("오류: 데이터 컬럼을 찾을 수 없습니다.", flush=True)
            return pd.DataFrame()
            
    except Exception as e:
        print(f"다운로드 오류: {e}", flush=True)
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    # 1. 이상치 제거 (Outlier Cleaning)
    data = clean_outliers(data)

    # 2. 데이터 품질 관리
    # 수정: 생존 편향 제거를 위해 dropna(axis=1) 및 dropna(axis=0, how='any') 삭제
    # 모든 자산이 NaN인 날짜만 제거
    print("  [Debug] 데이터 품질 관리 (dropna/ffill) 시작", flush=True)
    data = data.dropna(axis=0, how='all') 
    data = data.ffill() # 결측치 채우기 (중간 결측 방지)
    print("  [Debug] 데이터 품질 관리 완료", flush=True)
    
    # 벤치마크(^GSPC)가 있는 날짜부터 시작하도록 조정 (선택사항이나 그래프 매칭 위해 권장)
    if '^GSPC' in data.columns:
        first_valid_idx = data['^GSPC'].first_valid_index()
        if first_valid_idx is not None:
            data = data.loc[first_valid_idx:]
    
    print(f"데이터 수집 완료: {data.shape[1]}개 자산, {data.shape[0]}일 데이터 (Dynamic Universe 적용)", flush=True)
    return data

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
        volatility = risk_matrix.iloc[i]
        excess_returns = excess_return_matrix.iloc[i]
        
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
    # 수정: Rolling Window가 아닌 Expanding Window 사용 (Regime Dilution 방지)
    # 과거의 위기/호황 데이터를 잊지 않고 누적하여 현재 수준을 역사적 관점에서 평가함
    grai_mean = cs_grai_raw.expanding(min_periods=252).mean()
    grai_std = cs_grai_raw.expanding(min_periods=252).std()
    
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
            filename = f"GRAI_Event_Analysis_{datetime.now().strftime('%Y%m%d')}.csv"
            results_df.to_csv(filename, index=False)
            
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