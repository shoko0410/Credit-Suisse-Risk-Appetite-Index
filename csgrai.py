import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# 1. 자산 유니버스 정의 (Data Universe)
# ==============================================================================
# 선진국 주식
dm_equities = [
    'SPY', 'QQQ', 'IWM', 'EWJ', 'EWG', 'EWU', 'EWQ', 
    'EWL', 'EWC', 'EWA', 'EWD', 'EWH', 'EWS'
]
# 선진국 채권
dm_bonds = ['SHY', 'IEF', 'TLT', 'BWX', 'IGOV']
# 신흥국 주식
em_equities = ['EEM', 'FXI', 'EWY', 'EWT', 'INDA', 'EWZ', 'EWW', 'EZA', 'TUR']
# 신흥국 채권
em_bonds = ['EMB', 'EMLC']
# 원자재 및 리츠
commodities = ['GLD', 'SLV', 'USO', 'CPER', 'DBC', 'VNQ']
# 벤치마크 (S&P 500)
benchmark = ['^GSPC']

tickers = dm_equities + dm_bonds + em_equities + em_bonds + commodities + benchmark
unique_tickers = list(set(tickers))

print(f"총 {len(unique_tickers)}개의 자산으로 인덱스를 구성합니다.")

# ==============================================================================
# 2. 데이터 수집 함수 (수정됨: auto_adjust=False)
# ==============================================================================
def fetch_data(ticker_list, start_date="2010-01-01"):
    print("데이터 다운로드 중... (약 1분 정도 소요될 수 있습니다)")
    try:
        # 수정사항: auto_adjust=False 옵션 추가 (Adj Close 확보용)
        df = yf.download(ticker_list, start=start_date, progress=False, auto_adjust=False)
        
        if 'Adj Close' in df.columns:
            data = df['Adj Close']
        elif 'Close' in df.columns:
            print("알림: 'Adj Close' 대신 'Close' 데이터를 사용합니다.")
            data = df['Close']
        else:
            print("오류: 데이터 컬럼을 찾을 수 없습니다.")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"다운로드 오류: {e}")
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    # 데이터 품질 관리
    threshold = int(data.shape[0] * 0.6)
    data = data.dropna(axis=1, thresh=threshold) # 데이터 부족한 자산 제외
    data = data.ffill() # 결측치 채우기
    data = data.dropna(axis=0, how='any') # 앞부분 NaN 제거
    
    print(f"데이터 수집 완료: {data.shape[1]}개 자산, {data.shape[0]}일 데이터")
    return data

# ==============================================================================
# 3. CS GRAI 계산 엔진
# ==============================================================================
def calculate_cs_grai(price_data):
    RET_WINDOW = 126  # 6개월
    VOL_WINDOW = 252  # 12개월
    MIN_ASSETS = 15
    
    grai_results = {}
    valid_assets = [t for t in price_data.columns if t != '^GSPC']
    
    start_idx = max(RET_WINDOW, VOL_WINDOW)
    total_days = len(price_data)
    dates = price_data.index
    
    print("지수 산출 중... (잠시만 기다려주세요)")
    
    for i in range(start_idx, total_days):
        current_date = dates[i]
        
        # 데이터 슬라이싱
        subset_vol = price_data[valid_assets].iloc[i-VOL_WINDOW : i+1]
        p_current = price_data[valid_assets].iloc[i]
        p_past = price_data[valid_assets].iloc[i-RET_WINDOW]
        
        # Risk (변동성) & Return (수익률) 계산
        log_rets = np.log(subset_vol / subset_vol.shift(1))
        volatility = log_rets.std() * np.sqrt(252)
        returns = (p_current / p_past) - 1
        
        daily_df = pd.DataFrame({'Risk': volatility, 'Return': returns}).dropna()
        
        if len(daily_df) < MIN_ASSETS:
            grai_results[current_date] = np.nan
            continue
            
        # 회귀분석
        X = daily_df['Risk'].values.reshape(-1, 1)
        y = daily_df['Return'].values
        
        model = LinearRegression()
        model.fit(X, y)
        grai_results[current_date] = model.coef_[0]
        
        if i % 1000 == 0:
            print(f"Processing: {current_date.date()}")

    return pd.Series(grai_results)

# ==============================================================================
# 4. 실행 및 시각화 (Main Execution)
# ==============================================================================
# 1. 실행 (여기서 fetch_data를 호출합니다)
raw_data = fetch_data(unique_tickers, start_date="2010-01-01")

if not raw_data.empty:
    # 2. 계산
    cs_grai_raw = calculate_cs_grai(raw_data)

    # 3. 정규화 (Z-Score)
    rolling_window = 252 * 3
    grai_mean = cs_grai_raw.rolling(window=rolling_window, min_periods=252).mean()
    grai_std = cs_grai_raw.rolling(window=rolling_window, min_periods=252).std()
    
    cs_grai_z = (cs_grai_raw - grai_mean) / grai_std
    cs_grai_smooth = cs_grai_z.rolling(window=5).mean() # 스무딩

    # 4. 그리기
    fig, ax1 = plt.subplots(figsize=(15, 8))
    plt.style.use('bmh')

    # 메인 차트
    line1 = ax1.plot(cs_grai_smooth.index, cs_grai_smooth, label='Global Risk Appetite (Proxy)', color='#1f77b4', linewidth=1.5)
    
    # 기준선
    ax1.axhline(y=2.0, color='r', linestyle='--', alpha=0.6)
    ax1.axhline(y=-2.0, color='g', linestyle='--', alpha=0.6)
    ax1.axhline(y=0, color='black', linewidth=1, alpha=0.5)
    
    ax1.fill_between(cs_grai_smooth.index, 2.0, cs_grai_smooth, where=(cs_grai_smooth >= 2.0), facecolor='red', alpha=0.3)
    ax1.fill_between(cs_grai_smooth.index, -2.0, cs_grai_smooth, where=(cs_grai_smooth <= -2.0), facecolor='green', alpha=0.3)

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
    plt.show()

    # 결과 텍스트
    print(f"\n현재 CS GRAI 지수: {cs_grai_smooth.iloc[-1]:.2f}")

else:
    print("데이터 다운로드에 실패했습니다.")

# ==============================================================================
# 5. 심화 백테스팅: 이벤트(Episode)별 진입/극점/회복 수익률 분석
# ==============================================================================
def get_forward_returns(date, price_series, periods):
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

def analyze_episodes(grai_series, price_series, threshold=2.0):
    """
    이벤트 기반 분석:
    1. 진입 (Entry): 기준선 돌파
    2. 극점 (Extremum): 기간 중 최대/최소값
    3. 회복 (Exit): 기준선 복귀
    """
    periods = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '12M': 252}
    events = []
    
    # --- Panic Analysis (Below -2.0) ---
    in_panic = False
    panic_start_date = None
    panic_records = [] # (date, value) 저장용
    
    # --- Euphoria Analysis (Above +2.0) ---
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

# --- 실행 부분 ---
if '^GSPC' in raw_data.columns:
    print("\n이벤트 기반 수익률 분석 중...")
    sp500 = raw_data['^GSPC']
    
    # 분석 수행
    results_df = analyze_episodes(cs_grai_smooth, sp500, threshold=2.0)
    
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