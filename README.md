# Credit Suisse Risk Appetite Index (CS GRAI) Calculator

이 프로젝트는 **Credit Suisse Risk Appetite Index (CS GRAI)**를 계산하고 시각화하며, 과거 데이터를 기반으로 시장의 패닉(Panic) 및 과열(Euphoria) 구간을 분석하는 도구입니다.

## 📋 기능 (Features)

- **데이터 수집**: `yfinance`를 사용하여 전 세계 주식, 채권, 원자재 등 다양한 자산군의 데이터를 자동으로 수집합니다.
- **GRAI 계산**: 자산들의 Risk(변동성)와 Return(수익률) 간의 회귀분석을 통해 Risk Appetite Index를 산출합니다.
- **시각화**: 계산된 지수를 Z-Score로 정규화하여 시각적으로 표현하고, 기준선(±2.0)을 통해 시장 상태를 직관적으로 보여줍니다.
- **이벤트 백테스팅**: 패닉(Panic) 및 과열(Euphoria) 구간을 식별하고, 해당 시점 이후의 수익률(1주, 1개월, 3개월 등)을 분석하여 CSV로 저장합니다.

## 🛠️ 설치 방법 (Installation)

이 프로젝트를 실행하기 위해서는 Python 3.x가 필요합니다. 아래 명령어를 통해 필요한 라이브러리를 설치하세요.

```bash
pip install -r requirements.txt
```

### 필수 라이브러리
- `yfinance`: 금융 데이터 다운로드
- `pandas`: 데이터 처리 및 분석
- `numpy`: 수치 계산
- `scikit-learn`: 선형 회귀 분석
- `matplotlib`: 차트 시각화

## 🚀 사용 방법 (Usage)

터미널에서 아래 명령어를 실행하면 데이터 다운로드, 지수 계산, 시각화 및 분석이 순차적으로 진행됩니다.

```bash
python csrai.py
```

### 실행 과정
1. **데이터 다운로드**: 설정된 자산 유니버스(선진국/신흥국 주식 및 채권, 원자재 등)의 데이터를 가져옵니다.
2. **지수 산출**: 매일의 Risk-Return 관계를 분석하여 GRAI를 계산합니다.
3. **차트 출력**: CS GRAI 지수와 S&P 500 지수를 함께 시각화한 그래프가 팝업됩니다.
4. **결과 분석**: 이벤트(Panic/Euphoria) 발생 시점과 이후 수익률을 분석하여 콘솔에 출력하고, `GRAI_Event_Analysis_YYYYMMDD.csv` 파일로 저장합니다.

## 📊 결과물 (Output)

- **차트**: Risk Appetite의 변화 추이와 시장 국면(Panic/Euphoria)을 보여주는 그래프.
- **CSV 파일**: 이벤트 발생 시점별 상세 분석 데이터.

## ⚠️ 주의사항

- 초기 실행 시 데이터 다운로드에 시간이 소요될 수 있습니다.
- `yfinance`의 데이터 수신 상태에 따라 일부 티커의 데이터가 누락될 수 있습니다.
