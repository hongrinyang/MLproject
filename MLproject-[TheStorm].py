import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 파일 경로 지정 (파일이 존재하는 경로로 수정)
file_path = "C:\\Users\\{username}\\Downloads\\DGS10.csv"  # 경로 수정

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(file_path):
    # 데이터 로드 (observation_date 열을 날짜 형식으로 파싱)
    df = pd.read_csv(file_path, parse_dates=['observation_date'])
    
    # 'observation_date'를 인덱스로 설정
    df.set_index('observation_date', inplace=True)
    
    # 결측값 처리
    df.dropna(inplace=True)

    # 'year', 'month' 특성 추가 (예: 연도, 월 기준으로 금리 예측)
    df['year'] = df.index.year
    df['month'] = df.index.month

    # 'DGS10'을 예측하려는 타겟 변수로 사용
    X = df[['year', 'month', 'DGS10']]  # 특성 변수: 'year', 'month', 'DGS10'
    y = df['DGS10']  # 타겟 변수: 'DGS10' (10년물 국채 금리)
    
    return X, y

# 미실현 손실 계산 함수
def calculate_unrealized_losses(interest_rate, bond_duration=6, bond_asset_ratio=0.35, capital=2.3e12):
    # 금리가 1% 상승할 때 자산 가치 변화 계산
    bond_price_change = -bond_duration * (interest_rate / 100)  # 금리 변화에 따른 채권 가격 변화
    unrealized_loss = bond_asset_ratio * capital * bond_price_change  # 자산 비중에 따른 미실현 손실
    return unrealized_loss

# 파산 가능성 계산 함수
def bankruptcy_probability(unrealized_loss, capital, reserve_fund=1.664e11):
    # 미실현 손실이 자본금 대비 크면 파산 가능성 증가
    if unrealized_loss > (capital - reserve_fund):
        return 1.0  # 파산 확률 100%
    else:
        return unrealized_loss / (capital - reserve_fund)  # 손실 비율에 따라 확률 계산

# 나스닥 자금 이탈 예측 함수
def nasdaq_outflow(bankruptcy_probability, total_outflow=0.1):
    return bankruptcy_probability * total_outflow

# 데이터셋을 로드하고 전처리하기
X, y = load_and_preprocess_data(file_path)

# 데이터 분할 (훈련용 80%, 테스트용 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 (Linear Regression)
model = LinearRegression()

# 모델 훈련
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 성능 평가
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 성능 출력
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# 실제값과 예측값 시각화
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual DGS10', color='blue', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted DGS10', color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted 10-Year Treasury Yield')
plt.xlabel('Date')
plt.ylabel('10-Year Treasury Yield')
plt.legend()
plt.show()

# 금리 변화에 따른 미실현 손실, 파산 확률, 자금 이탈 예측
interest_rate_change = 1.0  # 금리 변화 예시 (1% 상승)
unrealized_loss = calculate_unrealized_losses(4.57 + interest_rate_change)  # 예시 금리: 4.57%
bankruptcy_prob = bankruptcy_probability(unrealized_loss, 2.3e12)
nasdaq_outflow_amount = nasdaq_outflow(bankruptcy_prob)

# 예측 결과 출력
print(f"Unrealized Loss: {unrealized_loss:.2f}")
print(f"Bankruptcy Probability: {bankruptcy_prob:.4f}")
print(f"Nasdaq Outflow: {nasdaq_outflow_amount:.4f}")
