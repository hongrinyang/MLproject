import numpy as np
import matplotlib.pyplot as plt

# 기본 가정값 설정
total_capital = 2.3e12  # 총 자본금 (2.3조 달러)
misrealized_losses_initial = 2.55e12  # 초기 미실현 손실 (2.55조 달러)
bond_ratio = 0.35  # 미국 은행들이 보유한 10년물 채권 비율 (35%)
total_dif_reserves = 1.664e11  # DIF 지급준비금 (1664억 달러)

# 금리 변화에 따른 미실현 손실의 비율 (듀레이션 6년 가정)
# 금리 상승 시 손실의 비율: 금리가 1% 상승할 때 손실 비율 6%로 설정 (듀레이션 기반)
duration = 6
interest_rate_change = np.linspace(-2, 2, 100)  # 금리 변화 범위 (-2%에서 +2%까지)

# 금리 변화에 따른 미실현 손실 변화 (금리가 오르면 손실이 커지고, 내리면 줄어듦)
misrealized_losses = misrealized_losses_initial * (1 + (interest_rate_change * duration * 0.01))

# 미실현 손실이 자본금보다 클 때 파산이 발생한다고 가정
def bankruptcy_possible(losses, total_capital, dif_reserves):
    return losses > (total_capital + dif_reserves)

# 자금 유출 계산 함수 (파산이 발생하면 유출 비율 증가)
def calculate_outflow(losses, total_capital, dif_reserves):
    if bankruptcy_possible(losses, total_capital, dif_reserves):
        return np.clip((losses - total_capital - dif_reserves) / total_capital, 0, 1) * 0.4  # 최대 40% 자금 유출
    return 0  # 파산 안 일어나면 자금 유출 없음

# 자금 유출 비율 계산
outflows = np.array([calculate_outflow(loss, total_capital, total_dif_reserves) for loss in misrealized_losses])

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(interest_rate_change, outflows * 100, label='NASDAQ Outflow (%)', color='b')
plt.xlabel('Change in Interest Rate (%)')
plt.ylabel('NASDAQ Outflow (%)')
plt.title('NASDAQ Outflow vs. Interest Rate Change')
plt.axhline(0, color='k', linewidth=1)
plt.axvline(0, color='k', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()
plt.show()

# 금리 변화에 따른 미실현 손실 및 자금 유출 정보 출력
for i in range(len(interest_rate_change)):
    print(f"금리 변화: {interest_rate_change[i]:.2f}% -> 미실현 손실: {misrealized_losses[i]/1e12:.2f}조 달러, 자금 유출: {outflows[i]*100:.2f}%")

