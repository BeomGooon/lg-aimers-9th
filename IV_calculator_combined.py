import pandas as pd
import numpy as np
from pathlib import Path

def calculate_iv(df, feature_name, target_name):
    # (앞서 제공해드린 IV 계산 함수와 완전히 동일합니다)
    total_1 = df[target_name].sum()
    total_0 = df[target_name].count() - total_1
    
    grouped = df.groupby(feature_name, observed=False)[target_name].agg(
        total='count', target_1='sum').reset_index()
    grouped['target_0'] = grouped['total'] - grouped['target_1']
    
    eps = 1e-6
    grouped['rate_1'] = (grouped['target_1'] / total_1).clip(lower=eps)
    grouped['rate_0'] = (grouped['target_0'] / total_0).clip(lower=eps)
    
    grouped['WoE'] = np.log(grouped['rate_1'] / grouped['rate_0'])
    grouped['IV'] = (grouped['rate_1'] - grouped['rate_0']) * grouped['WoE']
    
    return grouped, grouped['IV'].sum()

# ==========================================
# 📂 [두 속성의 조합 분석 코드]
# ==========================================

data_path = Path(__file__).resolve().parent / 'data' / 'train.csv'
df = pd.read_csv(data_path)


target_col = 'control_success'
feature_A = 'runner_on_2b'
feature_B = 'runner_on_3b'

# 1. 각각의 속성을 4, 3개의 구간으로 나누기
# 4x3 = 총 12개의 조합이 생성됨
bin_A = f'{feature_A}_binned'
bin_B = f'{feature_B}_binned'
# df[bin_A] = pd.qcut(df[feature_A], q=10, duplicates='drop')  
df[bin_A] = pd.cut(df[feature_A], bins=[-1, 0, 1], labels=["not 2b", "yes 2b"])
df[bin_B] = pd.cut(df[feature_B], bins=[-1, 0, 1], labels=["not 3b", "yes 3b"])


# ==========================================
# 분석 1: 2D 피벗 테이블로 조합별 성공률 한눈에 보기
# ==========================================
print("📊 [분석 1: 두 속성 조합별 제구 성공률 (Pivot Table)]")
# 평균을 구하면 타겟(0, 1)의 평균이므로 곧 '성공률'이 됩니다.
pivot_success = df.pivot_table(
    values=target_col, 
    index=bin_A, 
    columns=bin_B, 
    aggfunc='mean'
)
# 백분율(%)로 보기 좋게 변환
print((pivot_success * 100).round(2))
print("\n" + "="*50 + "\n")


# ==========================================
# 분석 2: 두 속성을 하나로 합쳐서 IV(Information Value) 계산
# ==========================================
print("🧮 [분석 2: 속성 조합의 Information Value (IV) 계산]")

# 두 구간을 문자열로 이어붙여 새로운 파생 변수(조합 변수) 생성
combined_feature = f'{feature_A}_AND_{feature_B}'
df[combined_feature] = df[bin_A].astype(str) + " | " + df[bin_B].astype(str)

# 기존 IV 함수에 결합된 변수를 넣고 계산
iv_table, total_iv = calculate_iv(df, feature_name=combined_feature, target_name=target_col)

print(f"🔹 결합 속성: {combined_feature}")
print(f"🔹 조합된 Total IV: {total_iv:.4f}")
print(f"🔹 (참고: 두 속성 단독 IV의 합보다 조합 IV가 훨씬 높다면 상호작용 효과가 큰 것입니다.)\n")

# IV값이 가장 높거나 성공률이 가장 높은 상위 5개 조합만 출력
top_combinations = iv_table.sort_values(by='IV', ascending=False).head(5)
print("🔹 제구에 가장 큰 영향을 미치는(IV가 높은) 상위 5개 조합:")
print(top_combinations[[combined_feature, 'total', 'target_1', 'WoE', 'IV']])