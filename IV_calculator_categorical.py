import pandas as pd
import numpy as np
from pathlib import Path

def calculate_categorical_iv(df, feature_name, target_name):
    """
    범주형 속성에 대한 WoE와 IV를 계산하는 함수
    """
    # 1. 전체 데이터에서의 제구 성공(1)과 실패(0) 총 개수
    total_1 = df[target_name].sum()
    total_0 = df[target_name].count() - total_1
    
    # 2. 범주(Category)별로 1과 0의 개수 집계
    grouped = df.groupby(feature_name, observed=False)[target_name].agg(
        total='count',
        target_1='sum'
    ).reset_index()
    
    grouped['target_0'] = grouped['total'] - grouped['target_1']
    
    # 3. 전체 대비 범주별 비율 계산 (극소값 eps 적용하여 log(0) 에러 방지)
    eps = 1e-6
    grouped['rate_1'] = (grouped['target_1'] / total_1).clip(lower=eps)
    grouped['rate_0'] = (grouped['target_0'] / total_0).clip(lower=eps)
    
    # 4. WoE (Weight of Evidence) 및 각 카테고리의 IV 계산
    grouped['WoE'] = np.log(grouped['rate_1'] / grouped['rate_0'])
    grouped['IV'] = (grouped['rate_1'] - grouped['rate_0']) * grouped['WoE']
    
    # 5. 총 IV 산출 및 IV 기여도가 높은 순으로 정렬
    total_iv = grouped['IV'].sum()
    grouped = grouped.sort_values(by='IV', ascending=False).reset_index(drop=True)
    
    return grouped, total_iv


# ==========================================
# 📂 [실제 데이터 파일 적용 부분]
# ==========================================

# 1. CSV 파일 경로 지정
file_path = Path(__file__).resolve().parent / 'data' / 'train.csv' 
df = pd.read_csv(file_path)

# 2. 타겟 및 분석할 속성 컬럼명 지정
target_col = 'control_success'    # 제구 성공 여부 컬럼명 (0 또는 1의 값을 가져야 함)
feature_col = 'game_type'  # 분석할 범주형 속성 컬럼명 (예: 구종, 타자스탠스, 볼카운트 등)

# 3. 결측치(NaN) 제거
# 타겟이나 해당 속성값에 빈 데이터가 있으면 계산 오류가 발생하므로 제외합니다.
df = df.dropna(subset=[target_col, feature_col])

# 4. IV 계산 함수 실행
iv_table, total_iv = calculate_categorical_iv(df, feature_name=feature_col, target_name=target_col)

# 5. 결과 출력
print(f"📊 분석 속성: {feature_col}")
print(f"🔹 총 Information Value (Total IV): {total_iv:.4f}\n")
print(f"🔹 카테고리별 상세 수치 (IV 기여도 순):")
print(iv_table[[feature_col, 'total', 'target_1', 'rate_1', 'rate_0', 'WoE', 'IV']])