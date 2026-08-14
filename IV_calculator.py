import pandas as pd
import numpy as np
from pathlib import Path

def calculate_iv(df, feature_name, target_name):
    """
    특정 속성(구간화된 변수)에 대한 WoE와 IV를 계산하는 함수
    """
    # 1. 전체 데이터에서의 제구 성공(1)과 실패(0) 총 개수
    total_1 = df[target_name].sum()
    total_0 = df[target_name].count() - total_1
    
    # 2. 속성의 각 구간별로 1과 0의 개수 집계
    grouped = df.groupby(feature_name, observed=False)[target_name].agg(
        total='count',
        target_1='sum'
    ).reset_index()
    
    grouped['target_0'] = grouped['total'] - grouped['target_1']
    
    # 3. 전체 대비 구간별 비율 계산
    eps = 1e-6 # log(0) 방지용 극소값
    grouped['rate_1'] = (grouped['target_1'] / total_1).clip(lower=eps)
    grouped['rate_0'] = (grouped['target_0'] / total_0).clip(lower=eps)
    
    # 4. WoE (Weight of Evidence) 계산
    grouped['WoE'] = np.log(grouped['rate_1'] / grouped['rate_0'])
    
    # 5. 각 구간의 IV 계산 및 총 IV 합산
    grouped['IV'] = (grouped['rate_1'] - grouped['rate_0']) * grouped['WoE']
    total_iv = grouped['IV'].sum()
    
    return grouped, total_iv


# ==========================================
# 📂 [실제 CSV 파일 적용 부분]
# ==========================================

# 1. CSV 파일 불러오기
# 가지고 계신 csv 파일의 경로나 이름을 아래에 입력해 주세요.
data_path = Path(__file__).resolve().parent
file_path = data_path / 'data' / 'train.csv' 
df = pd.read_csv(file_path)

# 2. 사용할 컬럼명 지정 (실제 csv 파일의 컬럼명에 맞게 수정)
target_col = 'control_success'       # 예: 제구 성공 여부 컬럼 (0 또는 1)
feature_col = 'season'  # 예: 분석하고자 하는 속성 컬럼 (수직 릴리스 포인트 등)

# 3. 데이터 결측치 처리 (선택 사항)
# 분석할 속성이나 타겟 변수에 결측치(NaN)가 있다면 제거해 줍니다.
df = df.dropna(subset=[target_col, feature_col])

# 4. 연속형 변수 구간화 (Binning)
# 분석할 속성이 소수점을 가지는 연속형 데이터라면 10개 등의 구간으로 나누어야 합니다.
# (만약 '구종'처럼 이미 나누어져 있는 범주형 데이터라면 이 과정은 생략하고 feature_name에 원래 컬럼명을 넣으시면 됩니다.)
binned_col = f'{feature_col}_binned'
df[binned_col] = pd.qcut(df[feature_col], q=10, duplicates='drop')                    # continuous한 경우
# df[binned_col] = pd.cut(df[feature_col], bins=[-1, 0, 1, 2, 3, 4, 5, 6], labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])    # discrete한 경우

# 5. IV 계산 함수 실행
iv_table, total_iv = calculate_iv(df, feature_name=binned_col, target_name=target_col)

# 6. 결과 출력
print(f"🔹 분석 속성: {feature_col}")
print(f"🔹 총 Information Value (Total IV): {total_iv:.4f}\n")
print(f"🔹 구간별 상세 수치:")
print(iv_table[[binned_col, 'total', 'target_1', 'rate_1', 'rate_0', 'WoE', 'IV']])