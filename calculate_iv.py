import os
import pandas as pd
import numpy as np
import warnings
# qcut에서 발생하는 일부 경고 무시
warnings.filterwarnings('ignore')

def preprocess_data_func(df):
    df = df.copy()
    
    # ---- RE24 (기대 득점) 피처 추가 ----
    re24_table = {
        0: 0.51, 1: 0.27, 2: 0.11,             # 주자 없음
        1000: 0.88, 1001: 0.53, 1002: 0.23,    # 1루
        100: 1.14, 101: 0.69, 102: 0.32,       # 2루
        10: 1.39, 11: 0.98, 12: 0.37,          # 3루
        1100: 1.47, 1101: 0.91, 1102: 0.44,    # 1, 2루
        1010: 1.74, 1011: 1.18, 1012: 0.50,    # 1, 3루
        110: 2.01, 111: 1.41, 112: 0.59,       # 2, 3루
        1110: 2.33, 1111: 1.57, 1112: 0.77     # 만루
    }
    state_code = df['runner_on_1b'].astype(int)*1000 + df['runner_on_2b'].astype(int)*100 + df['runner_on_3b'].astype(int)*10 + df['outs_before'].astype(int)
    df['re24'] = state_code.map(re24_table)
    
    # ---- 새로 추가된 변수 (투타 매치업 및 볼카운트 압박감) ----
    df['same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
    df['count_advantage'] = (df['strikes_before'] - df['balls_before'] + 3) / 5.0
    
    # ---- 수식 기반 전처리 (수치형) ----
    df['balls_before'] = df['balls_before'] / 3.0
    df['strikes_before'] = df['strikes_before'] / 2.0
    df['outs_before'] = df['outs_before'] / 2.0
    df['score_diff_pitcher_team'] = df['score_diff_pitcher_team'].abs().clip(upper=5) / 5.0
    df['asof_batter_n'] = np.log1p(df['asof_batter_n'])
    df['asof_pitcher_n'] = np.log1p(df['asof_pitcher_n'])
    df['win_expectancy_pitcher_team'] = np.where(df['top_bottom'] == 'T', df['home_win_expectancy'], df['away_win_expectancy'])
    df['asof_pitcher_pitchmix_n'] = np.log1p(df['asof_pitcher_pitchmix_n'])
    df['inning'] = (df['inning'] - 1).clip(upper=8) / 8.0
    df['li'] = df['li'].clip(upper=2) / 2.0
    
    # ---- 파생 변수 (모멘텀/컨디션) 추가 ----
    df['pitcher_momentum_1_vs_5'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate'].fillna(0)
    df['pitcher_condition_vs_baseline'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    
    # ---- 범주형 변수 결측치 처리 및 문자열 캐스팅 ----
    cat_cols = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id', 'pitcher_hand', 'batter_hand', 'game_type', 'top_bottom']
    for col in cat_cols:
        df[col] = df[col].fillna('MISSING').astype(str)
        
    use_features = [
        'balls_before', 'strikes_before', 'outs_before', 
        'score_diff_pitcher_team', 'runner_on_1b', 'runner_on_2b', 'runner_on_3b',
        'asof_batter_n', 'asof_pitcher_success_rate', 'asof_pitcher_n',
        'asof_pitcher_strike_rate', 'win_expectancy_pitcher_team', 
        'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate',
        'asof_batter_success_rate',
        'asof_pitcher_prev1_game_success_rate',
        'asof_pitcher_prev3_game_success_rate',
        'asof_pitcher_prev5_game_success_rate',
        'pitcher_momentum_1_vs_5',
        'pitcher_condition_vs_baseline',
        'asof_pitcher_prev1_game_middle_rate',
        'asof_pitcher_prev3_game_middle_rate',
        'asof_pitcher_prev5_game_middle_rate',
        'asof_pitcher_pitchmix_n',
        'inning',
        'li',
        'same_hand',
        'count_advantage',
        'asof_pitcher_middle_rate',
        'asof_pitcher_reverse_rate',
        'asof_pitcher_ball_rate',
        'asof_pitcher_fastball_rate',
        'season',
        'game_month',
        're24'
    ] + cat_cols
    
    return df[use_features], cat_cols

def calculate_iv(df, feature, target, bins=10):
    """
    특정 피처에 대한 Information Value(IV)를 계산하는 함수
    """
    temp_df = df[[feature, target]].copy()
    
    # 1. 구간 나누기 (Binning)
    if temp_df[feature].dtype.kind in 'bifc': # 수치형인 경우
        if temp_df[feature].nunique() <= bins:
            # 고유값이 적으면 그대로 사용
            temp_df['binned'] = temp_df[feature]
        else:
            # qcut을 통해 동일한 개수가 들어가도록 구간 분할
            try:
                temp_df['binned'] = pd.qcut(temp_df[feature], q=bins, duplicates='drop')
            except Exception:
                temp_df['binned'] = pd.cut(temp_df[feature], bins=bins)
    else: # 범주형인 경우
        temp_df['binned'] = temp_df[feature]
        
    # 2. 구간별 타겟 빈도 계산
    grouped = temp_df.groupby('binned', observed=False)[target].agg(['count', 'sum'])
    grouped.columns = ['Total', 'Goods'] # Target=1 (제구 성공)
    grouped['Bads'] = grouped['Total'] - grouped['Goods'] # Target=0 (제구 실패)
    
    # 3. 0값 보정 (log(0) 에러 방지 - 표준적인 0.5 더하기 기법)
    grouped['Goods'] = grouped['Goods'].replace(0, 0.5)
    grouped['Bads'] = grouped['Bads'].replace(0, 0.5)
    
    # 4. 전체 Good / Bad 개수
    total_goods = temp_df[target].sum()
    total_bads = temp_df[target].count() - total_goods
    
    # 5. 비율 계산
    grouped['Good_Rate'] = grouped['Goods'] / total_goods
    grouped['Bad_Rate'] = grouped['Bads'] / total_bads
    
    # 6. WoE 및 IV 계산
    grouped['WoE'] = np.log(grouped['Good_Rate'] / grouped['Bad_Rate'])
    grouped['IV'] = (grouped['Good_Rate'] - grouped['Bad_Rate']) * grouped['WoE']
    
    return grouped['IV'].sum()

def main():
    TRAIN_PATH = "data/train.csv"
    TARGET_COL = "control_success"
    
    print("Loading data...")
    raw_df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    y = raw_df[TARGET_COL]
    
    print("Preprocessing data (applying transformations)...")
    X, _ = preprocess_data_func(raw_df)
    
    # Target 변수를 X에 잠시 결합
    X[TARGET_COL] = y
    
    features = [col for col in X.columns if col != TARGET_COL]
    
    print(f"Calculating IV for {len(features)} features...")
    iv_results = []
    
    for i, feature in enumerate(features, 1):
        try:
            iv = calculate_iv(X, feature, TARGET_COL, bins=10)
            iv_results.append({'Feature': feature, 'IV': iv})
            print(f"[{i}/{len(features)}] {feature}: {iv:.4f}")
        except Exception as e:
            print(f"[{i}/{len(features)}] {feature}: Error ({e})")
            iv_results.append({'Feature': feature, 'IV': 0.0})
            
    # 결과를 DataFrame으로 변환 및 정렬
    iv_df = pd.DataFrame(iv_results).sort_values(by='IV', ascending=False).reset_index(drop=True)
    
    print("\n=============================================")
    print("Top 15 Features by Information Value (IV)")
    print("=============================================")
    print(iv_df.head(15).to_string(index=False))
    
    print("\n=============================================")
    print("Bottom 15 Features by Information Value (IV)")
    print("=============================================")
    print(iv_df.tail(15).to_string(index=False))
    
    # CSV로 저장
    save_path = "data/iv_results.csv"
    os.makedirs("data", exist_ok=True)
    iv_df.to_csv(save_path, index=False)
    print(f"\n[SUCCESS] Full IV calculation results saved to {save_path}")
    
    # IV 판단 기준 참고용 출력
    print("\n[참고: IV 해석 기준]")
    print("< 0.02    : 예측력 거의 없음 (Useless)")
    print("0.02~0.10 : 예측력 약함 (Weak)")
    print("0.10~0.30 : 예측력 중간 (Medium)")
    print("0.30~0.50 : 예측력 강함 (Strong)")
    print("> 0.50    : 예측력이 비정상적으로 높음 (Suspicious, Leakage 의심)")

if __name__ == "__main__":
    main()
