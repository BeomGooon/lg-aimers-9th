import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier, Pool

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
    
    # ---- 새로 추가된 변수 ----
    df['same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
    df['count_advantage'] = (df['strikes_before'] - df['balls_before'] + 3) / 5.0
    
    # ---- 수식 기반 전처리 ----
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
    
    # ---- 파생 변수 (모멘텀/컨디션) ----
    df['pitcher_momentum_1_vs_5'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate'].fillna(0)
    df['pitcher_condition_vs_baseline'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    
    # ---- 범주형 변수 결측치 처리 ----
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

def main():
    MODEL_PATH = "model/catboost_final.cbm"
    DATA_PATH = "data/train.csv"
    
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("먼저 train_cat.py를 실행하여 모델을 학습시켜주세요.")
        return
        
    print("Loading pre-trained CatBoost model...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    
    print("Preprocessing data...")
    X, cat_features = preprocess_data_func(df)
    
    # SHAP 분석은 연산량이 매우 많으므로 2024년 검증 데이터 중 5000개만 샘플링하여 진행
    print("Sampling 5,000 instances from 2024 validation data for SHAP analysis...")
    val_mask = df['season'] == 2024
    X_val = X[val_mask]
    
    if len(X_val) > 5000:
        X_sample = X_val.sample(n=5000, random_state=42)
    else:
        X_sample = X_val
        
    # CatBoost는 Pool 객체를 넘겨주어야 SHAP 계산 시 범주형 변수를 제대로 처리함
    sample_pool = Pool(X_sample, cat_features=cat_features)
    
    print("Calculating SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_pool)
    
    print("Generating SHAP Summary Plot...")
    # 화면 출력을 막고 이미지로 저장하기 위한 설정
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20) # 상위 20개 피처만 표시
    
    save_path = "shap_summary_plot.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[SUCCESS] SHAP Summary Plot saved to: {save_path}")
    print("해당 이미지를 열어서 오른쪽 꼬리가 긴지, 왼쪽 꼬리가 긴지, 0 근처에만 몰려있는지 확인해보세요!")

if __name__ == "__main__":
    main()
