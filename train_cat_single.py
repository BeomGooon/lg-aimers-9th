import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score

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
    # 판다스의 빠른 연산을 위해 상태 코드를 정수로 생성 (1루*1000 + 2루*100 + 3루*10 + 아웃카운트)
    state_code = df['runner_on_1b'].astype(int)*1000 + df['runner_on_2b'].astype(int)*100 + df['runner_on_3b'].astype(int)*10 + df['outs_before'].astype(int)
    df['re24'] = state_code.map(re24_table)
    df['base_out_state'] = state_code.astype(str)
    
    # ---- 새로 추가된 변수 (투타 매치업 및 볼카운트 압박감) ----
    df['same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
    df['count_advantage'] = (df['strikes_before'] - df['balls_before'] + 3) / 5.0
    
    # ---- 시간 변수 추가 (Season, Month) ----
    # season과 game_month는 그대로 수치형으로 사용 (트리 모델은 스케일링/주기 변환 불필요)
    
    # ---- 수식 기반 전처리 (수치형) ----
    df['balls_before'] = df['balls_before'] / 3.0
    df['strikes_before'] = df['strikes_before'] / 2.0
    df['outs_before'] = df['outs_before'] / 2.0
    df['score_diff_pitcher_team'] = df['score_diff_pitcher_team'].abs().clip(upper=5) / 5.0
    df['asof_batter_n'] = np.log1p(df['asof_batter_n'])
    df['asof_pitcher_n'] = np.log1p(df['asof_pitcher_n'])
    df['win_expectancy_pitcher_team'] = np.where(df['top_bottom'] == 'T', df['home_win_expectancy'], df['away_win_expectancy'])
    df['win_expectancy_pitcher_team'] = (df['win_expectancy_pitcher_team'] - 50).abs() / 50.0
    df['asof_pitcher_pitchmix_n'] = np.log1p(df['asof_pitcher_pitchmix_n'])
    df['inning'] = (df['inning'] - 1).clip(upper=8) / 8.0
    df['li'] = df['li'].clip(upper=2) / 2.0
    
    # ---- 파생 변수 (모멘텀/컨디션) 추가 ----
    df['pitcher_momentum_1_vs_5'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate'].fillna(0)
    df['pitcher_condition_vs_baseline'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    
    # ---- 범주형 변수 결측치 처리 및 문자열 캐스팅 ----
    cat_cols = ['pitcher_team_id', 'batter_team_id', 'pitcher_hand', 'batter_hand', 'game_type', 'base_out_state']
    for col in cat_cols:
        df[col] = df[col].fillna('MISSING').astype(str)
        
    use_features = [
        'balls_before', 'strikes_before', 
        'asof_pitcher_success_rate', 'asof_pitcher_n',
        'asof_pitcher_strike_rate', 'win_expectancy_pitcher_team', 
        'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate',
        'asof_batter_success_rate',
        'asof_pitcher_prev1_game_success_rate',
        'asof_pitcher_prev3_game_success_rate',
        'asof_pitcher_prev5_game_success_rate',
        'pitcher_momentum_1_vs_5',
        'pitcher_condition_vs_baseline',
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
    TRAIN_PATH = "data/train.csv"
    MODEL_DIR = "model"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    
    TARGET_COL = "control_success"
    y = df[TARGET_COL]
    
    # ---- 1. 전처리 ----
    print("Preprocessing data...")
    X, cat_features = preprocess_data_func(df)
    
    # ---- 2. 엄격한 연도별 홀드아웃 (검증용 데이터 분할) ----
    print("\nStarting Strict Season-based Holdout Validation...")
    train_mask = df['season'] <= 2023
    val_mask = df['season'] == 2024
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    print(f"Train samples (2019~2023): {len(X_train)}")
    print(f"Validation samples (2024): {len(X_val)}")
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    # ---- 3. 검증 모델 학습 (최적의 반복 횟수 찾기) ----
    print("\nTraining Validation Model with Early Stopping...")
    val_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.0234526884467792,
        depth=8,
        l2_leaf_reg=8.005912141245467,
        random_strength=0.07366560360327111,
        bagging_temperature=0.1459049142637005,
        border_count=254,
        loss_function='Logloss',
        eval_metric='Logloss',
        random_seed=42,
        od_type='Iter',
        od_wait=50,              # 조기 종료 50
        has_time=True,           # 시계열 누수 방지(과거 데이터로만 타겟 인코딩)
        allow_writing_files=False,
        verbose=100
    )
    
    val_model.fit(train_pool, eval_set=val_pool)
    
    best_iteration = val_model.get_best_iteration()
    print(f"\nBest iteration found: {best_iteration}")
    
    # 2024년 시즌 평가
    print("Predicting 2024 validation data...")
    val_preds = val_model.predict_proba(X_val)[:, 1]
    
    val_auc = roc_auc_score(y_val, val_preds)
    val_loss = log_loss(y_val, val_preds)
    
    brier_score = np.mean((val_preds - y_val)**2)
    r = np.mean(y_val)
    base_brier_score = r * (1 - r)
    brier_skill_score = max(0, 100000 * (1 - (brier_score / base_brier_score))) if base_brier_score > 0 else 0
    
    print("\n==================================")
    print("2024 Season Validation Results (CatBoost)")
    print(f"AUC: {val_auc:.4f}")
    print(f"LogLoss: {val_loss:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Brier Skill Score: {brier_skill_score:.2f}")
    print("==================================")
    
    # ---- 4. 최종 재학습 (Retraining on ALL Data) ----
    print("\nStarting Final Retraining on ENTIRE dataset (2019~2024)...")
    full_pool = Pool(X, y, cat_features=cat_features)
    
    # 조기종료 없이 best_iteration만큼만 학습
    final_model = CatBoostClassifier(
        iterations=best_iteration + 1,
        learning_rate=0.0234526884467792,
        depth=8,
        l2_leaf_reg=8.005912141245467,
        random_strength=0.07366560360327111,
        bagging_temperature=0.1459049142637005,
        border_count=254,
        loss_function='Logloss',
        random_seed=42,
        has_time=True,
        allow_writing_files=False,
        verbose=100
    )
    
    final_model.fit(full_pool)
    
    # 모델 저장 (테스트용)
    model_path = os.path.join(MODEL_DIR, "catboost_single.cbm")
    final_model.save_model(model_path)
    print(f"[SUCCESS] Saved {model_path}")
    
    # 마지막 모델의 Feature Importance 확인
    feature_importances = final_model.get_feature_importance()
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n[Feature Importance (ALL)]")
    print(importance_df.to_string(index=False))

if __name__ == "__main__":
    main()
