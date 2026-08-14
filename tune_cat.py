import os
import numpy as np
import pandas as pd
import optuna
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

def main():
    TRAIN_PATH = "data/train.csv"
    
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    TARGET_COL = "control_success"
    y = df[TARGET_COL]
    
    print("Preprocessing data...")
    X, cat_features = preprocess_data_func(df)
    
    print("\nSplitting Data for Tuning (Train: 2019~2023, Val: 2024)...")
    train_mask = df['season'] <= 2023
    val_mask = df['season'] == 2024
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    def objective(trial):
        # 튜닝할 하이퍼파라미터 공간 정의
        params = {
            'iterations': 1000, # 튜닝 속도를 위해 최대 1000번까지만 (Early Stopping 사용)
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_categorical('border_count', [128, 254]),
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 50,
            'has_time': True,
            'verbose': 0 # Optuna 출력창이 너무 지저분해지지 않게 0으로 설정
        }

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)
        
        # 성능 평가 (Brier Skill Score 기준 최적화)
        val_preds = model.predict_proba(X_val)[:, 1]
        
        brier_score = np.mean((val_preds - y_val)**2)
        r = np.mean(y_val)
        base_brier_score = r * (1 - r)
        brier_skill_score = max(0, 100000 * (1 - (brier_score / base_brier_score))) if base_brier_score > 0 else 0
        
        return brier_skill_score

    print("\nStarting Optuna Hyperparameter Tuning...")
    # Brier Skill Score는 높을수록 좋으므로 'maximize'
    study = optuna.create_study(direction='maximize', study_name="CatBoost_Optimization")
    
    # 너무 오래 걸릴 수 있으므로 30번의 Trial만 수행
    study.optimize(objective, n_trials=30)
    
    print("\n==================================")
    print("[Optimization Completed]")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Brier Skill Score: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print("==================================")
    print("이 파라미터들을 복사해서 train_cat.py의 모델 파라미터에 적용하세요!")

if __name__ == "__main__":
    main()
