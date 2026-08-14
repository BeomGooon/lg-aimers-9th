import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

ID_COL = "row_id"
TARGET_COL = "control_success"

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
    cat_cols = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id', 'pitcher_hand', 'batter_hand', 'game_type', 'top_bottom', 'base_out_state']
    for col in cat_cols:
        df[col] = df[col].fillna('MISSING').astype(str)
        
    use_features = [
        'balls_before', 'strikes_before', 
        'score_diff_pitcher_team', 
        'asof_batter_n', 'asof_pitcher_success_rate', 'asof_pitcher_n',
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
        'asof_pitcher_reverse_rate',
        'asof_pitcher_ball_rate',
        'asof_pitcher_fastball_rate',
        'season',
        'game_month',
        're24'
    ] + cat_cols
    
    return df[use_features], cat_cols

def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df

def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df

def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub

def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")

def main():
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")
    
    print("Load CatBoost Models for Ensemble...")
    SEEDS = [42, 43, 44, 45, 46]
    models = []
    for seed in SEEDS:
        model_path = os.path.join(MODEL_DIR, f"catboost_seed_{seed}.cbm")
        model = CatBoostClassifier()
        model.load_model(model_path)
        models.append(model)
    print(f" OK. Loaded {len(models)} models for ensembling.")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    X, cat_features = preprocess_data_func(test)
    
    test_pool = Pool(X, cat_features=cat_features)

    print("Inference model (CatBoost Predict with Seed Averaging)...")
    preds_list = []
    for idx, model in enumerate(models):
        print(f" -> Predicting with Model {idx+1}/{len(models)}...")
        preds = model.predict_proba(test_pool)[:, 1]
        preds_list.append(preds)
        
    final_preds = np.mean(preds_list, axis=0)
    
    print(f" final_preds={len(final_preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, final_preds)
    save_submission(OUT_PATH, sub)
    print(f"[SUCCESS] Saved: {OUT_PATH} (rows={len(sub)})")

if __name__ == "__main__":
    main()
