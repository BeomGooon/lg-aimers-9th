import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

ID_COL = "row_id"
TARGET_COL = "control_success"

def preprocess_data_func(df):
    df = df.copy()
    
    # ---- 새로 추가된 변수 (투타 매치업 및 볼카운트 압박감) ----
    df['same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
    df['count_advantage'] = (df['strikes_before'] - df['balls_before'] + 3) / 5.0
    
    # ---- 시간 변수 추가 (Season, Month) ----
    # season은 그대로 수치형으로 사용
    # game_month는 주기를 갖도록 sin/cos 변환 (보통 야구는 3~11월)
    df['game_month_sin'] = np.sin(2 * np.pi * df['game_month'] / 12.0)
    df['game_month_cos'] = np.cos(2 * np.pi * df['game_month'] / 12.0)
    
    # ---- 수식 기반 전처리 (수치형) ----
    df['balls_before'] = df['balls_before'] / 3.0
    df['strikes_before'] = df['strikes_before'] / 2.0
    df['outs_before'] = df['outs_before'] / 2.0
    df['score_diff_pitcher_team'] = df['score_diff_pitcher_team'].abs().clip(upper=5) / 5.0
    df['asof_batter_n'] = np.log1p(df['asof_batter_n'])
    df['asof_pitcher_n'] = np.log1p(df['asof_pitcher_n'])
    df['home_win_expectancy'] = (df['home_win_expectancy'] - 50).abs() / 50.0
    df['asof_pitcher_pitchmix_n'] = np.log1p(df['asof_pitcher_pitchmix_n'])
    df['inning'] = (df['inning'] - 1).clip(upper=8) / 8.0
    df['li'] = df['li'].clip(upper=2) / 2.0
    
    # ---- 파생 변수 (모멘텀/컨디션) 추가 ----
    df['pitcher_momentum_1_vs_5'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate']
    df['pitcher_condition_vs_baseline'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    
    # ---- 범주형 변수 결측치 처리 및 문자열 캐스팅 ----
    cat_cols = ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id', 'pitcher_hand', 'batter_hand', 'game_type']
    for col in cat_cols:
        df[col] = df[col].fillna('MISSING').astype(str)
        
    use_features = [
        'balls_before', 'strikes_before', 'outs_before', 
        'score_diff_pitcher_team', 'runner_on_1b', 'runner_on_2b', 'runner_on_3b',
        'asof_batter_n', 'asof_pitcher_success_rate', 'asof_pitcher_n',
        'asof_pitcher_strike_rate', 'home_win_expectancy', 
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
        'game_month_sin',
        'game_month_cos'
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
    
    MODEL_PATH = os.path.join(MODEL_DIR, "catboost_final.cbm")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load CatBoost Model...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print(" OK. Loaded single final model successfully.")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    X, cat_features = preprocess_data_func(test)
    
    test_pool = Pool(X, cat_features=cat_features)

    print("Inference model (CatBoost Predict)...")
    preds = model.predict_proba(test_pool)[:, 1]
    
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"[SUCCESS] Saved: {OUT_PATH} (rows={len(sub)})")

if __name__ == "__main__":
    main()
