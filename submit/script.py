import os
import joblib
import pandas as pd
import numpy as np

ID_COL = "row_id"
TARGET_COL = "control_success"

# =======================
# 모델에 등록된 커스텀 전처리 함수
# (Pickle 객체를 성공적으로 불러오기 위해 동일하게 선언되어야 함)
# =======================
def preprocess_data_func(df):
    df = df.copy()
    
    df['balls_before'] = df['balls_before'] / 3.0
    df['strikes_before'] = df['strikes_before'] / 2.0
    df['outs_before'] = df['outs_before'] / 2.0
    df['score_diff_pitcher_team'] = df['score_diff_pitcher_team'].abs().clip(upper=5) / 5.0
    df['asof_batter_n'] = np.log10(df['asof_batter_n'] + 1).clip(upper=3) / 3.0
    df['asof_pitcher_n'] = np.log10(df['asof_pitcher_n'] + 1).clip(upper=3) / 3.0
    df['home_win_expectancy'] = (df['home_win_expectancy'] - 50).abs() / 50.0
    df['asof_pitcher_pitchmix_n'] = np.log10(df['asof_pitcher_pitchmix_n'] + 1).clip(upper=3) / 3.0
    df['inning'] = (df['inning'] - 1).clip(upper=8) / 8.0
    df['li'] = df['li'].clip(upper=2) / 2.0
    
    # ---- 파생 변수 (모멘텀/컨디션) 추가 ----
    df['pitcher_momentum_1_vs_5'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate']
    df['pitcher_condition_vs_baseline'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    
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
        'top_bottom', 'game_type', 'base_state'
    ]
    return df[use_features]

# =======================
# 데이터 로드 유틸
# =======================
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

def build_features(df):
    """Pipeline 내부에 전처리 및 인코딩 로직이 모두 있으므로 row_id만 빼고 반환."""
    return df.drop(columns=[ID_COL])

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

# =======================
# main
# =======================
def main():
    TEST_DIR = "./data"
    MODEL_DIR = "./model"
    OUT_DIR = "./output"
    
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    
    MODEL_PATH = os.path.join(MODEL_DIR, "xgb_ensemble.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load 5-Fold Ensemble models...")
    models = joblib.load(MODEL_PATH)
    print(f" OK. Loaded {len(models)} models successfully.")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test)

    print("Inference model (Ensemble Predict)...")
    ensemble_preds = np.zeros(len(X))
    
    if len(X) > 0:
        for i, model in enumerate(models):
            print(f"  Predicting with Fold {i+1}...")
            ensemble_preds += model.predict_proba(X)[:, 1]
        
        ensemble_preds /= len(models)
    
    print(f" preds={len(ensemble_preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, ensemble_preds)
    save_submission(OUT_PATH, sub)
    print(f"✅ Saved: {OUT_PATH} (rows={len(sub)})")

if __name__ == "__main__":
    main()
