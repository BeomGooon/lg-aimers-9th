import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.base import clone

# =======================
# 1. 커스텀 전처리 로직
# =======================
def preprocess_data_func(df):
    df = df.copy()
    
    # ---- 새로 추가된 변수 (투타 매치업 및 볼카운트 압박감) ----
    df['same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
    df['count_advantage'] = (df['strikes_before'] - df['balls_before'] + 3) / 5.0
    
    # ---- 수식 기반 전처리 (수치형) ----
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
    # 결측치(NaN)는 억지로 채우지 않고 XGBoost가 스스로 판단하도록 그대로 둡니다.
    df['pitcher_momentum_1_vs_5'] = df['asof_pitcher_prev1_game_success_rate'] - df['asof_pitcher_prev5_game_success_rate']
    df['pitcher_condition_vs_baseline'] = df['asof_pitcher_prev3_game_success_rate'] - df['asof_pitcher_success_rate']
    
    # 사용할 모든 컬럼 반환 (범주형 인코딩은 파이프라인의 ColumnTransformer가 담당)
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
        'game_type'
    ]
    return df[use_features]

def main():
    TRAIN_PATH = "data/train.csv"
    MODEL_DIR = "model"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    
    TARGET_COL = "control_success"
    X = df  
    y = df[TARGET_COL]
    
    # ---- 2. 파이프라인 구성 요소 정의 ----
    numeric_features = [
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
        'asof_pitcher_fastball_rate'
    ]
    categorical_features = ['game_type']
    
    preprocessor = FunctionTransformer(preprocess_data_func, validate=False)
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    base_col_trans = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )
    
    # ---- 3. 엄격한 연도별 홀드아웃 검증 (2019~2023 학습, 2024 검증) ----
    print("\nStarting Strict Season-based Holdout Validation...")
    
    train_mask = X['season'] <= 2023
    val_mask = X['season'] == 2024
    
    train_idx = X.index[train_mask]
    val_idx = X.index[val_mask]
    
    print(f"Train samples (2019~2023): {len(train_idx)}")
    print(f"Validation samples (2024): {len(val_idx)}")
    
    models = []
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    col_trans_fold = clone(base_col_trans)
    
    prep_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('col_trans', col_trans_fold)
    ])
    
    print("Preprocessing data...")
    X_train_tf = prep_pipeline.fit_transform(X_train)
    X_val_tf = prep_pipeline.transform(X_val)
    
    # XGBClassifier: 극단적인 과적합 방지 셋업
    classifier = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,             # 트리를 아주 얕게 만들어 큰 흐름만 파악
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=300,    # 노이즈 차단 극대화 (아주 확실한 패턴만 채택)
        reg_alpha=0.1,           # L1 정규화 (불필요한 특징 무시)
        reg_lambda=1.0,          # L2 정규화 (극단적인 가중치 억제)
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=50,
        eval_metric="logloss"
    )
    
    print("Training model with Early Stopping...")
    classifier.fit(
        X_train_tf, y_train,
        eval_set=[(X_train_tf, y_train), (X_val_tf, y_val)], # 학습(Train)과 검증(Val) 점수 동시 비교
        verbose=100
    )
    
    # 안전장치: 조기 종료된 최적의 트리 개수로 고정 (오버피팅 원천 차단)
    if hasattr(classifier, 'best_iteration'):
        classifier.n_estimators = classifier.best_iteration + 1
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('col_trans', col_trans_fold),
        ('classifier', classifier)
    ])
    
    models.append(model) # 단일 모델이지만 앙상블 인터페이스(test_xgb.py) 유지를 위해 리스트로 감쌈
    
    # ---- 4. 2024년 시즌 평가 (OOF 대체) ----
    print("Predicting validation data...")
    val_preds = model.predict_proba(X_val)[:, 1]
    
    val_auc = roc_auc_score(y_val, val_preds)
    val_loss = log_loss(y_val, val_preds)
    
    brier_score = np.mean((val_preds - y_val)**2)
    r = np.mean(y_val)
    base_brier_score = r * (1 - r)
    brier_skill_score = max(0, 100000 * (1 - (brier_score / base_brier_score))) if base_brier_score > 0 else 0
    
    print("\n==================================")
    print("2024 Season Validation Results")
    print(f"AUC: {val_auc:.4f}")
    print(f"LogLoss: {val_loss:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print(f"Brier Skill Score: {brier_skill_score:.2f}")
    print("==================================")
    
    # ---- 5. 모델 저장 ----
    model_path = os.path.join(MODEL_DIR, "xgb_ensemble.pkl")
    joblib.dump(models, model_path)
    print(f"\n[SUCCESS] Holdout Model saved to {model_path} (wrapped in a list)")

if __name__ == "__main__":
    main()
