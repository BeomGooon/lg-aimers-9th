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
    
    # ---- 새로 추가된 변수 (투타 매치업) ----
    df['same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
    
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
        'asof_pitcher_middle_rate',
        'asof_pitcher_reverse_rate',
        'asof_pitcher_ball_rate',
        'asof_pitcher_fastball_rate',
        'top_bottom', 'game_type', 'base_state'
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
        'asof_pitcher_middle_rate',
        'asof_pitcher_reverse_rate',
        'asof_pitcher_ball_rate',
        'asof_pitcher_fastball_rate'
    ]
    categorical_features = ['top_bottom', 'game_type', 'base_state']
    
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
    
    # ---- 3. 시계열 교차 검증 (앙상블) 설정 ----
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    models = []
    oof_preds = np.zeros(len(X))
    val_indices = []
    
    print(f"Starting {n_splits}-Fold Time-Series Cross Validation Training...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold+1} / {n_splits} ---")
        
        val_indices.extend(val_idx)
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 🚨 핵심 버그 픽스: 각 Fold마다 독립적인 ColumnTransformer 객체를 복제(clone)해서 사용!
        col_trans_fold = clone(base_col_trans)
        
        prep_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('col_trans', col_trans_fold)
        ])
        
        print("Preprocessing data for fold...")
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
            eval_set=[(X_val_tf, y_val)],
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
        
        models.append(model)
        
        print("Predicting validation data...")
        fold_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = fold_preds
        
        fold_auc = roc_auc_score(y_val, fold_preds)
        fold_loss = log_loss(y_val, fold_preds)
        print(f"Fold {fold+1} - AUC: {fold_auc:.4f} | LogLoss: {fold_loss:.4f}")

    # ---- 4. 전체 OOF 평가 ----
    print("\n==================================")
    print("Overall Out-Of-Fold Validation Results")
    
    # TimeSeriesSplit 특성상 검증에 사용되지 않은 초기 데이터가 존재하므로,
    # 실제로 검증된 데이터(val_indices)만 추려서 점수 계산
    valid_mask = np.zeros(len(X), dtype=bool)
    valid_mask[val_indices] = True
    
    y_valid = y[valid_mask]
    oof_valid = oof_preds[valid_mask]
    
    overall_auc = roc_auc_score(y_valid, oof_valid)
    overall_loss = log_loss(y_valid, oof_valid)
    
    brier_score = np.mean((oof_valid - y_valid)**2)
    r = np.mean(y_valid)
    base_brier_score = r * (1 - r)
    brier_skill_score = max(0, 100000 * (1 - (brier_score / base_brier_score))) if base_brier_score > 0 else 0
    
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Overall LogLoss: {overall_loss:.4f}")
    print(f"Overall Brier Score: {brier_score:.4f}")
    print(f"Overall Brier Skill Score: {brier_skill_score:.2f}")
    print("==================================")
    
    # ---- 5. 모델 저장 ----
    model_path = os.path.join(MODEL_DIR, "xgb_ensemble.pkl")
    joblib.dump(models, model_path)
    print(f"\n[SUCCESS] 5-Fold Ensemble Models saved to {model_path}")

if __name__ == "__main__":
    main()
