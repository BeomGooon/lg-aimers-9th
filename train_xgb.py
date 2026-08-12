import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

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
    df['asof_pitcher_success_rate'] = df['asof_pitcher_success_rate'].fillna(0.5)
    df['asof_pitcher_n'] = np.log10(df['asof_pitcher_n'] + 1).clip(upper=3) / 3.0
    df['asof_pitcher_strike_rate'] = df['asof_pitcher_strike_rate'].fillna(0.6)
    df['home_win_expectancy'] = (df['home_win_expectancy'] - 50).abs() / 50.0
    df['asof_pitcher_breaking_rate'] = df['asof_pitcher_breaking_rate'].fillna(0.3)
    df['asof_pitcher_offspeed_rate'] = df['asof_pitcher_offspeed_rate'].fillna(0.15)
    df['asof_batter_success_rate'] = df['asof_batter_success_rate'].fillna(0.5)
    
    # 사용할 모든 컬럼 반환 (범주형 인코딩은 파이프라인의 ColumnTransformer가 담당)
    use_features = [
        'balls_before', 'strikes_before', 'outs_before', 
        'score_diff_pitcher_team', 'runner_on_1b', 'runner_on_2b', 'runner_on_3b',
        'asof_batter_n', 'asof_pitcher_success_rate', 'asof_pitcher_n',
        'asof_pitcher_strike_rate', 'home_win_expectancy', 
        'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate',
        'asof_batter_success_rate',
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
        'asof_batter_success_rate'
    ]
    categorical_features = ['top_bottom', 'game_type', 'base_state']
    
    preprocessor = FunctionTransformer(preprocess_data_func, validate=False)
    
    # 범주형 데이터를 안전하게 변환하는 인코더 (Test 데이터에 처음 보는 카테고리가 와도 에러 안남)
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    col_trans = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )
    
    # ---- 3. 5-Fold 교차 검증 (앙상블) 설정 ----
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = []
    oof_preds = np.zeros(len(X))
    
    print(f"Starting {n_splits}-Fold Cross Validation Training...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1} / {n_splits} ---")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 모델 파이프라인 (추천해드린 하이퍼파라미터 적용)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('col_trans', col_trans),
            ('classifier', XGBClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=50,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'
            ))
        ])
        
        print("Training model...")
        model.fit(X_train, y_train)
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
    
    overall_auc = roc_auc_score(y, oof_preds)
    overall_loss = log_loss(y, oof_preds)
    
    brier_score = np.mean((oof_preds - y)**2)
    r = np.mean(y)
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
    print(f"\n✅ 5-Fold Ensemble Models saved to {model_path}")

if __name__ == "__main__":
    main()
