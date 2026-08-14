import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

import train_cat_single

def get_brier_skill_score(y_true, y_prob):
    brier_score = np.mean((y_prob - y_true)**2)
    brier_ref = np.mean((y_true.mean() - y_true)**2)
    return (1.0 - (brier_score / brier_ref)) * 1000

def main():
    print("Loading data...")
    df = pd.read_csv('data/train.csv')
    df['control_success'] = df['control_success'].fillna(0).astype(int)
    
    # We will test adding these features back one by one to the base features
    features_to_test = [
        'pitcher_id',
        'batter_id',
        'top_bottom',
        'score_diff_pitcher_team',
        'asof_batter_n'
    ]
    
    results = {}
    
    for test_feature in features_to_test:
        print(f"\n==========================================")
        print(f"Testing addition of feature: {test_feature}")
        print(f"==========================================")
        
        # 1. Preprocess using the base function
        processed_df, base_cat_cols = train_cat_single.preprocess_data_func(df)
        use_features = list(processed_df.columns)
        
        # 2. Add the test feature back if it's not already in the processed dataframe
        # Some require preprocessing, some don't.
        temp_df = df.copy()
        
        # Apply preprocessing rules for the test features if needed
        if test_feature == 'score_diff_pitcher_team':
            temp_df['score_diff_pitcher_team'] = temp_df['score_diff_pitcher_team'].abs().clip(upper=5) / 5.0
        elif test_feature == 'asof_batter_n':
            temp_df['asof_batter_n'] = np.log1p(temp_df['asof_batter_n'])
        elif test_feature in ['pitcher_id', 'batter_id', 'top_bottom']:
            temp_df[test_feature] = temp_df[test_feature].fillna('MISSING').astype(str)
            
        processed_df[test_feature] = temp_df[test_feature]
        use_features.append(test_feature)
        
        cat_cols = list(base_cat_cols)
        if test_feature in ['pitcher_id', 'batter_id', 'top_bottom']:
            cat_cols.append(test_feature)
            
        # 3. Split data
        train_mask = (processed_df['season'] >= 2019) & (processed_df['season'] <= 2023)
        valid_mask = (processed_df['season'] == 2024)
        
        X_train = processed_df.loc[train_mask, use_features]
        y_train = df.loc[train_mask, 'control_success']
        X_valid = processed_df.loc[valid_mask, use_features]
        y_valid = df.loc[valid_mask, 'control_success']
        
        # 4. Train Model
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            eval_metric='Logloss',
            random_seed=42,
            cat_features=cat_cols,
            verbose=100,
            early_stopping_rounds=50,
            task_type="CPU",
            thread_count=-1
        )
        
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
        
        # 5. Evaluate
        preds = model.predict_proba(X_valid)[:, 1]
        bss = get_brier_skill_score(y_valid.values, preds)
        print(f"\n[Result] +{test_feature} -> BSS: {bss:.2f}")
        results[test_feature] = bss
        
    print("\n\n==========================================")
    print("FINAL ABLATION RESULTS (Base Score ~ 807.21)")
    print("==========================================")
    for f, score in results.items():
        print(f"+ {f.ljust(25)} : {score:.2f}")

if __name__ == "__main__":
    main()
