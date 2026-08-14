import pandas as pd
import numpy as np

def check_mapping():
    print("Loading train.csv...")
    train = pd.read_csv('data/train.csv', usecols=[
        'row_id', 'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom', 
        'balls_before', 'strikes_before', 'outs_before', 
        'pitcher_hand', 'batter_hand', 'pitcher_id', 'batter_id'
    ])
    
    print("Loading trackman_history.csv...")
    hist = pd.read_csv('data/trackman_history.csv', usecols=[
        'trackman_id', 'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom', 
        'balls_before', 'strikes_before', 'outs_before', 
        'pitcher_hand', 'batter_hand', 'pitcher_trackman_id', 'batter_trackman_id'
    ])
    
    # 조인 키 설정 (hand 제외)
    join_keys = [
        'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
        'balls_before', 'strikes_before', 'outs_before'
    ]
    
    print(f"\nTrain size: {len(train)}")
    print(f"Hist size: {len(hist)}")
    
    hist['top_bottom'] = hist['top_bottom'].apply(lambda x: str(x)[0] if pd.notnull(x) else x)
    
    # 타입 통일 (문자열)
    for col in join_keys:
        train[col] = train[col].astype(str)
        hist[col] = hist[col].astype(str)
        
    train_unique = train.drop_duplicates(subset=join_keys, keep=False)
    hist_unique = hist.drop_duplicates(subset=join_keys, keep=False)
        
    # 이 상황 피처만으로 고유하게 1:1 매칭되는 행이 있는지 조인
    matched = pd.merge(train_unique, hist_unique, on=join_keys, how='inner')
    print(f"\nExact 1:1 matched rows using basic situational keys: {len(matched)}")
    
    if len(matched) > 0:
        print("\n[ID Mapping Example based on Exact Matches]")
        mapping_sample = matched[['pitcher_id', 'pitcher_trackman_id']].drop_duplicates().head(10)
        print(mapping_sample)
        
        # 투수 ID 매핑 테이블 생성 시도
        pitcher_mapping = matched[['pitcher_id', 'pitcher_trackman_id']].drop_duplicates()
        print(f"\nUnique pitcher_id matched: {pitcher_mapping['pitcher_id'].nunique()} / {train['pitcher_id'].nunique()}")

if __name__ == '__main__':
    check_mapping()
