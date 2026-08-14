import pandas as pd
import numpy as np

def extract_mapping():
    print("Loading train.csv...")
    train = pd.read_csv('data/train.csv', usecols=[
        'row_id', 'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom', 
        'balls_before', 'strikes_before', 'outs_before', 'pitcher_id', 'batter_id',
        'pitcher_hand', 'batter_hand'
    ])
    
    print("Loading trackman_history.csv...")
    hist = pd.read_csv('data/trackman_history.csv', usecols=[
        'trackman_id', 'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom', 
        'balls_before', 'strikes_before', 'outs_before', 'pitcher_trackman_id', 'batter_trackman_id',
        'pitcher_hand', 'batter_hand'
    ])
    
    # 1. 키 정규화
    hist['top_bottom'] = hist['top_bottom'].apply(lambda x: str(x)[0] if pd.notnull(x) else x)
    
    # Hand 정규화 (1: Left, 2: Right)
    hand_map = {1: 'Left', 2: 'Right', 3: 'Switch'}
    train['pitcher_hand'] = train['pitcher_hand'].map(hand_map).fillna(train['pitcher_hand'])
    train['batter_hand'] = train['batter_hand'].map(hand_map).fillna(train['batter_hand'])
    
    join_keys = [
        'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom',
        'balls_before', 'strikes_before', 'outs_before',
        'pitcher_hand', 'batter_hand'
    ]
    
    for col in join_keys:
        train[col] = train[col].astype(str)
        hist[col] = hist[col].astype(str)
        
    print("Merging on situational keys...")
    # 교차 조인 (메모리 폭발 방지를 위해 inner join)
    merged = pd.merge(train, hist, on=join_keys, how='inner')
    
    print("Extracting Pitcher Mapping...")
    # 투수 ID 매핑
    p_counts = merged.groupby(['pitcher_id', 'pitcher_trackman_id']).size().reset_index(name='count')
    # 각 pitcher_id별로 가장 매칭 횟수가 많은 trackman_id 선택
    p_best = p_counts.sort_values('count', ascending=False).drop_duplicates('pitcher_id', keep='first')
    
    # 신뢰도 계산 (가장 많은 매칭 횟수가 압도적인지)
    p_total = p_counts.groupby('pitcher_id')['count'].sum().reset_index(name='total')
    p_best = pd.merge(p_best, p_total, on='pitcher_id')
    p_best['confidence'] = p_best['count'] / p_best['total']
    
    print("Extracting Batter Mapping...")
    # 타자 ID 매핑
    b_counts = merged.groupby(['batter_id', 'batter_trackman_id']).size().reset_index(name='count')
    b_best = b_counts.sort_values('count', ascending=False).drop_duplicates('batter_id', keep='first')
    
    b_total = b_counts.groupby('batter_id')['count'].sum().reset_index(name='total')
    b_best = pd.merge(b_best, b_total, on='batter_id')
    b_best['confidence'] = b_best['count'] / b_best['total']
    
    # 저장
    p_best.to_csv('pitcher_mapping.csv', index=False)
    b_best.to_csv('batter_mapping.csv', index=False)
    
    print("\n--- PITCHER MAPPING SUMMARY ---")
    print(f"Total pitcher_ids mapped: {len(p_best)} / {train['pitcher_id'].nunique()}")
    print(f"Average confidence: {p_best['confidence'].mean():.2%}")
    
    print("\n--- BATTER MAPPING SUMMARY ---")
    print(f"Total batter_ids mapped: {len(b_best)} / {train['batter_id'].nunique()}")
    print(f"Average confidence: {b_best['confidence'].mean():.2%}")
    
    print("\nFiles saved: pitcher_mapping.csv, batter_mapping.csv")

if __name__ == '__main__':
    extract_mapping()
