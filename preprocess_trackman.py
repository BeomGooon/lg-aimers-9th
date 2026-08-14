import pandas as pd
import numpy as np

def get_trackman_stats(trackman_path, mapping_path):
    """
    trackman_history.csv와 pitcher_mapping.csv를 읽어
    투수별/시즌별 평균 및 표준편차 물리 지표를 계산합니다.
    """
    print("  [Trackman] Loading trackman history and mapping...")
    hist = pd.read_csv(trackman_path, usecols=[
        'pitcher_trackman_id', 'season', 'extension', 'rel_side', 'rel_height'
    ])
    p_map = pd.read_csv(mapping_path)
    
    # 조인을 위해 타입 통일
    hist['pitcher_trackman_id'] = hist['pitcher_trackman_id'].astype(str)
    p_map['pitcher_trackman_id'] = p_map['pitcher_trackman_id'].astype(str)
    
    print("  [Trackman] Aggregating features by pitcher and season...")
    # 투수 x 시즌별 폼 안정성(표준편차) 계산
    agg_funcs = {
        'extension': ['std'],
        'rel_side': ['std'],
        'rel_height': ['std']
    }
    
    stats = hist.groupby(['pitcher_trackman_id', 'season']).agg(agg_funcs).reset_index()
    
    # 멀티인덱스 컬럼 단일화
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    # pitcher_id 매핑
    stats = pd.merge(stats, p_map[['pitcher_id', 'pitcher_trackman_id']], on='pitcher_trackman_id', how='inner')
    
    # 2019년 데이터에 대한 예외 처리: 2019년 데이터를 2018년 데이터인 것처럼 복제하여 추가 (2019년 train에 매핑되도록)
    stats_2019 = stats[stats['season'] == 2019].copy()
    stats_2019['season'] = 2018
    stats = pd.concat([stats, stats_2019], ignore_index=True)
    
    # N-1년도 데이터를 N년도로 적용하기 위해 season + 1 처리
    # 예: 2018년에 세운 기록(복제본)은 2019년 평가에, 2019년에 세운 기록은 2020년 평가에 사용
    stats['season'] = stats['season'] + 1
    
    stats.drop(columns=['pitcher_trackman_id'], inplace=True)
    return stats

def append_trackman_features(df, trackman_path='data/trackman_history.csv', mapping_path='pitcher_mapping.csv'):
    """
    메인 데이터프레임(train/test)에 N-1년도 Trackman 통계를 병합합니다.
    """
    stats = get_trackman_stats(trackman_path, mapping_path)
    
    # 병합 전 컬럼명에 접두사 추가 (충돌 방지 및 식별 용이)
    rename_dict = {c: f"prev_yr_{c}" for c in stats.columns if c not in ['pitcher_id', 'season']}
    stats.rename(columns=rename_dict, inplace=True)
    
    print("  [Trackman] Merging features into main dataset...")
    # Left Join: 신인 투수나 전년도 데이터가 없는 경우 NaN 발생 (CatBoost가 자동 처리)
    df = pd.merge(df, stats, on=['pitcher_id', 'season'], how='left')
    
    return df
