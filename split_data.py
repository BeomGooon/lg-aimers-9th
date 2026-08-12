import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv_data(input_path, train_out_path, test_out_path, test_size=0.2, random_seed=42):
    """
    CSV 파일을 읽어 학습용 데이터와 테스트용 데이터로 랜덤 분할하고 저장합니다.
    """
    print(f"데이터를 불러오는 중: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"전체 데이터 개수: {len(df)}행")
    
    # 층화추출(Stratified)을 위한 타겟 컬럼 이름 (필요시 변경)
    TARGET_COL = "control_success"
    
    # 데이터 랜덤 분할
    if TARGET_COL in df.columns:
        print(f"타겟 변수('{TARGET_COL}')의 비율을 유지하며 분할합니다. (층화추출)")
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_seed, 
            stratify=df[TARGET_COL]
        )
    else:
        print("순수 랜덤으로 데이터를 분할합니다.")
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_seed
        )
        
    print(f"학습 데이터(Train) 개수: {len(train_df)}행")
    print(f"테스트 데이터(Test) 개수: {len(test_df)}행")
    
    # 저장할 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(train_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_out_path), exist_ok=True)
    
    # 결과 저장
    print("분할된 데이터를 저장하는 중...")
    train_df.to_csv(train_out_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_out_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 저장 완료!\nTrain: {train_out_path}\nTest : {test_out_path}")

if __name__ == "__main__":
    # 파일 경로 설정 (현재 프로젝트 구조 기준)
    INPUT_CSV = "data/train.csv"
    TRAIN_CSV = "data/train_split.csv"
    TEST_CSV = "data/val_split.csv"
    
    split_csv_data(INPUT_CSV, TRAIN_CSV, TEST_CSV)
