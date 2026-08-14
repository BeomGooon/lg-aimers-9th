import pandas as pd

df = pd.read_csv('data/train.csv')
print(df.shape[0])  # 행의 개수 출력
print(df.shape)     # (행 개수, 열 개수) 출력

dfs = pd.read_csv('data/trackman_history.csv')
print(dfs.shape[0])  # 행의 개수 출력
print(dfs.shape)     # (행 개수, 열 개수) 출력