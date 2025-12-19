# # 터미널에서 실행
# pip install pyarrow

import pandas as pd

# pyarrow 설치 후 실행
df = pd.read_csv('Data/criteo-uplift-v2.1.csv.gz')
df.to_parquet('Data/criteo-uplift-v2.1.parquet')

# 이후부터는
df = pd.read_parquet('Data/criteo-uplift-v2.1.parquet')