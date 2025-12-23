import pandas as pd
import time

t0 = time.perf_counter()

df = pd.read_parquet('Data/criteo-uplift-v2.1.parquet')

t1 = time.perf_counter()

mem_gb = df.memory_usage(deep=True).sum() / (1024**3)

print(f"Load time : {t1 - t0:.2f} seconds")
print(f"DF memory : {mem_gb:.2f} GB")
