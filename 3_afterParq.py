import pandas as pd
import time

# 시작 시간 기록
start_time = time.time()

# 2단계: 이후부터는 - Parquet로 빠르게 로드
df = pd.read_parquet('Data/criteo-uplift-v2.1.parquet')

# 3단계: DataFrame 확인
print("=" * 50)
print("1. 기본 정보")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print("\n" + "=" * 50)
print("2. 처음 5행")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("3. 데이터 타입 및 결측치")
print("=" * 50)
df.info()

print("\n" + "=" * 50)
print("4. 기술통계")
print("=" * 50)
print(df.describe())

print("\n" + "=" * 50)
print("5. 각 컬럼별 유니크 값 개수")
print("=" * 50)
print(df.nunique())

print("\n" + "=" * 50)
print("6. 주요 변수 분포 (uplift 데이터)")
print("=" * 50)
if 'treatment' in df.columns:
    print("\nTreatment 분포:")
    print(df['treatment'].value_counts())
if 'conversion' in df.columns:
    print("\nConversion 분포:")
    print(df['conversion'].value_counts())
if 'visit' in df.columns:
    print("\nVisit 분포:")
    print(df['visit'].value_counts())

# 추가적으로 분석
print(df.shape)


# 종료 시간 기록 및 출력
end_time = time.time()
elapsed_time = end_time - start_time

print("\n" + "=" * 50)
print("실행 시간")
print("=" * 50)
print(f"총 실행 시간: {elapsed_time:.2f}초")
print(f"총 실행 시간: {elapsed_time:.4f}초 (상세)")