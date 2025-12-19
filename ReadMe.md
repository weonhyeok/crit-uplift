# Criteo Uplift Modeling Dataset Analysis

Criteo에서 제공하는 uplift modeling 공개 데이터셋을 활용한 증분 효과(incremental effect) 분석 프로젝트

## 📊 데이터셋 개요

- **출처**: Criteo Research (프랑스 디지털 광고 회사)
- **목적**: 광고 캠페인의 실제 증분 효과 측정
- **데이터 규모**: 
  - 총 행(Rows): 13,979,592개
  - 총 열(Columns): 16개
  - 파일 크기: 296MB (압축), 메모리 사용량: 1.7GB
- **활용**: Uplift modeling, CATE 추정, 개인화 타겟팅 전략

## 📋 데이터 구조

### 주요 변수

| 변수 | 설명 | 분포 |
|------|------|------|
| `treatment` | 처치 여부 (광고 노출) | 0: 2,096,937 / 1: 11,882,655 |
| `conversion` | 전환 여부 (구매/클릭) | 0: 13,938,818 / 1: 40,774 (**0.29%**) |
| `visit` | 방문 여부 | 0: 13,322,663 / 1: 656,929 (**4.7%**) |
| `exposure` | 노출 여부 | 평균: 3.06% |
| `f0-f11` | 익명화된 사용자 특성 | 12개의 feature 변수 |

### 주요 통계

- **전환율(Conversion Rate)**: 0.29%
- **방문율(Visit Rate)**: 4.7%
- **처치군 비율**: 85% (treatment=1)
- **대조군 비율**: 15% (treatment=0)

## 🗂️ 프로젝트 구조
```
crit-uplift/
├── Data/                              # 데이터 폴더 (gitignore)
│   ├── criteo-uplift-v2.1.csv.gz     # 원본 압축 데이터
│   └── criteo-uplift-v2.1.parquet    # 변환된 Parquet 파일
├── 1_test.py                          # 데이터 로드 테스트
├── 2_parq.py                          # Parquet 변환
├── 3_afterParq.py                     # 데이터 탐색 분석
├── .gitignore
└── README.md
```

## 🚀 시작하기

### 필수 라이브러리 설치
```bash
pip install pandas pyarrow
```

### 📥 데이터 다운로드

데이터 파일은 용량이 커서 레포지토리에 포함되어 있지 않습니다.

1. [Criteo AI Lab](https://ailab.criteo.com/ressources/)에서 `criteo-uplift-v2.1.csv.gz` 다운로드
2. 프로젝트 루트에 `Data/` 폴더 생성
3. 다운로드한 파일을 `Data/` 폴더에 저장

### 데이터 준비 및 로드

**1단계: Parquet 포맷으로 변환** (처음 한 번만 실행)
```python
import pandas as pd

df = pd.read_csv('Data/criteo-uplift-v2.1.csv.gz')
df.to_parquet('Data/criteo-uplift-v2.1.parquet')
```

**2단계: 이후 빠른 로드**
```python
df = pd.read_parquet('Data/criteo-uplift-v2.1.parquet')
print(f"Shape: {df.shape}")  # (13979592, 16)
```

## ⚡ 성능 최적화

### 로딩 속도 비교

| 포맷 | 로딩 시간 | 권장도 |
|------|----------|--------|
| CSV (압축 .gz) | ~10-15초 | ⭐ |
| CSV (비압축) | ~5-8초 | ⭐⭐ |
| **Parquet** | **~1-2초** | ⭐⭐⭐ |

### 메모리 최적화 팁
```python
# 필요한 컬럼만 선택
df = pd.read_parquet('Data/criteo-uplift-v2.1.parquet',
                     columns=['treatment', 'conversion', 'visit', 'exposure'])

# 데이터 타입 최적화
df['treatment'] = df['treatment'].astype('int8')
df['conversion'] = df['conversion'].astype('int8')
df['visit'] = df['visit'].astype('int8')
df['exposure'] = df['exposure'].astype('int8')
```

## 📈 주요 분석 예시

### 기본 전환율 분석
```python
# 처치군 vs 대조군 전환율
treatment_cvr = df[df['treatment']==1]['conversion'].mean()
control_cvr = df[df['treatment']==0]['conversion'].mean()

print(f"처치군 전환율: {treatment_cvr:.4%}")
print(f"대조군 전환율: {control_cvr:.4%}")
print(f"증분 효과: {treatment_cvr - control_cvr:.4%}")
```

### 방문자 전환율
```python
# 방문자 중 전환율
visit_df = df[df['visit']==1]
visit_cvr = visit_df['conversion'].mean()
print(f"방문자 전환율: {visit_cvr:.4%}")
```

## 💡 활용 사례

- **Uplift Modeling**: 광고 효과가 높은 사용자 식별
- **CATE 추정**: 조건부 평균 처치 효과 (Conditional Average Treatment Effect) 분석
- **A/B Testing**: 실험 설계 및 효과 분석
- **Causal Inference**: 인과 추론 방법론 적용 (DiD, IV, RDD)
- **개인화 마케팅**: 타겟팅 전략 수립

## 🔧 기술 스택

- Python 3.12
- pandas
- pyarrow
- numpy (분석용)
- scikit-learn (모델링용)

## 📚 참고 자료

- [Criteo AI Lab Research](https://ailab.criteo.com/)
- [Uplift Modeling Paper](https://arxiv.org/abs/1804.10219)
- [Causal Inference Methods](https://www.econometrics-with-r.org/)

## 👤 Author

**Marvin**

---

⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!