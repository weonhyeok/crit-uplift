# Criteo Uplift Modeling Dataset Analysis

Criteo에서 제공하는 **Uplift Modeling 공개 데이터셋**을 활용하여,
광고 노출이 실제로 만들어내는 **증분 전환 효과 (Incremental Conversion)**를 추정하고
이를 **타겟팅 및 예산 최적화 의사결정**으로 연결하는 프로젝트입니다.

> 핵심 질문:
> **"누구에게 광고를 보여줘야 실제 매출이 증가하는가?"**

---

## 📊 데이터셋 개요

* **출처**: Criteo Research (프랑스 디지털 광고 회사)
* **목적**: 광고 캠페인의 *실제 증분 효과* 측정
* **데이터 규모**:

  * 총 행(Rows): **13,979,592**
  * 총 열(Columns): **16**
  * 파일 크기: 296MB (압축), 메모리 사용량: ~1.7GB
* **활용 영역**:

  * Uplift Modeling (X-/DR-Learner)
  * CATE (Conditional Average Treatment Effect)
  * 개인화 광고 타겟팅
  * Budget / ROI Optimization

---

## 📋 데이터 구조

### 주요 변수

| 변수           | 설명          | 분포                                    |
| ------------ | ----------- | ------------------------------------- |
| `treatment`  | 광고 노출 여부    | 0: 2,096,937 / 1: 11,882,655          |
| `conversion` | 전환 여부       | 0: 13,938,818 / 1: 40,774 (**0.29%**) |
| `visit`      | 방문 여부       | 0: 13,322,663 / 1: 656,929 (**4.7%**) |
| `exposure`   | 광고 노출 비율    | 평균 3.06%                              |
| `f0`–`f11`   | 익명화된 사용자 특성 | 12개 feature                           |

### 데이터 특성 요약

* **극단적 희소 이벤트**: CVR ≈ **0.29%**
* **처치 불균형**: Treatment 비율 **85%**
* **Uplift modeling에 현실적인 난이도**를 가진 산업용 데이터

---

## 🗂️ 프로젝트 구조

```
crit-uplift/
├── Data/                              # 데이터 폴더 (gitignore)
│   ├── criteo-uplift-v2.1.csv.gz     # 원본 압축 데이터
│   └── criteo-uplift-v2.1.parquet    # Parquet 변환 데이터
├── 1_test.py                          # 데이터 로드 테스트
├── 2_parq.py                          # Parquet 변환
├── 3_afterParq.py                     # EDA
├── 7_2_uplift_qini.py                 # X-Learner + Qini 평가
├── 7_3_uplift_segment.py              # SHAP 기반 uplift 해석
├── .gitignore
└── README.md
```

---

## 🚀 시작하기

### 필수 라이브러리

```bash
pip install pandas pyarrow numpy scikit-learn lightgbm shap matplotlib
```

---

## 📥 데이터 다운로드

데이터 용량이 커 GitHub에는 포함되어 있지 않습니다.

1. [Criteo AI Lab](https://ailab.criteo.com/ressources/) 접속
2. `criteo-uplift-v2.1.csv.gz` 다운로드
3. 프로젝트 루트에 `Data/` 폴더 생성 후 저장

---

## ⚙️ 데이터 준비

### 1️⃣ Parquet 변환 (최초 1회)

```python
import pandas as pd

df = pd.read_csv('Data/criteo-uplift-v2.1.csv.gz')
df.to_parquet('Data/criteo-uplift-v2.1.parquet')
```

### 2️⃣ 이후 빠른 로딩

```python
df = pd.read_parquet('Data/criteo-uplift-v2.1.parquet')
print(df.shape)  # (13979592, 16)
```

---

## ⚡ 성능 최적화

| 포맷          | 로딩 시간    | 권장도 |
| ----------- | -------- | --- |
| CSV (.gz)   | 10–15초   | ⭐   |
| CSV         | 5–8초     | ⭐⭐  |
| **Parquet** | **1–2초** | ⭐⭐⭐ |

```python
# 필요한 컬럼만 로드
df = pd.read_parquet(
    'Data/criteo-uplift-v2.1.parquet',
    columns=['treatment','conversion','visit','exposure']
)
```

---

## 🧠 분석 접근 방법

### 왜 Uplift Modeling인가?

단순 전환율 예측은 다음을 구분하지 못합니다:

* 광고를 보지 않아도 살 사람
* 광고를 봐야만 살 사람

👉 **Uplift modeling은 이 차이를 직접 추정**합니다.

---

## 📈 모델링 개요

* **Outcome Models**: μ₁(x), μ₀(x) (LightGBM)
* **Learner**: **X-Learner**
* **Uplift Target**: P(Y=1|T=1,X) − P(Y=1|T=0,X)
* **Evaluation**: Qini Curve / Qini AUC

---

## 📊 주요 결과 요약

### Qini Curve

* 무작위 타겟팅 대비 **명확한 incremental lift 관측**
* 상위 고객군에 uplift 효과 집중

### Uplift Feature Importance (GAIN 기준)

가장 중요한 uplift driver:

1. **f4**
2. **f10**
3. **f9**
4. f11, f2, f6

→ **광고 효과의 이질성은 소수 feature에서 발생**

---

## 🔍 SHAP 기반 해석

SHAP 분석을 통해 확인한 사실:

* f4, f9 값이 높은 고객 → 광고 효과 **증가**
* 값이 낮은 고객 → 광고 효과 거의 없음 또는 역효과

> 광고는 “더 많이”가 아니라 **“더 정확하게” 써야 함**

---

## 💰 Budget & ROI Optimization

```python
profit = uplift * value_per_conversion - cost_per_user
```

* uplift > 0 인 고객만 타겟팅
* 동일 예산 대비 **추가 전환 극대화**
* 광고 낭비 구간 명확히 제거 가능

---

## 💡 활용 시나리오

* 광고 타겟팅 자동화
* 캠페인 예산 배분 최적화
* Causal ML 연구 / 실험 설계
* Decision Science / Marketing Science

---

## 🔧 기술 스택

* Python 3.12+
* pandas / numpy / pyarrow
* LightGBM
* scikit-learn
* SHAP
* matplotlib

---

## 📚 참고 자료

* Criteo AI Lab
* Kuusisto et al. (2018), *Uplift Modeling*
* Athey & Imbens (2016), *Causal Trees*

---

## 👤 Author

**Marvin**

---

⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!
