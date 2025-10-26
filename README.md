# 매출수량 예측 프로젝트 (Sales Volume Prediction Project)
https://dacon.io/competitions/official/236559/overview/description
---

## 1. 데이터 로드 및 결콩치 처리

```python
import pandas as pd
import numpy as np

# 데이터 로드하기
df = pd.read_csv("train.csv")

# 기본 정보 확인
print(df.info())
print(df.describe())

# 결콩치 확인
print("\n[결측치 확인]")
print(df.isnull().sum())

# 결측치 처리 (수치형: 평균, 범주형: 최빈값)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

print("\n결측치 처리 완료")
print(df.isnull().sum())
```

**해석:**  
데이터셋은 결측치가 일부 존재하였으나, 수치형은 평균값으로, 범주형은 최빈값으로 보완하였다.  
이 과정을 통해 전체 데이터가 안정적으로 학습에 활용될 수 있는 형태로 전처리되었다.

---

## 2. 탐색적 데이터 분석 (EDA)

### (1) 요일별 평균 매출수량

```python
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='요일', y='매출수량', estimator='mean', errorbar=None,
            order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title("요일별 평균 매출수량")
plt.xlabel("요일")
plt.ylabel("평균 매출수량")
plt.show()
```
<img width="840" height="544" alt="image" src="https://github.com/user-attachments/assets/e73b560a-ed32-40af-becb-a4ed5b068b4f" />


**해석:**  
주중보다 주말(토·일)에 매출이 뚜렷하게 증가하는 패턴을 보인다.  
특히 토요일의 평균 매출수량이 가장 높으며, 이는 주말 소비 활동 및 외식·배달 수요 집중 현상을 반영한다.

---

### (2) 월별 총 매출수량 추이

```python
plt.figure(figsize=(10,6))
sns.lineplot(data=df, x='월', y='매출수량', estimator='sum', errorbar=None, marker='o')
plt.title("월별 매출수량 추이")
plt.xlabel("월")
plt.ylabel("매출수량 합계")
plt.show()
```
<img width="873" height="544" alt="image" src="https://github.com/user-attachments/assets/54fdcb4f-6b34-4c29-ba2a-528d8744f294" />


**해석:**  
1~2월 초반에 매출이 높고, 3월 이후 하락하는 형태를 보인다.  
이는 신년·명절 시즌의 프로모션 효과 및 계절적 요인을 반영한 결과로 해석된다.

---

### (3) 메뉴별 평균 매출수량 TOP10

```python
top_items = df.groupby('영업장명_메뉴명')['매출수량'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_items.values, y=top_items.index)
plt.title("메뉴별 평균 매출수량 TOP10")
plt.xlabel("평균 매출수량")
plt.ylabel("영업장명_메뉴명")
plt.show()
```

<img width="1159" height="544" alt="image" src="https://github.com/user-attachments/assets/4a870233-2455-4464-a78f-bd09a6bcc2f3" />


**해석:**  
‘포레스트릿_고치어묵’, ‘화담숲주막_해물파전’, ‘포레스트릿_떡볶이’가 상위 3위에 해당하며  
이는 특정 인기 메뉴군이 전체 매출을 견인하고 있음을 의미한다.  
향후 예측 모델에서는 해당 메뉴군의 계절성·요일 패턴을 주요 변수로 반영할 필요가 있다.

---

### (4) 업체별 총 매출수량 분포

```python
df[['영업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
sales_by_store = (
    df.groupby('영업장명', as_index=False)['매출수량']
    .sum()
    .sort_values(by='매출수량', ascending=True)
)

plt.figure(figsize=(10,6))
sns.barplot(data=sales_by_store, x='매출수량', y='영업장명', palette='Blues_r')
plt.title("업체별 총 매출수량 분포")
plt.xlabel("총 매출수량")
plt.ylabel("업체명")
plt.tight_layout()
plt.show()
```
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/e7792070-2fd5-40fd-8871-3d99fb535324" />


**해석:**  
‘포레스트릿’, ‘카페테리아’, ‘화담숲주막’이 전체 매출의 중심을 차지하며  
특히 ‘포레스트릿’의 매출이 가장 두드러져 주력 매장임을 확인할 수 있다.

---

## 3. 군집화 (KMeans 파생변수 생성)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

num_cols = df.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
scaled = scaler.fit_transform(df[num_cols])

kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled)
print(df[['영업일자', '매출수량', 'cluster']].head())
```

**해석:**  
3개의 군집(cluster)을 통해 매장·메뉴의 유사 매출 패턴을 분류하였다.  
이 군집 변수는 이후 회귀분석 및 예측 모델의 주요 설명 변수로 활용된다.

---

## 4. 일반화 선형모형(GLM) 분석

**결과 요약:**  

| 변수 | 계수 | 영향 방향 | 유의성 |
|------|------|------------|--------|
| 요일(Saturday) | +1.60 | 매출 증가 | p<0.001 |
| 요일(Monday~Thursday) | 음(-) | 매출 감소 | p<0.001 |
| cluster 1 | +332.75 | 매우 강한 매출 영향 | p<0.001 |
| 월 | -0.33 | 하락 추세 | p<0.001 |

**해석:**  
토요일은 유의한 양(+)의 영향을 보여 주말 매출이 강하게 상승하며,  
월~목은 음(-)의 영향을 보여 평일 매출이 낮은 구조를 보인다.  
또한 특정 군집(cluster 1)은 매출을 강하게 견인하며,  
Pseudo R²=0.7435로 높은 설명력을 확보하였다.

---

## 5. 회귀모델 성능 비교 (Ridge vs RandomForest vs GradientBoosting)

| 모델 | MAE | RMSE | R² |
|------|------|------|------|
| Ridge | 14.38 | 35.72 | 0.21 |
| RandomForest | **5.97** | **19.19** | **0.77** |
| GradientBoosting | 7.90 | 20.21 | 0.75 |

**해석:**  
비선형 트리 기반 모델(RandomForest, GradientBoosting)이 Ridge보다 월등히 높은 예측 성능을 보였다.  
특히 RandomForest가 가장 안정적인 예측력을 보여, 비선형 상호작용이 중요한 데이터 구조임을 시사한다.

---

## 6. LightGBM 기반 Stacking 모델

```python
stack_model = StackingRegressor(
    estimators=[('rf', RandomForestRegressor()), ('gb', GradientBoostingRegressor())],
    final_estimator=LGBMRegressor(),
    passthrough=False
)
stack_model.fit(X_train_reduced, y_train)
```

**결과:**  
- R² = 0.7836  
- MAE = 6.30  
- RMSE = 18.74  
- 피처 56개 제거 후 최종 132개 사용  

**해석:**  
불필요한 변수를 제거하고 LightGBM을 메타모델로 적용함으로써  
모델의 일반화 성능이 향상되었다.  
이는 변수 선택(Feature Selection)이 효과적으로 작동한 결과로 해석된다.

---

## 7. Feature Importance (LightGBM)

<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/5270e498-41f6-43d6-9df6-a2ef7b12aa33" />


**해석:**  
- 상위 중요 변수: **월, cluster, 요일(토·일·월), 인기메뉴(고치어묵·떡볶이)**  
- ‘월’ 변수의 중요도는 계절적 매출 변동성을 의미하며,  
  cluster 변수는 매장·메뉴군의 구조적 차이를 반영한다.  
- 주말 요인과 특정 인기메뉴의 높은 중요도는  
  실제 매출 집중 패턴과 일치한다.

---

## 8. 종합 결론

본 프로젝트는 **시간(월·요일)**, **군집(cluster)**, **개별 매장·메뉴 특성**이  
매출수량에 미치는 영향을 다각도로 분석하였다.  

GLM과 머신러닝 기반 모델 모두  
주말 중심 매출 집중 및 특정 매장군(cluster 1)의 매출 주도 현상을 확인하였으며,  
최종 LightGBM 스태킹 모델은 **R²=0.78** 수준의 우수한 예측 성능을 보였다.  

향후 매출 예측 및 재고관리 시스템 구축 시,  
**계절성 + 요일 + 클러스터 + 주요 메뉴 변수**를 중심으로 모델을 최적화하는 것이 효과적이다.

---

```
Sales_Volume_Prediction
 ┣ train.csv
 ┣ sales_analysis.ipynb
 ┣ README.md
 ┗ requirements.txt
```

---

## Author

**윤해정 (Yoon Haejeong)**  
heajeongy@naver.com  
Data Analytics & Visualization / Python, SQL, Power BI  
[GitHub](https://github.com/heajeongy-design)
