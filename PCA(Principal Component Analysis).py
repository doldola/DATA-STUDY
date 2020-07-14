#!/usr/bin/env python
# coding: utf-8

# # PCA(Principal Component Analysis)

# ### PCA의 개요

# - PCA(Principal Component Analysis)는 가장 대표적인 차원축소 기법이다.
# 
# 
# - PCA는 여러 변수 간에 존재하는 상관관계를 이용해 이를 대표하는 주성분(Principal Component)을 추출해 차원을 축소하는 기법이다.
# 
# 
# - PCA로 차원을 축소할때 기존 데이터의 정보 유실이 최소화된다.
#     - 이를 위해 PCA는 가장 높은 분산을 가지는 데이터의 축을 찾아 이 축으로 차원축소 -> PCA의 주성분

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# - PCA는 제일 먼저 가장 큰 데이터 변동성(Variance)를 기반으로 첫 번째 벡터 축을 생성
# 
# 
# - 두 번째 축은 이 벡터 축에 직각이 되는 벡터(직교 벡터)를 축으로 함
# 
# 
# - 세 번째 축은 다시 두 번째 축과 직각이 되는 벡터를 설정하는 방식으로 축을 생성함
# 
# 
# - 이렇게 생성된 벡터 축에 원본 데이터를 투영
#     - 벡터 축의 개수만큼의 차원으로 원본 데이터가 차원 축소된다.
# 

# 그림삽입

# - PCA, 즉 주성분 분석은 원본 데이터의 피처 개수에 비해 매우 작은 주성분으로 원본 데이터의 총 변동성을 대부분 설명할 수 있다.

# ### PCA의 선형대수 관점에서의 해석

# - 입력 데이터의 공분산 행렬(Covariance Matrix)을 고유값 분해하고, 이렇게 구한 고유벡터에 입력 데이터를 선형 변환.
#     - 공분산 행렬 : 2 이상의 변량들에서, 다수의 두 변량 값들 간의 공분산 또는 상관계수들을 행렬로 표현한 것
#     
#     
# - 고유벡터가 PCA의 주성분 벡터로서 입력 데이터의 분산이 큰 방향을 나타냄
# 
# 
# - 고윳값(eigenvalue)은 바로 이 고유벡터의 크기이며, 동시에 입력 데이터의 분산을 나타냄

# ##### 선형변환
# - 선형 변환은 특정 벡터에 행렬 A를 곱해 새로운 벡터로 변환하는 것을 의미한다.
# 
# 
# - 이를 특정 벡터를 하나의 공간에서 다른 공간으로 투영하는 개념으로도 볼 수 있으며, 이 경우 이 행렬을 바로 공간으로 가정하는 것이다.

# - 보통 분산은 한 개의 특정한 변수의 데이터 변동을 의미하나, 공분산은 두 변수 간의 변동을 의미한다.
# 
# 
#     - 공분산 행렬은 여러 변수와 관련된 공분산을 포함하는 정방형 행렬이다.
# 
# ##### 고유벡터
# - 고유벡터는 행렬 A를 곱하더라도 방향이 변하지 않고 그 크기만 변하는 벡터를 지칭한다.
# 
# 
#     - 즉, Ax = ax(A는 행렬, x는 고유벡터, a는 스칼라값)이다.
#     
#     
#     - 이 고유벡터는 여러 개가 존재하며, 정방 행렬은 최대 그 차원 수만큼의 고유벡터를 가질 수 있다.
#     
#     
#     - 2x2 행렬은 2개, 3x3 행렬은 3개
#     
#     
#     - 고유벡터는 행렬이 작용하는 힘의 방향과 관계가 있어서 행렬을 분해하는 데 사용됨.
# 
# ##### 공분산 행렬, 정방행렬, 대칭행렬
# ![image.png](attachment:image.png)
# - 공분산 행렬은 정방행렬(Diagonal Matrix)이며 대칭행렬(Symmetric Matrix)이다.
# 
# 
#     - 정방행렬은 열과 행이 같은 행렬을 지칭하는데, 
#     
#     
#     - 정방행렬 중에서 대각 원소를 중심으로 원소 값이 대칭되는 행렬, 즉 A^T = A 인 행렬을 대칭행렬이라고 부른다.
#     
#     
#     - 공분산 행렬은 개별 분산값을 대각 원소로 하는 대칭행렬이다.
#     
#     
#         - 이 대칭행렬은 항상 교유벡터를 직교행렬(orthogonal matrix)로, 고유값을 정방 행렬로 대각화할 수 있다.

# ## 즉, 입력 데이터의 공분산 행렬이 고유벡터와 고유값으로 분해될 수 있으며,
# ## 이렇게 분해된 고유벡터를 이용해 입력 데이터를 선형 변환하는 방식이 PCA이다.

# ### PCA 수행과정
# - 1. 입력 데이터 세트의 공분산 행렬을 생성합니다.
# 
# 
# - 2. 공분산 행렬의 고유벡터와 고유값을 계산합니다.
# 
# 
# - 3. 고유값이 가장 큰 순으로 K개(PCA 변환 차수만큼)만큼 고유벡터를 추출합니다.
# 
# 
# - 4. 고유값이 가장 큰 순으로 추출된 고유벡터를 이용해 새롭게 입력 데이터를 변환합니다.

# In[12]:


from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

iris = load_iris()
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
irisDF = pd.DataFrame(iris.data, columns = columns)
irisDF['target']=iris.target
irisDF.head(3)


# In[8]:


#원본 붓꽃 데이터세트 분포를 보기위해 2차원으로 시각화
#sepal length와 sepal width를 x,y축으로 해 품종 데이터 분포를 나타냄

#setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=['^', 's', 'o']

#setosa의 target 값은 0, versicolor는 1, virginica는 2. 
#각 타겟별로 다른 모양으로 산점도 표시
for i, marker in enumerate(markers):
    x_axis_data = irisDF[irisDF['target']==i]['sepal_length']
    y_axis_data = irisDF[irisDF['target']==i]['sepal_width']
    plt.scatter(x_axis_data, y_axis_data, marker = marker, label = iris.target_names[i])
    
plt.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()


# In[9]:


# 세토사는 분포가 일정하나 versicolor 와 virginica 의 경우 sepal width 와  sepal length 로는 분류하기 어렵다.
# PCA로 4개의 속성을 2개로 압축한 뒤 2개의 PCA 속성으로 붓꽃 품종 데이터 2차원으로 시각화


# In[10]:


from sklearn.preprocessing import StandardScaler

#Target 값을 제외한 모든 속성 값을 StnadardScaler를 이용해 표준 정규 분포를 가지는 값들로 변환
iris_scaled = StandardScaler().fit_transform(irisDF.iloc[:,:-1])


# In[11]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

#fit()과 transform()을 호출해 PCA 변환 데이터 반환
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)
print(iris_pca.shape)


# In[13]:


irisDF.shape


# In[17]:


#PCA 변환된 데이터의 칼럼 명을 가각 pca_component_1, pca_component_2로 명명
pca_columns = ['pca_component_1', 'pca_component_2']
irisDF_pca = pd.DataFrame(iris_pca, columns = pca_columns)
irisDF_pca['target'] = iris.target
irisDF_pca.head(3)


# In[18]:


#setosa는 세모, versicolor는 네모, virginica는 동그라미로 표현
markers=['^', 's', 'o']

# pca_component_1을 x축, pca_component_2를 y축으로 scatter plot 수행
#각 타겟별로 다른 모양으로 산점도 표시
for i, marker in enumerate(markers):
    x_axis_data = irisDF_pca[irisDF['target']==i]['pca_component_1']
    y_axis_data = irisDF_pca[irisDF['target']==i]['pca_component_2']
    plt.scatter(x_axis_data, y_axis_data, marker = marker, label = iris.target_names[i])
    
plt.legend()
plt.xlabel('pca_component_1')
plt.ylabel('pca_component_2')
plt.show()


# In[19]:


print(pca.explained_variance_ratio_)


# - 첫 번째 PCA 변환요소인 pca_component_1이 전체 변동성의 약 72.9% 차지
# - 두 번째 PCA 변환요소인 pca_component_2가 전체 변동성의 약 22.8% 차지
# 

# In[21]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rcf = RandomForestClassifier(random_state =156)
scores = cross_val_score(rcf, iris.data, iris.target, scoring = 'accuracy', cv =3)
print('원본 데이터 교차 검증 개별 정확도:', scores)
print('원본 데이터 평균 정확도:', np.mean(scores))


# In[22]:


pca_X = irisDF_pca[['pca_component_1','pca_component_2']]
scores_pca = cross_val_score(rcf, pca_X, iris.target, scoring='accuracy', cv=3)
print('PCA 변환 데이터 교차 검증 개별 정확도:', scores_pca)
print('PCA 벼노한 데이터 평균 정확도:', np.mean(scores_pca))


# - 원본 데이터 세트 대비 예측 정확도는 PCA 변환 차원 개수에 따라 예측 성능이 떨어질 수 밖에 없다.
# - 위의 경우 10% 하락, 비교적 큰 수치지만 속성개수가 50% 감소한 것을 고려한다면 PCA 변한 후에도 원본 데이터의 특성을 상당 부분 유지하고 있음을 알 수 있다.

# In[ ]:





# # kernel PCA

# - 회귀분석과 대부분의 분류 알고리즘에서는 데이터를 선형분리 가능하다는 가정이 필요했다.
# 
# 
# - 심지어 인공신경망의 기초격인 퍼셉트론에서조차 선형분리 가능을 전제로 알고리즘이 동작했다.
# 
# 
# - 이러한 문제점들을 PCA, LDA같은 차원 축소 기법으로 해결했으나
# 
# 
# - 데이터의 모양이 정말로, 완전히, 앱솔루틀리 하게 비선형인 경우는 문제가 달라진다.
# 
# 
# - PCA, LDA 같은 알고리즘은 차원축소를 위한 선형 변환 기법을 이용하기 때문에,
# 
# 
# - 선형으로 분리 불가능한 데이터에 대해서는 적당하지 않다. 

# ### 이런 문제를 극복하기 위해 커널 PCA를 사용할 수 있다

# ![image.png](attachment:image.png)

# In[ ]:


Non-linear problem을 가진 data를 extraction방식으로 Dimensionality Reduction을 하기 위한 Kernel PCA는

original feature의 2D space를 higher dimension을 추가하는 방식인 kernel trick을 써서 3D space상에 dataset을 mapping하고 

dimension을 추가하기 때문에 Dimensionality Reduction기법인 PCA를 통해 새로운 변수를 추출하고 

이를 통해서 Non-linear dataset을  classification으로 분석할 수 있게 합니다.


# In[29]:


from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)


# In[30]:


from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)


# In[31]:


X_reduced


# In[32]:


from sklearn.decomposition import KernelPCA

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

y = t > 6.9

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), 
                            (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), 
                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    if subplot == 132:
        X_reduced_rbf = X_reduced
    
    plt.subplot(subplot)
    #plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
    #plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

plt.show()


# In[34]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.figure(figsize=(6, 5))

X_inverse = pca.inverse_transform(X_reduced_rbf)

ax = plt.subplot(111, projection='3d')
ax.view_init(10, -70)
ax.scatter(X_inverse[:, 0], X_inverse[:, 1], X_inverse[:, 2], c=t, cmap=plt.cm.hot, marker="x")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

plt.show()


# In[35]:


X_reduced = rbf_pca.fit_transform(X)

plt.figure(figsize=(11, 4))
plt.subplot(132)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot, marker="x")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)


# #### Selecting a Kernel and Tuning Hyperparameters
# - 그리드 서치를 사용하여 커널과 하이퍼파라미터를 고를수 있음
# - 두단계의 파이프라인이 필요 1) 2차원으로축소, 2) 로지스틱회귀로 분류를 적용
# - 그러면 GridSearchCV를 활용해 최적의 커널과 감마 값을 찾아줌

# In[36]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[37]:


clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
])

param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)


# In[38]:


print(grid_search.best_params_)


# In[ ]:





# 참조 : http://textmining.kr/?p=362
# 참조 : 파이썬 머신러닝 완벽가이드
# 참조 : https://yamalab.tistory.com/42
# 참조 : 
