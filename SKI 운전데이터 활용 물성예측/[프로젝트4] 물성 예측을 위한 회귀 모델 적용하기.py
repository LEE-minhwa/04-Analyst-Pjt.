#!/usr/bin/env python
# coding: utf-8

# # [프로젝트4] 물성 예측을 위한 회귀 모델 적용하기

# ---

# ## 프로젝트 목표
# ---
# - 회귀 모델 적용을 위해 데이터 정규화하기
# - 다양한 회귀 모델을 정규화된 데이터에 적용하고 예측하기
# - 5가지 평가지표를 통해 회귀 모델 성능 평가하기

# ## 프로젝트 목차
# ---
# 
# 1. **프로젝트 준비하기:** 프로젝트 수행에 필요한 추가 패키지를 설치하고 기존 패키지를 가져옵니다.
# 
# 2. **데이터 불러오고 전처리하기:** 운전 조건 데이터를 불러오고 Train/Test로 분리합니다. 데이터셋 내에 존재하는 결측치를 KNN 단일 대치법으로 처리합니다.
# 
# 3. **물성 예측을 위한 회귀 모델 적용하기:** 6가지 회귀모델을 적용해보고 5가지 평가지표로 각각의 모델 성능을 확인해봅니다.  
# 
# 
# 

# ## 프로젝트 개요
# ---
# 
# 회귀 모델을 적용하기에 앞서, `sklearn`에 있는 StandardScaler, MinMaxScaler, RobustScaler를 사용하여 데이터를 정규화해봅니다.
# 
# 
# 
# MinMaxScaler를 통해 정규화 된 데이터를 Multiple Linear Regression, Polynomial Regression과 정규화를 적용한 회귀인 Ridge Regression, Lasso Regression, ElasticNet Regression, 그리고 Support Vector Regression 모델에 적용해보고 평가지표를 활용하여 각 모델의 성능을 비교해봅니다.
# 
# 
# 

# ## 1. 프로젝트 준비하기
# ---
# 추가 패키지를 설치하고, 설치되어있는 기존 패키지를 가져옵니다.

# In[1]:


get_ipython().system('pip install missingpy')


# In[2]:


get_ipython().system('pip install scikit-learn==0.22.1')
import sklearn


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

pd.options.display.max_rows = 80
pd.options.display.max_columns = 80


# ## 2. 데이터 불러오고 전처리하기
# ---

# In[4]:


df = pd.read_csv('./Process_data.csv')
df


# In[5]:


df = df.drop('Unnamed: 66', axis = 1)


# In[6]:


df["x62"]
df["x62"] = df['x62'].str.strip("%")
df["x62"]
df["x62"] = df["x62"].astype('float')


# In[7]:


df_date = df['Date']
df = df.set_index("Date")


# In[8]:


train_data = df.iloc[0:691,:] #17년 12월 31일
train_data
test_data = df.iloc[691:,:] #18년 4월 22일
test_data


# In[9]:


tranin_del = train_data.copy()
test_del = test_data.copy()

tranin_del = tranin_del.dropna(axis=1)
test_del = test_del.dropna(axis=1)


# In[10]:


full_columns=['Y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
       'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21',
       'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31',
       'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41',
       'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 'x51',
       'x52', 'x53', 'x54', 'x55', 'x56', 'x57', 'x58', 'x59', 'x60', 'x61',
       'x62', 'x63', 'x64']


# In[11]:


from sklearn.impute import KNNImputer

KNN = KNNImputer(n_neighbors =3)
KNN.fit(train_data)
df_knn_train = KNN.transform(train_data)
df_knn_test = KNN.transform(test_data)
df_knn_train = pd.DataFrame(df_knn_train, columns=full_columns, index = train_data.index)
df_knn_test = pd.DataFrame(df_knn_test, columns=full_columns, index = test_data.index)


# In[12]:


test_data=df_knn_test.copy()
train_data=df_knn_train.copy()


# In[13]:


train_y = train_data['Y']
train_x = train_data.copy()
del train_x['Y']
test_y = test_data['Y']
test_x = test_data.copy()
del test_x['Y']


# ---

# ## 3. 물성 예측을 위한 회귀 모델 적용하기
# ---

# ### 3.1 회귀 모델 적용을 위한 데이터 정규화
# ---

# 회귀 모델에 데이터를 적용하기 위해 먼저, 데이터를 정규화하도록 하겠습니다. 

# ### StandardScaler

# ### [TODO] Standard Scaler를 만들어서 훈련 데이터와 테스트 데이터에 적용하는 코드를 작성하세요.
# - `StandardScaler` 객체를 생성한 후 훈련 데이터로 학습시킵니다.
# - 훈련 데이터와 테스트 데이터에 학습된 `StandardScaler`를 적용합니다.

# In[14]:


import sklearn
from sklearn.preprocessing import *

ss=StandardScaler()
ss.fit(train_x)
ss_train = ss.transform(train_x)
ss_test = ss.transform(test_x)

ss_train = pd.DataFrame(ss_train, columns=train_x.columns, index=train_x.index)
ss_test = pd.DataFrame(ss_test, columns=test_x.columns, index=test_x.index)


# ### MinMaxScaler

# ### [TODO] Min-Max Scaler를 만들어서 훈련 데이터와 테스트 데이터에 적용하는 코드를 작성하세요.
# - `MinMaxScaler` 객체를 생성한 후 훈련 데이터로 학습시킵니다.
# - 훈련 데이터와 테스트 데이터에 학습된 `MinMaxScaler`를 적용합니다.

# In[15]:


ms=MinMaxScaler()
ms.fit(train_x)
ms_train = ms.transform(train_x)
ms_test = ms.transform(test_x)


ms_train = pd.DataFrame(ms_train, columns=train_x.columns, index=train_x.index)
ms_test = pd.DataFrame(ms_test, columns=test_x.columns, index=test_x.index)


# ### RobustScaler

# ### [TODO] Robust Scaler를 만들어서 훈련 데이터와 테스트 데이터에 적용하는 코드를 작성하세요.
# - `RobustScaler` 객체를 생성한 후 훈련 데이터로 학습시킵니다.
# - 훈련 데이터와 테스트 데이터에 학습된 `RobustScaler`를 적용합니다.

# In[16]:


robust=RobustScaler()
robust.fit(train_x)
robust_train = robust.transform(train_x)
robust_test = robust.transform(test_x)

robust_train = pd.DataFrame(robust_train, columns=train_x.columns, index=train_x.index)
robust_test = pd.DataFrame(robust_test, columns=test_x.columns, index=test_x.index)


# ### 3.2 Train/Test Data set 분리
# ---
# 

# 어떤 방법으로 정규화된 데이터를 사용할 것인지 결정합니다. 
# 
# 이번 프로젝트에서는 MinMaxScaler로 정규화된 데이터를 사용하였습니다. 

# In[17]:


# MinMaxScaler 로 정규화된 데이터를 사용합니다.
X_train = ms_train.copy()
X_test = ms_test.copy()
y_train = train_y.copy()
y_test = test_y.copy()


# ### 3.3 평가지표 구현하기
# ---
# 모델의 평가 지표를 직접 함수를 만들어 설정하거나 `sklearn`에서 이미 구현되어있는 평가지표를 가져올 수도 있습니다.
# 

# ### [TODO] 강의에서 본 수식을 참고하여 평가 지표를 직접 구하는 코드를 작성하세요.

# In[18]:


def MAE(y_test, y_pred):
    return np.mean(np.abs(y_test-y_pred))
def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100
def MSE(y_test, y_pred):
    return np.mean(np.square(y_test-y_pred))
def RMSE(y_test, y_pred):
    return np.sqrt(np.mean(np.square(y_test-y_pred))) 
def MPE(y_test, y_pred):
    return np.mean((y_test-y_pred)/y_test)*100


# ### 3.4 피쳐를 제거하지 않은 데이터에 회귀 모델 적용하기
# ---
# 어떤 피쳐도 제거하지 않고 동시에 모델의 하이퍼파라미터를 설정하지 않은 채로 데이터를 모델에 적용해보고, 적용한 결과를 시각화를 통해 확인합니다.

# ### Multiple Linear Regression

# ### [TODO] 선형 회귀를 수행하는 코드를 작성하세요.
# - `LinearRegression` 객체를 만듭니다.
# - 만들어진 `LinearRegression`을 훈련 데이터로 학습시킵니다.
# - 테스트 데이터를 통해 예측값을 얻어냅니다.

# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
linear_r2 = r2_score(y_test,lr_predict)
plt.scatter(y_test, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Linear REGRESSION")
plt.show()
print("linear훈련 세트의 정확도 : {:.2f}".format(lr.score(X_train,y_train)))
print("linear테스트 세트의 정확도 : {:.2f}".format(lr.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))


# ### Polynomial Regression

# ### [TODO] 다항 회귀를 수행하는 코드를 작성하세요.
# - `PolynomialFeatures` 객체를 만들어서 훈련 데이터로 학습시킵니다.
# - 학습된 `PolynomialFeatures`를 통해 훈련 데이터와 테스트 데이터를 변환합니다.
# - `LinearRegression`을 만들어서 변환된 훈련 데이터로 학습시킵니다.
# - 변환된 테스트 데이터를 통해 예측값을 얻어냅니다.

# In[20]:


from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(X_train)
poly_ftr = poly.transform(X_train)
poly_ftr_test = poly.transform(X_test)

plr = LinearRegression()
plr.fit(poly_ftr, y_train)
plr_predict = plr.predict(poly_ftr_test)

plt.scatter(y_test, plr_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(y_test,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))


# ### Ridge Regression 

# ### [TODO] Ridge 회귀를 수행하는 코드를 작성하세요.
# - `Ridge` 객체를 만듭니다.
# - 만들어진 `Ridge`를 훈련 데이터로 학습시킵니다.
# - 테스트 데이터를 통해 예측값을 얻어냅니다.

# In[21]:


from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train,y_train)
ridge_predict = ridge.predict(X_test)
plt.scatter(y_test, ridge_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train,y_train)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))


# ### Lasso Regression

# ### [TODO] Lasso 회귀를 수행하는 코드를 작성하세요.
# - `Lasso` 객체를 만듭니다.
# - 만들어진 `Lasso`를 훈련 데이터로 학습시킵니다.
# - 테스트 데이터를 통해 예측값을 얻어냅니다.

# In[22]:


from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_predict = lasso.predict(X_test)
plt.scatter(y_test, lasso_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_train,y_train)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))


# ### ElasticNet Regression

# ### [TODO] ElasticNet 회귀를 수행하는 코드를 작성하세요.
# - `ElasticNet` 객체를 만듭니다.
# - 만들어진 `ElasticNet`을 훈련 데이터로 학습시킵니다.
# - 테스트 데이터를 통해 예측값을 얻어냅니다.

# In[23]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)
Elastic_pred = elasticnet.predict(X_test)
plt.scatter(y_test, Elastic_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_train,y_train)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))


# ### Support Vector Regression

# ### [TODO] Support Vector Machine으로 회귀를 수행하는 코드를 작성하세요.
# - `SVR` 객체를 만듭니다.
# - 만들어진 `SVR`을 훈련 데이터로 학습시킵니다.
# - 테스트 데이터를 통해 예측값을 얻어냅니다.

# In[24]:


from sklearn.svm import SVR
svm_regressor = SVR()
svm_regressor.fit(X_train, y_train)
svm_pred = svm_regressor.predict(X_test)
plt.scatter(y_test, svm_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("svm훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_train,y_train)))
print("svm테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_test,y_test)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))


# ---
