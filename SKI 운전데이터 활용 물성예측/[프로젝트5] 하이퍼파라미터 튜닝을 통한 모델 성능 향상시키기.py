#!/usr/bin/env python
# coding: utf-8

# # [프로젝트5] 하이퍼파라미터 튜닝을 통한 모델 성능 향상시키기

# ---

# ## 프로젝트 목표
# ---
# - GridSearch를 활용하여 각 모델의 최적이 하이퍼파라미터 튜닝 방법 알아보기
# - GridSearch를 통해 하이퍼파라미터 튜닝 후 모델 성능 향상 여부 확인하기

# ## 프로젝트 목차
# ---
# 
# 1. **프로젝트 준비하기:** 프로젝트 수행에 필요한 추가 패키지를 설치하고 기존 패키지를 가져옵니다.
# 
# 2. **데이터 불러오고 전처리하기:** 운전 조건 데이터를 불러오고 Train/Test로 분리합니다. 데이터셋 내에 존재하는 결측치를 KNN 단일 대치법으로 처리합니다.
# 
# 3. **회귀 모델 적용하기:** 6가지 회귀모델을 적용해보고 5가지 평가지표로 각각의 모델 성능을 확인해봅니다.  
# 
# 3. **하이퍼파라미터 튜닝:** GridSearch를 통해 튜닝한 하이퍼파라미터가 적용된 모델의 성능 향상 여부를 확인해봅니다.   
# 
#  
# 
# 

# ## 프로젝트 개요
# ---
# Polynomial, Ridge, Lasso, ElasticNet Regression, SVR 모델의 최적의 하이퍼파라미터를 GridSearch를 이용해 찾아보고 성능 향상 정도가 유효한지 생각해봅니다.
# 
# 
# 

# ## 1. 프로젝트 준비하기
# ---
# 추가 패키지를 설치하고 기존 설치되어있는 패키지를 가져옵니다.

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


import sklearn


# In[12]:


from sklearn.impute import KNNImputer

KNN = KNNImputer(n_neighbors =3)
KNN.fit(train_data)
df_knn_train = KNN.transform(train_data)
df_knn_test = KNN.transform(test_data)
df_knn_train = pd.DataFrame(df_knn_train, columns=full_columns, index = train_data.index)
df_knn_test = pd.DataFrame(df_knn_test, columns=full_columns, index = test_data.index)


# In[13]:


test_data=df_knn_test.copy()
train_data=df_knn_train.copy()


# In[14]:


train_y = train_data['Y']
train_x = train_data.copy()
del train_x['Y']
test_y = test_data['Y']
test_x = test_data.copy()
del test_x['Y']


# ---

# ## 3. 물성 예측을 위한 회귀 모델 적용하기
# ---

# 데이터 정규화 이후, 기본 하이퍼파라미터를 활용한 모델 적용 및 예측 결과입니다.

# ### StandardScaler

# In[15]:


import sklearn
from sklearn.preprocessing import *

ss=StandardScaler()
ss.fit(train_x)
ss_train = ss.transform(train_x)
ss_test = ss.transform(test_x)

ss_train = pd.DataFrame(ss_train, columns=train_x.columns, index=train_x.index)
ss_test = pd.DataFrame(ss_test, columns=test_x.columns, index=test_x.index)


# ### MinMaxScaler

# In[16]:


ms=MinMaxScaler()
ms.fit(train_x)
ms_train = ms.transform(train_x)
ms_test = ms.transform(test_x)

ms_train = pd.DataFrame(ms_train, columns=train_x.columns, index=train_x.index)
ms_test = pd.DataFrame(ms_test, columns=test_x.columns, index=test_x.index)


# ### RobustScaler

# In[17]:


robust=RobustScaler()
robust.fit(train_x)
robust_train = robust.transform(train_x)
robust_test = robust.transform(test_x)

robust_train = pd.DataFrame(robust_train, columns=train_x.columns, index=train_x.index)
robust_test = pd.DataFrame(robust_test, columns=test_x.columns, index=test_x.index)


# 본 프로젝트에서는 MinMaxScaler로 정규화된 데이터를 활용합니다.

# In[18]:


X_train = ms_train.copy()
X_test = ms_test.copy()
y_train = train_y.copy()
y_test = test_y.copy()


# In[19]:


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


# ### Multiple Linear Regression

# In[20]:


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


# ###  Polynomial Regression

# In[21]:


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

# In[22]:


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

# In[23]:


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

# In[24]:


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

# In[25]:


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

# ## 4. 하이퍼파라미터 튜닝
# ---
# Polynomial, Ridge, Lasso, ElasticNet Regression, SVR은 하이퍼파라미터 튜닝을 통해 성능 향상을 기대할 수 있습니다.
# 
# 직접 하이퍼파라미터를 찾아 설정하는 것은 Manual Search, 
# 
# 하이퍼파라미터들을 여러 개 정하고 그 중에서 가장 좋은 것을 찾는 알고리즘은 GridSearch라고 합니다.

# ### Polynomial regression

# In[26]:


from sklearn.preprocessing import *
poly = PolynomialFeatures(degree=3)
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


# ### GridSearch용 Train/Validation dataset 분리

# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

X_train_val, X_val, y_train_val, y_val = train_test_split(ms_train, train_y, test_size=0.2, shuffle=False) 
##shuffle = True 무작위 추출 False는 순서대로 추출


# ### Ridge GridSearch

# ### [TODO] Ridge 회귀 모델에 grid search를 수행하는 코드를 작성하세요.
# - `GridSearchCV` 객체를 생성합니다. 생성할 때 설정해야 할 파라미터는 아래와 같습니다.
#     - `estimator`
#     - `param_grid`
#     - `cv`
#     - `n_jobs=-1`
#     - `verbose=2`
# - 생성한 `GridSearchCV`를 훈련 데이터로 학습시킵니다.
# - 학습된 `GridSearchCV`에서 best parameter가 무엇인지 확인합니다.

# In[28]:


## Ridge GridSearch
from sklearn.model_selection import GridSearchCV


param_grid = {
    'alpha': [0.1, 0.5, 1.5 , 2, 3, 4]
}

estimator = Ridge()

from sklearn.model_selection import KFold

kf = KFold(
           n_splits=30,
           shuffle=True,
          )

grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )
grid_search.fit(X_train, y_train) # grid_search 학습
grid_search.best_params_ # best parameter 확인


# 각 모델들을 GridSearch로 찾은 하이퍼파라미터를 validation data set으로 확인하여 유효한지 검증합니다.

# In[29]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0.1)
ridge.fit(X_train_val,y_train_val)
ridge_predict = ridge.predict(X_val)
plt.scatter(y_val, ridge_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train_val,y_train_val)))
print("ridge검증 세트의 정확도 : {:.2f}".format(ridge.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(ridge_predict,y_val)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,y_val)))
print("MSE : {:.2f}".format(MSE(ridge_predict,y_val)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,y_val)))
print("MPE : {:.2f}".format(MPE(ridge_predict,y_val)))


# In[30]:


#다른 하이퍼파라미터 값도 넣어 train에서 Overfitting이 생겼는지 확인합니다.
from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 3)
ridge.fit(X_train_val,y_train_val)
ridge_predict = ridge.predict(X_val)
plt.scatter(y_val, ridge_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train_val,y_train_val)))
print("ridge검증 세트의 정확도 : {:.2f}".format(ridge.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(ridge_predict,y_val)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,y_val)))
print("MSE : {:.2f}".format(MSE(ridge_predict,y_val)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,y_val)))
print("MPE : {:.2f}".format(MPE(ridge_predict,y_val)))


# *val 정확도가 -0.61 --> -0.27 GridSearch로 찾은 하이퍼파라미터값이 train data set에서 overfitting이 일어난것을 알 수 있습니다.

# ### Lasso GridSearch

# ### [TODO] Lasso 회귀 모델에 grid search를 수행하는 코드를 작성하세요.
# - `GridSearchCV` 객체를 생성합니다. 생성할 때 설정해야 할 파라미터는 아래와 같습니다.
#     - `estimator`
#     - `param_grid`
#     - `cv`
#     - `n_jobs=-1`
#     - `verbose=2`
# - 생성한 `GridSearchCV`를 훈련 데이터로 학습시킵니다.
# - 학습된 `GridSearchCV`에서 best parameter가 무엇인지 확인합니다.

# In[31]:


## Lasso GridSearch
from sklearn.model_selection import GridSearchCV


param_grid = {
    'alpha': [0.0001, 0.001,0.005, 0.02, 0.1, 0.5, 1, 2]
}

estimator = Lasso()

from sklearn.model_selection import KFold

kf = KFold(
           n_splits=30,
           shuffle=True,
          )

grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )
grid_search.fit(X_train, y_train) # grid_search 학습
grid_search.best_params_ # best parameter 확인


# In[32]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.001)
lasso.fit(X_train_val,y_train_val)
lasso_predict = lasso.predict(X_val)
plt.scatter(y_val, lasso_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_train_val,y_train_val)))
print("lasso검증 세트의 정확도 : {:.2f}".format(lasso.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(lasso_predict,y_val)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,y_val)))
print("MSE : {:.2f}".format(MSE(lasso_predict,y_val)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,y_val)))
print("MPE : {:.2f}".format(MPE(lasso_predict,y_val)))


# In[33]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(X_train_val,y_train_val)
lasso_predict = lasso.predict(X_val)
plt.scatter(y_val, lasso_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_train_val,y_train_val)))
print("lasso검증 세트의 정확도 : {:.2f}".format(lasso.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(lasso_predict,y_val)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,y_val)))
print("MSE : {:.2f}".format(MSE(lasso_predict,y_val)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,y_val)))
print("MPE : {:.2f}".format(MPE(lasso_predict,y_val)))


# ### ElasticNet GridSearch

# ### [TODO] ElasticNet 회귀 모델에 grid search를 수행하는 코드를 작성하세요.
# - `GridSearchCV` 객체를 생성합니다. 생성할 때 설정해야 할 파라미터는 아래와 같습니다.
#     - `estimator`
#     - `param_grid`
#     - `cv`
#     - `n_jobs=-1`
#     - `verbose=2`
# - 생성한 `GridSearchCV`를 훈련 데이터로 학습시킵니다.
# - 학습된 `GridSearchCV`에서 best parameter가 무엇인지 확인합니다.

# In[34]:


## ElasticNet GridSearch
from sklearn.model_selection import GridSearchCV


param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1, 2],
    'l1_ratio': [0.01, 0.1, 0.5, 0.7],
}

estimator = ElasticNet()

from sklearn.model_selection import KFold

kf = KFold(
           n_splits=30,
           shuffle=True,
          )

grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )

grid_search.fit(X_train, y_train) # grid_search 학습
grid_search.best_params_ # best parameter 확인


# In[35]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001, l1_ratio = 0.01)
elasticnet.fit(X_train_val, y_train_val)
Elastic_pred = elasticnet.predict(X_val)
plt.scatter(y_val, Elastic_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_train_val,y_train_val)))
print("Elastic검증 세트의 정확도 : {:.2f}".format(elasticnet.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,y_val)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,y_val)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,y_val)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,y_val)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,y_val)))


# In[36]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(X_train_val, y_train_val)
Elastic_pred = elasticnet.predict(X_val)
plt.scatter(y_val, Elastic_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_train_val,y_train_val)))
print("Elastic검증 세트의 정확도 : {:.2f}".format(elasticnet.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,y_val)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,y_val)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,y_val)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,y_val)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,y_val)))


# ### Support Vector Machine GridSearch

# ### [TODO] Support Vector Machine 회귀 모델에 grid search를 수행하는 코드를 작성하세요.
# - `GridSearchCV` 객체를 생성합니다. 생성할 때 설정해야 할 파라미터는 아래와 같습니다.
#     - `estimator`
#     - `param_grid`
#     - `cv`
#     - `n_jobs=-1`
#     - `verbose=2`
# - 생성한 `GridSearchCV`를 훈련 데이터로 학습시킵니다.
# - 학습된 `GridSearchCV`에서 best parameter가 무엇인지 확인합니다.

# In[37]:


param_grid = {
    'kernel': ['linear','poly','rbf','sigmoid']
}
estimator = SVR()

kf = KFold(
           n_splits=30,
           shuffle=True,
          )

grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )

grid_search.fit(X_train, y_train) # grid_search 학습
grid_search.best_params_ # best parameter 확인


# In[38]:


from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'poly')
svm_regressor.fit(X_train_val, y_train_val)
svm_pred = svm_regressor.predict(X_val)
plt.scatter(y_val, svm_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("SVM REGRESSION")
plt.show()
print("svm훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_train_val,y_train_val)))
print("svm검증 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(svm_pred,y_val)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,y_val)))
print("MSE : {:.2f}".format(MSE(svm_pred,y_val)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,y_val)))
print("MPE : {:.2f}".format(MPE(svm_pred,y_val)))


# In[39]:


from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(X_train_val, y_train_val)
svm_pred = svm_regressor.predict(X_val)
plt.scatter(y_val, svm_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("SVM REGRESSION")
plt.show()
print("svm훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_train_val,y_train_val)))
print("svm검증 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_val,y_val)))
print("MAE : {:.2f}".format(MAE(svm_pred,y_val)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,y_val)))
print("MSE : {:.2f}".format(MSE(svm_pred,y_val)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,y_val)))
print("MPE : {:.2f}".format(MPE(svm_pred,y_val)))


# ---
