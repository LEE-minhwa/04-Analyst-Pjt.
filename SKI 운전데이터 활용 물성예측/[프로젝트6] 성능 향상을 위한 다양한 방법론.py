#!/usr/bin/env python
# coding: utf-8

# # [프로젝트6] 성능 향상을 위한 다양한 방법론

# ---

# ## 프로젝트 목표
# ---
# - 피쳐 셀렉션(Correlation, Univariate, Feature Importance, RFE)을 통해 모델의 성능 높이기
# - 모델의 성능을 향상시키기 위해 추가로 어떤 방법을 적용할 수 있을지 생각해보기

# ## 프로젝트 목차
# ---
# 
# 1. **프로젝트 준비하기:** 프로젝트 수행을 위해 추가 패키지를 설치하고 기존 패키지를 가져옵니다.
# 
# 2. **데이터 불러오고 전처리하기:** 운전 조건 데이터를 불러오고 Train/Test로 분리합니다. KNN 단일 대치법으로 결측치를 대치합니다.
# 
# 3. **높은 상관계수 피쳐 제거:** 변수들의 상관관계를 확인하고 상관성이 높은 피쳐를 선정하여 제거하고 모델에 적용합니다.
# 
# 4. **Univariate를 통한 피쳐 셀렉:** Univariate를 통한 피쳐 셀렉 후 모델에 적용합니다. 
# 
# 5. **Select From Model 피쳐 제거 방법:**  SelectFromModel을 통해 모델이 학습하면서 중요하다고 판단되는 피쳐를 셀렉합니다.
# 
# 6. **MODEL SELECT + RFE:**  Model Select를 통해 피쳐를 선정하고 선정된 피쳐들 중 학습에 중요한 피쳐를 RFE를 사용하여 선정합니다.
# 
# 7. **Select K + Outlier Detection:**  Univariate 을 통해 셀렉션한 최종 피쳐의 데이터로 이상치를 제거합니다. 
# 
# 8. **Model Select + RFE + 이상치 제거(Isolation Forest):** Model Select + RFE를 통해 선택한 피쳐 데이터 셋에서 이상치를 탐지하고 제거합니다.

# ## 프로젝트 개요
# ---
# 다양한 방법으로 피쳐 셀렉션을 진행하여 피쳐를 최종적으로 선택하고 더 좋은 결과를 도출하기 위해 이상치를 제거하여 모델에 적용합니다.
# 
# 이 밖에 추가적으로 모델의 성능을 향상시키기 위해서 어떤 방법들이 있을지 생각해봅니다.
# 
# 

# ## 1. 프로젝트 준비하기
# ---
# 추가 패키지를 설치하고 설치되어있는 기존 패키지를 가져옵니다.

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


# In[2]:


df_date = df['Date']
df = df.set_index("Date")


# In[3]:


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


# ## 3. 높은 상관계수 피쳐 제거
# ---

# 피처 셀렉션(Correlation, Univariate, Feature Importance, RFE)을 통해 모델의 성능을 높이도록 합니다.
# 

# ### 3.1 회귀 모델 적용을 위한 데이터 정규화
# ---

# ### StandardScaler

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

# In[15]:


ms=MinMaxScaler()
ms.fit(train_x)
ms_train = ms.transform(train_x)
ms_test = ms.transform(test_x)

ms_train = pd.DataFrame(ms_train, columns=train_x.columns, index=train_x.index)
ms_test = pd.DataFrame(ms_test, columns=test_x.columns, index=test_x.index)


# ###  RobustScaler

# In[16]:


robust=RobustScaler()
robust.fit(train_x)
robust_train = robust.transform(train_x)
robust_test = robust.transform(test_x)

robust_train = pd.DataFrame(robust_train, columns=train_x.columns, index=train_x.index)
robust_test = pd.DataFrame(robust_test, columns=test_x.columns, index=test_x.index)


# In[17]:


X_train = ms_train.copy()
X_test = ms_test.copy()
y_train = train_y.copy()
y_test = test_y.copy()


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


# ### 3.2 상관계수가 높은 피쳐들을 제거한 후 모델 적용하기
# ---
# 상관분석을 통해 상관관계가 높은 피쳐를 선정하여 제거하고 모델에 적용합니다.

# In[19]:


X_train_corr = X_train.copy()
X_test_corr = X_test.copy()


# In[20]:


# 상관계수가 0.9 이상인 피처(Train Data)
del X_train_corr['x54']
del X_train_corr['x55']
del X_train_corr['x60']
del X_train_corr['x40'] 
del X_train_corr['x41'] 
del X_train_corr['x42']
del X_train_corr['x43']
del X_train_corr['x45']
del X_train_corr['x4']
del X_train_corr['x20']
del X_train_corr['x18']
del X_train_corr['x14']
del X_train_corr['x16']
del X_train_corr['x25']
del X_train_corr['x62']


# In[21]:


# 상관계수가 0.9 이상인 피처(Test Data)

del X_test_corr['x54']
del X_test_corr['x55']
del X_test_corr['x60']
del X_test_corr['x40'] 
del X_test_corr['x41'] 
del X_test_corr['x42']
del X_test_corr['x43']
del X_test_corr['x45']
del X_test_corr['x4']
del X_test_corr['x20']
del X_test_corr['x18']
del X_test_corr['x14']
del X_test_corr['x16']
del X_test_corr['x25']
del X_test_corr['x62']


# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
lr = LinearRegression(n_jobs=-1)
lr.fit(X_train_corr, y_train)
lr_predict = lr.predict(X_test_corr)
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
print("linear훈련 세트의 정확도 : {:.2f}".format(lr.score(X_train_corr,y_train)))
print("linear테스트 세트의 정확도 : {:.2f}".format(lr.score(X_test_corr,y_test)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))


# In[23]:


from sklearn.preprocessing import *
poly = PolynomialFeatures(degree=3)
poly.fit(X_train_corr)
poly_ftr = poly.transform(X_train_corr)

poly_ftr_test = poly.transform(X_test_corr)
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


# In[24]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 3)
ridge.fit(X_train_corr,y_train)
ridge_predict = ridge.predict(X_test_corr)
plt.scatter(y_test, ridge_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train_corr,y_train)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_test_corr,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))


# In[25]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(X_train_corr,y_train)
lasso_predict = lasso.predict(X_test_corr)
plt.scatter(y_test, lasso_predict, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_train_corr,y_train)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_test_corr,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))


# In[26]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(X_train_corr, y_train)
Elastic_pred = elasticnet.predict(X_test_corr)
plt.scatter(y_test, Elastic_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_train_corr,y_train)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_test_corr,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))


# In[27]:


from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(X_train_corr, y_train)
svm_pred = svm_regressor.predict(X_test_corr)
plt.scatter(y_test, svm_pred, alpha=0.4)
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.xlim([7, 8])      # X축의 범위: [xmin, xmax]
plt.ylim([7, 8])     # Y축의 범위: [ymin, ymax]
plt.title("SVM REGRESSION")
plt.show()
print("svm훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_train_corr,y_train)))
print("svm테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_test_corr,y_test)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))


# ## 4. Univariate를 통한 피쳐 셀렉
# ---
# 

# ### 4.1 Train/Validation Data 분리
# ---
# Univariate 피쳐 셀렉션 방법을 사용하기 위해 Train과 Validation을 분리하고 검증하도록 합니다.
# 

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(ms_train, train_y, test_size=0.2, shuffle=False) 
##shuffle = True 무작위 추출 False는 순서대로 추출


# ### 4.2 K선택을 통한 피쳐 셀렉
# ---

# ### [TODO] K선택을 통해 피쳐 셀렉을 수행하는 코드를 작성하세요.
# - `SelectKBest` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#     - `score_func=f_classif`
#     - `k=i+1`
# - Min-Max Scaler를 통해 변환된 훈련 데이터로 `SelectKBest`를 학습시킵니다.
# - 훈련 데이터와 검증 데이터에 학습된 `SelectKBest`를 적용합니다.

# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, f_regression, mutual_info_regression,SelectPercentile, SelectFwe


r2=[]
for i in range(0,64):
    selectK = SelectKBest(score_func=f_classif, k=i+1)
    selectK.fit(ms_train, train_y) # selectK 학습
    X = selectK.transform(X_train) # 학습 데이터 변환
    X_v = selectK.transform(X_val) # 검증 데이터 변환
    lr.fit(X, y_train)
    lr.score(X_v,y_val)
    r2.append(lr.score(X_v,y_val))
plt.figure()
plt.plot(range(0, 64), r2, label='R2' )
plt.xlabel("number of features")
plt.ylabel("R2")
plt.legend()
plt.show()


# In[30]:


#Validation 정확도 최대값 확인
np.max(r2)


# In[31]:


#해당 피처 개수 확인
r2.index(np.max(r2))


# In[32]:


#피처 개수별 val 정확도 확인
r2


# ### [TODO] K선택을 통해 피쳐 셀렉을 수행하는 코드를 작성하세요.
# - `SelectKBest` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#     - `score_func=f_classif`
#     - `k=9`
# - Min-Max Scaler를 통해 변환된 훈련 데이터로 `SelectKBest`를 학습시킵니다.
# - Min-Max Scaler를 통해 변환된 훈련 데이터에 학습된 `SelectKBest`를 적용합니다. (`X`)
# - 훈련, 검증, 테스트 데이터에 학습된 `SelectKBest`를 적용합니다. (`X_tr`, `X_v`, `X_t`)

# In[33]:


##val 정확도를 확인하고 셀렉할 피처 개수를 정해줍니다. 
selectK = SelectKBest(score_func=f_classif, k=9)
selectK.fit(ms_train, train_y) # selectK 학습
X_tr = selectK.transform(X_train) # 일반 훈련 데이터 변환
X = selectK.transform(ms_train) # Min-Max Scaler를 통해 변환된 훈련 데이터를 변환
X_v = selectK.transform(X_val) # 검증 데이터 변환
X_t = selectK.transform(ms_test) # 테스트 데이터 변환
TF = selectK.get_support() # get_support통해 어떤 피처가 제거 되었는지 확인할 수 있습니다.
TF = pd.DataFrame(TF, index=ms_train.columns)


# ### 4.3 모델 적용 후 성능 확인하기
# ---

# ### Linear Regression

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(X, train_y)
lr_predict = lr.predict(X_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(X,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_t,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))


# ### Polynomial Regression

# In[35]:


from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(X)
poly_ftr = poly.transform(X)

poly_ftr_test = poly.transform(X_t)
plr = LinearRegression()
plr.fit(poly_ftr, train_y)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))


# ### Ridge Regression

# In[36]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 3)
ridge.fit(X,train_y)
ridge_predict = ridge.predict(X_t)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X,train_y)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))


# ### Lasso Regression

# In[37]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(X,train_y)
lasso_predict = lasso.predict(X_t)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X,train_y)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))


# ### ElasticNet Regression

# In[38]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(X, train_y)
Elastic_pred = elasticnet.predict(X_t)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X,train_y)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))


# ### Support Vector Regression

# In[39]:


from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(X, train_y)

# Predicting a new result
svm_pred = svm_regressor.predict(X_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_t,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))


# ## 5. Select From Model  피쳐 제거 방법
# ---
# SelectFromModel을 통해 모델이 학습하면서 중요하다고 판단되는 Feature를 셀렉할 수 있습니다.
# 이때 모델은 사용자가 직접 정할 수 있습니다.

# ### [TODO] Select From Model을 통해 피쳐 셀렉을 수행하는 코드를 작성하세요.
# - `SelectFromModel` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#     - `estimator=LinearRegression()`
# - Min-Max Scaler를 통해 변환된 훈련 데이터로 `SelectFromModel`를 학습시킵니다.
# - Min-Max Scaler를 통해 변환된 훈련 데이터에 학습된 `SelectFromModel`를 적용합니다.
# - Min-Max Scaler를 통해 변환된 테스트 데이터에 학습된 `SelectFromModel`를 적용합니다.

# In[40]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression

selector = SelectFromModel(estimator=LinearRegression())
selector.fit(train_x, train_y) # selector 학습
selector.estimator_.coef_
X_train_select=selector.transform(ms_train) # Min-Max Scaler로 변환된 훈련 데이터 적용
select=selector.get_support()
X_test_select=selector.transform(ms_test) # Min-Max Scaler로 변환된 테스트 데이터 적용

select = pd.DataFrame(select, index=ms_train.columns)
select


# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(X_train_select, train_y)
lr_predict = lr.predict(X_test_select)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(X_train_select,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_test_select,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))


# In[42]:


from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(X_train_select)
poly_ftr = poly.transform(X_train_select)

poly_ftr_test = poly.transform(X_test_select)
plr = LinearRegression()
plr.fit(poly_ftr, train_y)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))


# In[43]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 3)
ridge.fit(X_train_select,train_y)
ridge_predict = ridge.predict(X_test_select)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_train_select,train_y)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_test_select,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))


# In[44]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(X_train_select,train_y)
lasso_predict = lasso.predict(X_test_select)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_train_select,train_y)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_test_select,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))


# In[45]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(X_train_select, train_y)
Elastic_pred = elasticnet.predict(X_test_select)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_train_select,train_y)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_test_select,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))


# In[46]:


from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(X_train_select, train_y)

# Predicting a new result
svm_pred = svm_regressor.predict(X_test_select)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_train_select,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_test_select,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))


# ## 6. MODEL SELECT + RFE
# ---
# Model Select와 RFE를 사용하여 Model Select에서 선정한 피쳐 중에 RFE로 한번 더 학습에 중요한 피처를 선정할 수 있습니다.
# 

# ### [TODO] RFE를 통해 피쳐 셀렉을 수행하는 코드를 작성하세요.
# - `RFE` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#     - `model`
#     - `6`
# - Model select를 통해 변환된 훈련 데이터(`X_train_select`)로 `RFE`를 학습시킵니다.
# - Model select를 통해 변환된 훈련 데이터(`X_train_select`)에 학습된 `RFE`를 적용합니다.
# - Model select를 통해 변환된 테스트 데이터(`X_test_select`)에 학습된 `RFE`를 적용합니다.

# In[47]:


from sklearn.feature_selection import RFE

model = LinearRegression()
rfe = RFE(model, 6)
rfe.fit(X_train_select, train_y) # rfe 학습
X_RFE = rfe.transform(X_train_select) # X_train_select 적용
X_RFE_t = rfe.transform(X_test_select) # X_test_select 적용


# In[48]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(X_RFE, train_y)
lr_predict = lr.predict(X_RFE_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(X_RFE,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_RFE_t,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))


# In[49]:


from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(X_RFE)
poly_ftr = poly.transform(X_RFE)

poly_ftr_test = poly.transform(X_RFE_t)
plr = LinearRegression()
plr.fit(poly_ftr, train_y)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))


# In[50]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha =3)
ridge.fit(X_RFE,train_y)
ridge_predict = ridge.predict(X_RFE_t)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(X_RFE,train_y)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))


# In[51]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(X_RFE,train_y)
lasso_predict = lasso.predict(X_RFE_t)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(X_RFE,train_y)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))


# In[52]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(X_RFE, train_y)
Elastic_pred = elasticnet.predict(X_RFE_t)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(X_RFE,train_y)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))


# In[53]:


from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(X_RFE, train_y)

# Predicting a new result
svm_pred = svm_regressor.predict(X_RFE_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_RFE,train_y)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_RFE_t,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))


# ## 7. Select K + Outlier Detection
# ---
# Univariate 피쳐 셀렉션한 최종 피처의 데이터로 이상치를 제거합니다 

# In[54]:


selectK = SelectKBest(score_func=f_classif, k=12)
selectK.fit(ms_train, train_y)
X_tr = selectK.transform(X_train)
X = selectK.transform(ms_train)
X_v = selectK.transform(X_val)
X_t = selectK.transform(ms_test)
TF = selectK.get_support()
TF = pd.DataFrame(TF, index=ms_train.columns)


# In[55]:


train_data_K = X.copy()
train_data_K = pd.DataFrame(train_data_K, index = train_y.index)
train_data_K = pd.concat([train_y,train_data_K], axis=1)


# 이상치 제거 알고리즘 IsolationForest을 통해 이상치를 탐지하고 이를 시각화 한 후 제거합니다.

# ### [TODO] Isolation Forest를 통해 이상치 제거를 수행하는 코드를 작성하세요.
# - `IsolationForest` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#     - `n_estimators=100`
#     - `contamination=0.01`
# - `train_data_K`로 `IsolationForest`를 학습시킵니다.

# In[56]:


from sklearn.ensemble import IsolationForest

# 이상치 제거 isolation forest model 설정
clf=IsolationForest(n_estimators=100, contamination = 0.01)


# In[57]:


#이상치 제거 피팅
clf.fit(train_data_K) # train_data_K로 clf 학습
pred = clf.predict(train_data_K)
train_data_K['anomaly']=pred
train_data_K = train_data_K.reset_index()
outliers=train_data_K.loc[train_data_K['anomaly']==-1]
outlier_index=list(outliers.index)
print(train_data_K['anomaly'].value_counts())


# In[58]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3) 
scaler = StandardScaler()
#normalize the metrics
train_data_K.set_index('Date', inplace = True)
X = scaler.fit_transform(train_data_K)
X_reduce = pca.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3")
# Plot the compressed data points
ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
# Plot x's for the ground truth outliers
ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
plt.show()


# In[59]:


# 이상치로 판단된 데이터를 제거합니다.
idx_nm_1 = train_data_K[train_data_K['anomaly'] == -1].index
train_data_outlier = train_data_K.drop(idx_nm_1)
del train_data_outlier['anomaly']


# In[60]:


train_y_outlier = train_data_outlier['Y']
train_x_outlier = train_data_outlier.copy()
del train_x_outlier['Y']
train_data_outlier_index = train_data_outlier.index
test_data_index = test_data.index


# In[61]:


X_t


# In[62]:


train_x_outlier


# In[63]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(train_x_outlier, train_y_outlier)
lr_predict = lr.predict(X_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(train_x_outlier,train_y_outlier)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_t,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))


# In[64]:


from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(train_x_outlier)
poly_ftr = poly.transform(train_x_outlier)

poly_ftr_test = poly.transform(X_t)
plr = LinearRegression()
plr.fit(poly_ftr, train_y_outlier)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))


# In[65]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha =3)
ridge.fit(train_x_outlier,train_y_outlier)
ridge_predict = ridge.predict(X_t)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(train_x_outlier,train_y_outlier)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))


# In[66]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(train_x_outlier,train_y_outlier)
lasso_predict = lasso.predict(X_t)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(train_x_outlier,train_y_outlier)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))


# In[67]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(train_x_outlier, train_y_outlier)
Elastic_pred = elasticnet.predict(X_t)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(train_x_outlier,train_y_outlier)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_t,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))


# In[68]:


from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(train_x_outlier, train_y_outlier)

# Predicting a new result
svm_pred = svm_regressor.predict(X_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(train_x_outlier,train_y_outlier)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_t,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))


# ## 8. Model Select + RFE + 이상치 제거(Isolation Forest)
# ---
# Model Select + RFE 통해 선택한 피쳐 데이터 셋에서 이상치 탐지 및 시각화를 하고 제거합니다.

# In[69]:


train_data_RFE = X_RFE.copy()
train_data_RFE = pd.DataFrame(train_data_RFE, index = train_y.index)
train_data_RFE = pd.concat([train_y,train_data_RFE], axis=1)


# ### [TODO] Isolation Forest를 통해 이상치 제거를 수행하는 코드를 작성하세요.
# - `IsolationForest` 객체를 생성합니다. 파라미터는 아래와 같이 지정합니다.
#     - `n_estimators=100`
#     - `contamination=0.01`
# - `train_data_RFE`로 `IsolationForest`를 학습시킵니다.

# In[70]:


from sklearn.ensemble import IsolationForest

# 이상치 제거 isolation forest model 설정
clf=IsolationForest(n_estimators=100, contamination = 0.01)


# In[71]:


#이상치 제거 피팅
clf.fit(train_data_RFE) # train_data_RFE로 clf 훈련
pred = clf.predict(train_data_RFE)
train_data_RFE['anomaly']=pred
train_data_RFE = train_data_RFE.reset_index()
outliers=train_data_RFE.loc[train_data_RFE['anomaly']==-1]
outlier_index=list(outliers.index)
print(train_data_RFE['anomaly'].value_counts())


# In[72]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3) 
scaler = StandardScaler()
#normalize the metrics
train_data_RFE.set_index('Date', inplace = True)
X = scaler.fit_transform(train_data_RFE)
X_reduce = pca.fit_transform(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3")
# Plot the compressed data points
ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
# Plot x's for the ground truth outliers
ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
plt.show()


# In[73]:


idx_nm_1 = train_data[train_data_RFE['anomaly'] == -1].index
train_data_outlier = train_data_RFE.drop(idx_nm_1)
del train_data_outlier['anomaly']


# In[74]:


train_data_outlier


# In[75]:


train_y_outlier = train_data_outlier['Y']
train_x_outlier = train_data_outlier.copy()
del train_x_outlier['Y']
train_data_outlier_index = train_data_outlier.index
test_data_index = test_data.index


# In[76]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

lr = LinearRegression()
lr.fit(train_x_outlier, train_y_outlier)
lr_predict = lr.predict(X_RFE_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, lr_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(lr.score(train_x_outlier,train_y_outlier)))
print("테스트 세트의 정확도 : {:.2f}".format(lr.score(X_RFE_t,test_y)))
print("MAE : {:.2f}".format(MAE(lr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lr_predict,test_y)))


# In[77]:


from sklearn.preprocessing import *
poly = PolynomialFeatures()
poly.fit(train_x_outlier)
poly_ftr = poly.transform(train_x_outlier)

poly_ftr_test = poly.transform(X_RFE_t)
plr = LinearRegression()
plr.fit(poly_ftr, train_y_outlier)
plr_predict = plr.predict(poly_ftr_test)
plt.scatter(test_y, plr_predict, alpha=0.4)
X1 = [7,8.5]
Y1 = [7,8.5]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Polynomial REGRESSION")
plt.show()
plinear_r2 = r2_score(test_y,plr_predict)
print("Polynomial 테스트 세트의 정확도 : {:.2f}".format(plinear_r2))
print("MAE : {:.2f}".format(MAE(plr_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(plr_predict,test_y)))
print("MSE : {:.2f}".format(MSE(plr_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(plr_predict,test_y)))
print("MPE : {:.2f}".format(MPE(plr_predict,test_y)))


# In[78]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha =3)
ridge.fit(train_x_outlier,train_y_outlier)
ridge_predict = ridge.predict(X_RFE_t)
plt.scatter(test_y, ridge_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Ridge REGRESSION")
plt.show()
print("ridge훈련 세트의 정확도 : {:.2f}".format(ridge.score(train_x_outlier,train_y_outlier)))
print("ridge테스트 세트의 정확도 : {:.2f}".format(ridge.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(ridge_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(ridge_predict,test_y)))
print("MSE : {:.2f}".format(MSE(ridge_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(ridge_predict,test_y)))
print("MPE : {:.2f}".format(MPE(ridge_predict,test_y)))


# In[79]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.0001)
lasso.fit(train_x_outlier,train_y_outlier)
lasso_predict = lasso.predict(X_RFE_t)
plt.scatter(test_y, lasso_predict, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Lasso REGRESSION")
plt.show()
print("lasso훈련 세트의 정확도 : {:.2f}".format(lasso.score(train_x_outlier,train_y_outlier)))
print("lasso테스트 세트의 정확도 : {:.2f}".format(lasso.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(lasso_predict,test_y)))
print("MAPE : {:.2f}".format(MAPE(lasso_predict,test_y)))
print("MSE : {:.2f}".format(MSE(lasso_predict,test_y)))
print("RMSE : {:.2f}".format(RMSE(lasso_predict,test_y)))
print("MPE : {:.2f}".format(MPE(lasso_predict,test_y)))


# In[80]:


from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(alpha =  0.001)
elasticnet.fit(train_x_outlier, train_y_outlier)
Elastic_pred = elasticnet.predict(X_RFE_t)
plt.scatter(test_y, Elastic_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("ElasticNet REGRESSION")
plt.show()
print("Elastic훈련 세트의 정확도 : {:.2f}".format(elasticnet.score(train_x_outlier,train_y_outlier)))
print("Elastic테스트 세트의 정확도 : {:.2f}".format(elasticnet.score(X_RFE_t,y_test)))
print("MAE : {:.2f}".format(MAE(Elastic_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(Elastic_pred,test_y)))
print("MSE : {:.2f}".format(MSE(Elastic_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(Elastic_pred,test_y)))
print("MPE : {:.2f}".format(MPE(Elastic_pred,test_y)))


# In[81]:


from sklearn.svm import SVR
svm_regressor = SVR(kernel = 'linear')
svm_regressor.fit(train_x_outlier, train_y_outlier)

# Predicting a new result
svm_pred = svm_regressor.predict(X_RFE_t)
import matplotlib.pyplot as plt
plt.scatter(test_y, svm_pred, alpha=0.4)
X1 = [7,8]
Y1 = [7,8]
plt.plot(X1, Y1, color="r", linestyle="dashed")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("SVM REGRESSION")
plt.show()
print("훈련 세트의 정확도 : {:.2f}".format(svm_regressor.score(train_x_outlier,train_y_outlier)))
print("테스트 세트의 정확도 : {:.2f}".format(svm_regressor.score(X_RFE_t,test_y)))
print("MAE : {:.2f}".format(MAE(svm_pred,test_y)))
print("MAPE : {:.2f}".format(MAPE(svm_pred,test_y)))
print("MSE : {:.2f}".format(MSE(svm_pred,test_y)))
print("RMSE : {:.2f}".format(RMSE(svm_pred,test_y)))
print("MPE : {:.2f}".format(MPE(svm_pred,test_y)))


# ---
