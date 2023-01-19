#!/usr/bin/env python
# coding: utf-8

# # [프로젝트2] 운전 조건 데이터 결측치 확인 및 처리하기

# ---

# ## 프로젝트 목표
# ---
# - 데이터셋 내에 존재하는 결측치 확인하기
# - 단일 대치법(Mean)과 다중 대치법(KNN, MICE, MissForest)으로 결측치 처리하기

# ## 프로젝트 목차
# ---
# 
# 1. **프로젝트 준비하기:** 프로젝트 수행에 필요한 추가 패키지를 설치하고 기존 패키지를 가져옵니다.
# 
# 2. **데이터 불러오고 전처리하기:** 운전 조건 데이터를 불러오고, Train/Test로 분리합니다.
# 
# 3. **결측치 확인하기:** 결측치를 확인하고 결측치가 존재하는 피처를 제거합니다. 
# 
# 4. **결측치 처리하기:** 단일 대치법과 다중 대치법으로 결측치를 대치하고, 대치된 데이터를 확인합니다.
# 
# 

# ## 프로젝트 개요
# ---
# 단일 대치법과 다중 대치법으로 결측치를 대치해 본 후, 대치된 데이터를 확인합니다.
# 
# 
# 데이터 확인을 통해 어떤 방식으로 대치된 데이터를 학습 데이터로 결정할지 생각해봅시다.
# 

# ## 1. 프로젝트 준비하기
# ---
# 
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

# 데이터를 불러와 기본적인 전처리를 진행합니다. 

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


# 기초 전처리가 완료된 데이터를 학습(Train), 테스트(Test) 용으로 분할합니다.

# In[8]:


train_data = df.iloc[0:691,:] #17년 12월 31일
train_data
test_data = df.iloc[691:,:] #18년 4월 22일
test_data


# ---

# ## 3. 결측치 확인하기
# ---
# 
# 

# ### [TODO] `train_data`와 `test_data`에서 각 column 별로 결측치인 데이터의 개수가 몇개인지 구하는 코드를 작성하세요.

# In[9]:


train_data.isnull().sum()


# In[10]:


test_data.isnull().sum()


# ### [TODO] 훈련 데이터와 테스트 데이터에서 결측치가 포함된 column은 모두 제거하는 코드를 작성하세요.
# - `train_data`와 `test_data`를 복사한 `train_del`과 `test_del` 변수에 적용합니다.

# In[11]:


train_del = train_data.copy()
test_del = test_data.copy()

train_del = train_del.dropna(axis=1)
test_del = test_del.dropna(axis=1)


# 아래 변수는 결측치 처리 이후 컬럼명을 설정하기 위한 리스트입니다. 

# In[12]:


full_columns=['Y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11',
       'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21',
       'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31',
       'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41',
       'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 'x51',
       'x52', 'x53', 'x54', 'x55', 'x56', 'x57', 'x58', 'x59', 'x60', 'x61',
       'x62', 'x63', 'x64']


# ## 4. 결측치 처리하기(단일 대치, 다중 대치)
# ---
# 
# 단일 대치법과 다중 대치법으로 결측치를 대치하도록 합니다.
# 

# ### 4.1 결측치 단일 대치(Mean) 
# ---
# 평균으로 대치하여 결측치를 처리합니다.

# ### [TODO] 평균 값으로 결측치를 대치하는 코드를 작성하세요.
# - 결측치 처리를 위한 `SimpleImputer` 객체를 `mean` strategy를 통해 생성합니다.
# - 훈련 데이터로 위에서 생성한 객체를 학습시킵니다.
# - 훈련 데이터와 테스트 데이터에 훈련된 `SimpleImputer` 객체를 적용합니다.

# In[13]:


# 결측치 단일 대치 (평균으로 대치하여 결측치를 처리합니다.)
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy='mean') #'median'을 쓰면 중앙값 사용
imp_mean.fit(train_data) # imp_mean 학습
mean_train = imp_mean.transform(train_data) # 학습된 결측치 대치 적용
mean_test = imp_mean.transform(test_data)  # 학습된 결측치 대치 적용 

# 위 코드 적용 이후 Array 구조로 변하므로, 다시 DataFrame 형태로 변환해주는 코드
mean_train = pd.DataFrame(mean_train, columns=full_columns, index = train_data.index) 
mean_test = pd.DataFrame(mean_test, columns=full_columns, index = test_data.index)


# In[14]:


mean_train


# In[15]:


mean_test


# ### 4.2 결측치 다중 대치(KNN, MICE, MissForest)
# ---

# ### 결측치 다중 대치 - KNN

# ### [TODO] K-Nearest Neighbors Imputer를 통해 결측치를 대치하는 코드를 작성하세요.
# - 결측치 처리를 위한 `KNNImputer` 객체를 이웃의 수를 3개로 설정하여 생성합니다.
# - 훈련 데이터로 위에서 생성한 객체를 학습시킵니다.
# - 훈련 데이터와 테스트 데이터에 훈련된 `KNNImputer` 객체를 적용합니다.

# In[16]:


from sklearn.impute import KNNImputer

KNN = KNNImputer(n_neighbors =3)
KNN.fit(train_data) # KNN 학습
df_knn_train = KNN.transform(train_data)
df_knn_test = KNN.transform(test_data)

# 위 코드 적용 이후 Array 구조로 변하므로, 다시 DataFrame 형태로 변환해주는 코드
df_knn_train = pd.DataFrame(df_knn_train, columns=full_columns, index = train_data.index)
df_knn_test = pd.DataFrame(df_knn_test, columns=full_columns, index = test_data.index)


# In[17]:


df_knn_train


# In[18]:


df_knn_test


# 
# ### 결측치 다중 대치  - MICE

# ### [TODO] MICE Imputer를 통해 결측치를 대치하는 코드를 작성하세요.
# - 학습 데이터의 **값**들에 `mice`를 적용합니다.
# - 테스트 데이터의 **값**들에 `mice`를 적용합니다.

# In[19]:


from impyute.imputation.cs import mice
df_mice_train=mice(train_data.values)
df_mice_test=mice(test_data.values)

# 위 코드 적용 이후 Array 구조로 변하므로, 다시 DataFrame 형태로 변환해주는 코드
df_mice_train = pd.DataFrame(df_mice_train, columns=full_columns, index = train_data.index)
df_mice_test = pd.DataFrame(df_mice_test, columns=full_columns, index = test_data.index)


# In[20]:


df_mice_train


# In[21]:


df_mice_test


# ### 결측치 다중 대치 - MissForest

# ### [TODO] MissForest Imputer를 통해 결측치를 대치하는 코드를 작성하세요.
# - 결측치 처리를 위한 `MissForest` 객체를 생성합니다.
# - 훈련 데이터로 위에서 생성한 객체를 학습시킵니다.
# - 훈련 데이터와 테스트 데이터에 훈련된 `MissForest` 객체를 적용합니다.

# In[22]:


from missingpy import MissForest

miss_imputer = MissForest()
miss_imputer.fit(train_data) # miss_imputer 학습
df_miss_train = miss_imputer.transform(train_data)
df_miss_test = miss_imputer.transform(test_data)

# 위 코드 적용 이후 Array 구조로 변하므로, 다시 DataFrame 형태로 변환해주는 코드
df_miss_train = pd.DataFrame(df_miss_train, columns=full_columns, index = train_data.index)
df_miss_test = pd.DataFrame(df_miss_test, columns=full_columns, index = test_data.index)


# In[23]:


df_miss_train


# In[24]:


df_miss_test


# ### 결측치 제거 이후 학습할 데이터 결정

# 아래는 KNN 방법으로 대치된 데이터를 활용할 경우입니다.

# In[25]:


# KNN 기반으로 결측치가 대치된 데이터를 이용할 경우
test_data=df_knn_test.copy()
train_data=df_knn_train.copy()


# ### 통계량 확인

# KNN 방법으로 결측치가 대치된 데이터의 기초 통계량을 확인합니다.

# In[26]:


df_knn_train.describe()


# ---
