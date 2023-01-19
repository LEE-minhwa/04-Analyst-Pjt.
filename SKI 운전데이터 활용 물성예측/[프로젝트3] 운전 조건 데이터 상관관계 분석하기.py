#!/usr/bin/env python
# coding: utf-8

# # [프로젝트3] 운전 조건 데이터 상관관계 분석하기

# ---

# ## 프로젝트 목표
# ---
# - 데이터 내에 존재하는 상관관계 확인하기
# - 데이터 상관관계를 분석하고 상관계수에 따라 각 변수 시각화하기

# ## 프로젝트 목차
# ---
# 
# 1. **프로젝트 준비하기:** 프로젝트 수행에 필요한 추가 패키지를 설치하고 기존 패키지를 가져옵니다.
# 
# 2. **데이터 불러오고 전처리하기:** 운전 조건 데이터를 불러오고 Train/Test로 분리합니다. 분리한 데이터의 결측치를 KNN 단일 대치법으로 대치합니다.
# 
# 3. **데이터 상관관계 분석하기:** 변수들의 상관관계를 확인하고 상관성이 높은 두 변수가 어떤 특징을 갖고 있는지 시각화하여 확인합니다. 
# 
# 
# 

# ## 프로젝트 개요
# ---
# 데이터 내 변수들의 상관관계를 확인하고 상관계수가 0.9 초과의 두 변수를 시각화하여 어떤 특징이 이와 같은 결과를 도출했는지 이해해봅니다.
# 
# 

# ## 1. 프로젝트 준비하기
# ---
# 
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


# ---

# ## 3. 데이터 상관관계 분석하기
# ---

# 데이터의 각 변수들의 상관관계를 분석해봅니다.

# ### 3.1 데이터 종속변수 독립변수 분리
# ---
# 
# 데이터의 상관관계 분석을 위하여 종속변수(Y)와 독립변수(X)를 분리합니다. 

# In[13]:


train_y = train_data['Y']     # 종속변수 (target variable) Y 분리
train_x = train_data.copy()
del train_x['Y']              # Y를 제거한 나머지 데이터는 독립변수 (input variable) 에 해당
# test data도 마찬가지 작업 수행
test_y = test_data['Y']
test_x = test_data.copy()
del test_x['Y']


# ### 3.2 피처 특징별 상관 분석, 상관계수 확인
# ---

# In[14]:


# 데이터의 feature에 대한 사전정보가 있다면, 비슷한 카테고리의 feature끼리 나누어서 상관분석을 진행하는 것이 좋음
train_x29=train_x.iloc[:,:29]
train_x46=train_x.iloc[:,29:46]
train_x60=train_x.iloc[:,46:61]
train_x64=train_x.iloc[:,61:64]

# 사전지식이 없다면 corr = train_data.corr(), 즉 모든 feature들을 한번에 상관분석 수행


# ### [TODO] 피처별 상관관계를 구하는 코드를 작성하세요.
# - `train_x29` 에 대한 상관관계를 구하는 코드를 작성하세요.
# - `train_x46` 에 대한 상관관계를 구하는 코드를 작성하세요.
# - `train_x60` 에 대한 상관관계를 구하는 코드를 작성하세요.
# - `train_x64` 에 대한 상관관계를 구하는 코드를 작성하세요.

# In[15]:


corr29 = train_x29.corr()
corr46 = train_x46.corr()
corr60 = train_x60.corr()
corr64 = train_x64.corr()
corr29


# 위 결과는 `corr29` (~x29 피쳐까지)의 상관관계를 보여주고 있습니다. 

# ### [TODO] `corr29`에서 상관계수의 절댓값이 0.9 초과인 피처들만 확인하는 코드를 작성하세요.

# In[16]:


#상관계수의 절댓값이 0.9 초과인 변수만을 확인하는 방법
condition = pd.DataFrame(columns=corr29.columns, index=corr29.columns) 

for i in range(0,29):
    condition.iloc[:,[i]] = corr29[abs(corr29.iloc[:,[i]]) > 0.9].iloc[:,[i]]

condition


# ### 3.3 상관분석 및 데이터 시각화하기
# ---
# 상관관계를 확인하고 상관성이 높은 두 변수가 어떤 특징을 갖고있는지 시각화를 통해 확인합니다.
# 

# In[17]:


# kdeplot(Kernel Density Estimator plot)은 히스토그램보다 더 부드러운 형태로 분포 곡선을 보여줌
# sns => seaborn 패키지
sns.kdeplot(x=train_x['x1'])
sns.kdeplot(x=train_x['x4'],color='r')   #디폴트 색상은 파란색, color='r' 은 붉은색 적용


# In[18]:


# distplot => 히스토그램과 kdeplot을 같이 그려줌
sns.distplot(x=train_x['x13'])
sns.distplot(x=train_x['x14'])


# In[19]:


# violinplot => x축이 feature값, y축이 밀도. feature값의 분포를 보여줌.
sns.violinplot(x=train_x['x23'], figsize=(20,20))
sns.violinplot(x=train_x['x25'], figsize=(20,20),color='r')


# In[20]:


train_x['x25'].plot()
train_x['x23'].plot()

# x축이 겹치지 않도록 회전시키기  
plt.xticks(rotation=50)


# ### [TODO] `corr46`에서 상관계수의 절댓값이 0.9 초과인 피처들만 확인하는 코드를 작성하세요.

# In[21]:


condition = pd.DataFrame(columns=corr46.columns, index=corr46.columns)

for i in range(0,17):
    condition.iloc[:,[i]] = corr46[abs(corr46.iloc[:,[i]]) > 0.9].iloc[:,[i]]

condition


# In[22]:


sns.lmplot(x='x39', y='x40', data= train_x)


# In[23]:


sns.violinplot(x=train_x['x39'], figsize=(20,20))
sns.violinplot(x=train_x['x40'], figsize=(20,20),color='r')


# In[24]:


sns.violinplot(x=train_x['x44'], figsize=(20,20))
sns.violinplot(x=train_x['x45'], figsize=(20,20),color='r')


# In[25]:


sns.distplot(x=train_x['x39'])
sns.distplot(x=train_x['x40'],color='r')


# ### [TODO] `corr60`에서 상관계수의 절댓값이 0.9 초과인 피처들만 확인하는 코드를 작성하세요.

# In[26]:


condition = pd.DataFrame(columns=corr60.columns, index=corr60.columns)

for i in range(0,corr60.shape[1]):
    condition.iloc[:,[i]] = corr60[abs(corr60.iloc[:,[i]]) > 0.9].iloc[:,[i]]

condition


# In[27]:


sns.kdeplot(x=train_x['x53'])
sns.kdeplot(x=train_x['x60'],color='r')


# In[28]:


sns.kdeplot(x=train_x['x54'])
sns.kdeplot(x=train_x['x55'],color='r')
sns.kdeplot(x=train_x['x57'], color='g')


# ### [TODO] `corr64`에서 상관계수의 절댓값이 0.7 초과인 피처들만 확인하는 코드를 작성하세요.

# In[29]:


condition = pd.DataFrame(columns=corr64.columns, index=corr64.columns)

for i in range(0,corr64.shape[1]):
    condition.iloc[:,[i]] = corr64[abs(corr64.iloc[:,[i]]) > 0.7].iloc[:,[i]]
condition


# In[30]:


sns.kdeplot(x=train_x['x62'])
sns.kdeplot(x=train_x['x63'],color='r')


# *상관관계가 높은 경우 삭제하는 것도 방법이지만 피처의 정보를 알 수 없기때문에 다 지우게 되면 모델에 중요한 피처를 제거하게 될 수 도 있습니다.

# ---
