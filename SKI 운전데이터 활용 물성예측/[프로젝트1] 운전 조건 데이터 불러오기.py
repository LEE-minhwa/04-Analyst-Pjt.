#!/usr/bin/env python
# coding: utf-8

# # [프로젝트1] 운전 조건 데이터 불러오기

# ---

# 
# ## 프로젝트 목표
# ---
# - 모델 학습에 필요한 운전 조건 데이터 불러오기
# - 불러온 운전 조건 데이터의 기초적인 통계 정보 확인하기

# ## 프로젝트 목차
# ---
# 
# 1. **기본 패키지 불러오기:** 프로젝트 수행에 필요한 기본 패키지를 불러옵니다.
# 
# 2. **데이터 불러오기:** 운전 조건 데이터를 dataframe 형태로 불러오고 각 컬럼들의 데이터 타입과 기초 통계를 확인합니다. 
# 
# 

# ## 프로젝트 개요
# ---
# 데이터의 결측치를 확인하기 전, `Process_data.csv`의 데이터를 dataframe 형태로 불러오고 각 컬럼들의 데이터 타입과 기초 통계를 확인해봅니다.
# 
# 앞으로의 프로젝트 과정에서 활용할 데이터셋이므로 데이터에 익숙해지는 과정입니다. 

# ## 1. 기본 패키지 불러오기
# ---
# 프로젝트 수행에 필요한 추가 패키지를 설치하고 기존 설치되어있는 패키지를 가져옵니다.

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
import sklearn 
pd.options.display.max_rows = 80
pd.options.display.max_columns = 80


# ## 2. 데이터 불러오기
# ---

# ### [TODO] pandas를 통해 `Process_data.csv` 파일을 읽는 코드를 작성하세요.

# In[4]:


df = pd.read_csv('./Process_data.csv')
df


# ### 2.1 데이터 피쳐 확인
# ---

# ### [TODO] 데이터의 column들을 확인하는 코드를 작성하세요.

# In[5]:


df.columns


# ### [TODO] 'Unnamed: 66'이라는 이름의 column을 제거하는 코드를 작성하세요.

# In[6]:


df = df.drop('Unnamed: 66', axis = 1)


# ### 2.2 데이터 타입 확인 및 처리

# In[7]:


df.dtypes 


# In[8]:


# "x62" 컬럼의 값에서 %를 제거하고 objecct type에서 float type으로 바꿔줍니다.

df["x62"]
df["x62"] = df['x62'].str.strip("%")
df["x62"]
df["x62"] = df["x62"].astype('float')


# ### 2.3 데이터 기초 통계 확인
# ---

# ### [TODO] `df` dataframe을 묘사(describe)하는 코드를 작성하세요.
# - 출력되는 값들은 소수점 셋 째자리까지 나오게 반올림하세요.

# In[9]:


df.describe().round(3)


# ### 2.4 Date Index 설정
# ---

# In[10]:


# Date는 feature로 사용되지는 않으나 데이터에 계속 남겨두기 위해 index로 활용
df_date = df['Date']
df = df.set_index("Date")


# ### 2.5 Train/Test Data Set 분리
# ---

# In[11]:


train_data = df.iloc[0:691,:] #17년 12월 31일
train_data
test_data = df.iloc[691:,:] #18년 4월 22일
test_data


# ---
