#!/usr/bin/env python
# coding: utf-8

# In[772]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython import display
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style='darkgrid')


# In[773]:


df = pd.read_csv('./vehicles.csv')


# In[774]:


df.head()


# In[775]:


df.columns


# ### Drop Columns that aren't Relevant to the Analysis

# In[776]:


df = df.drop(["url","region_url","VIN","image_url","county","lat","long","id","posting_date"],axis=1)


# In[777]:


nans = (df.isnull().sum().sort_values(ascending=False)/(len(df))).to_frame()
#nans.plot(kind='bar',title="Missing Values Summary",figsize=(8,6))


# In[778]:


nans['count'] = nans[0]
nans.reset_index(inplace=True)


# In[779]:


plt.figure(figsize = (10,10))
sns.barplot(x = 'index', y= 'count', data=nans)
plt.xticks(rotation=90)
plt.show()


# In[780]:


df.head(50)


# In[781]:


def caps(s):
    if type(s) is str:
        return s.capitalize()
    else:
        return None


# In[782]:


df['manufacturer'] = df['manufacturer'].apply(caps)


# ### Create Column that Concatenates Make and Model Data

# In[783]:


df['manufacturer'].replace(to_replace='Rover',value='Land rover',inplace=True)
df['manufacturer'].replace(to_replace='Mercedes-benz',value='Mercedes',inplace=True)


# In[ ]:





# In[784]:


df.describe()


# In[ ]:





# In[785]:


df['manufacturer'].value_counts().head()


# ### Drop Null Values From Key Columns

# In[786]:


df1 = df.copy()
df1.dropna(axis=0,how='any',subset=['manufacturer','title_status','odometer','year','price'],inplace=True)


# In[787]:


df1['year'] = df1['year'].apply(lambda x: int(x))


# In[788]:


filters = (df1['odometer'] > 5000.0) & (df1['odometer'] < 500000.0) & (df1['year'] >= 1955) & (df1['year'] < 2021) & (df1['price'] > 1500.0) & (df1['price'] < 99999) & (df1['manufacturer'] != 'harley-davidson') & (df1['cylinders'] != 'other')


# In[789]:


df1 = df1[filters]


# In[790]:


df1['manufacturer'].replace(to_replace='mercedes-benz',value='mercedes',inplace=True)


# In[791]:


#df1 = df1.groupby('manufacturer').filter(lambda x : len(x)>20)


# In[792]:


df1['year'].median()


# In[793]:


df1['price'].median()


# In[794]:


df1['odometer'].median()


# In[795]:


df1.hist(figsize=(10,10))


# In[796]:


#df['manufacturer'].value_counts().head(30).plot(kind='pie',colors=colors,title='manufacturer',figsize=(10,10))


# In[797]:


df['price'].mean()


# In[798]:


#df['manufacturer'].value_counts().head(25)


# In[799]:


plt.figure(figsize = (10,10))
sns.countplot(x = 'manufacturer',data = df1,order= df1['manufacturer'].value_counts().head(20).index)
plt.xticks(rotation=60)
plt.show()


# In[800]:


plt.figure(figsize = (5,5))
sns.countplot(x = 'drive',data = df1,order= df1['drive'].value_counts().index)
plt.xticks(rotation=60)
plt.show()


# In[801]:


plt.figure(figsize = (5,5))
sns.countplot(x = 'fuel',data = df1,order= df1['fuel'].value_counts().index)
plt.xticks(rotation=60)
plt.show()


# In[802]:


plt.figure(figsize = (5,5))
sns.countplot(x = 'condition',data = df1,order= df1['condition'].value_counts().index)
plt.xticks(rotation=60)
plt.show()


# In[803]:


plt.figure(figsize = (6,6))
sns.countplot(x = 'type',data = df1,order= df1['type'].value_counts().index)
plt.xticks(rotation=60)
plt.show()


# In[804]:


plt.figure(figsize = (5,5))
sns.countplot(x = 'size',data = df1,order= df1['size'].value_counts().index)
plt.xticks(rotation=60)
plt.show()


# ### Age vs. Listing Price

# In[805]:


df2 = df1.copy()


# In[806]:


plt.figure(figsize=(13,8))
sns.boxplot(y="price", x="year",data=df2,showfliers = False)
plt.xticks(rotation=90)
plt.show()


# ### View Correlations Between Numerical Data

# In[807]:


filt2 = (df2['year'] > 1999) & (df1['year'] < 2021) & (df2['manufacturer'] != 'Harley-davidson')


# In[808]:


df2 = df2[filt2]


# In[809]:


plt.figure(figsize=(8,6))
sns.heatmap(df2.corr(method='pearson'),annot=True,fmt='.2f')
plt.show()


# ### Create New Dataframe Using Features Relevant for Our Analysis

# In[810]:


df3 = df2[['price','manufacturer','condition','cylinders','fuel','odometer','year',
          'title_status','transmission','drive','size','type','paint_color']]


# In[849]:


df3=df3.copy()
df3['age'] = (2022 - df3['year'])


# In[850]:


#df3.describe()


# In[851]:


df3.shape


# ### Filter Make / Manufacturer Data With Less than 50 Entries

# In[852]:


df3 = df3.groupby('manufacturer').filter(lambda x : len(x)>50)


# ### Group by Fuel Type and Manufacturer

# In[853]:


fuel_group = df3.groupby(['fuel'])


# In[854]:


manu_group = df3.groupby(['manufacturer'])


# ### View Price/Age and Price/Mileage correlations by group

# In[855]:


age_corr_df = df3[['age','price']].corr(method='pearson')
odo_corr_df = df3[['odometer','price']].corr(method='pearson')


# In[856]:


age_corr = age_corr_df.iloc[0,1]
odo_corr = odo_corr_df.iloc[0,1]


# In[857]:


age_corr


# In[858]:


age_corr_manu = manu_group[['age','price']].corr(method='pearson')
odo_corr_manu = manu_group[['odometer','price']].corr(method='pearson')
age_corr_fuel = fuel_group[['age','price']].corr(method='pearson')
odo_corr_fuel = fuel_group[['odometer','price']].corr(method='pearson')


# In[859]:


#age_corr_fuel


# In[860]:


fuel1 = age_corr_fuel.reset_index()
fuel1.set_index('fuel',inplace=True)


# In[861]:


fuel2 = odo_corr_fuel.reset_index()
fuel2.set_index('fuel',inplace=True)


# In[862]:


age1 = age_corr_manu.reset_index()
age1.set_index('manufacturer',inplace=True)


# In[863]:


odo1 = odo_corr_manu.reset_index()
odo1.set_index('manufacturer',inplace=True)


# In[864]:


fuel1 = fuel1[fuel1['level_1'] == 'age']


# In[865]:


fuel2 = fuel2[fuel2['level_1'] == 'odometer']


# In[866]:


fuel1 = fuel1.rename(columns={'price':'age_corr'})


# In[867]:


fuel2 = fuel2.rename(columns={'price': "odo_corr"})


# In[868]:


fuelplt = pd.concat([fuel1,fuel2],axis=1)


# In[869]:


#fuelplt


# In[870]:


fuelplt['Miles'] = (fuelplt['odo_corr'] - odo_corr) / odo_corr


# In[871]:


fuelplt['Age'] = (fuelplt['age_corr'] - age_corr) / age_corr


# In[872]:


age1 = age1[age1['level_1'] == 'age']
odo1 = odo1[odo1['level_1'] == 'odometer']


# In[873]:


odo1 = odo1.rename(columns={'price':'odo-price'})


# In[874]:


age1 = age1.rename(columns={'price':'age-price'})


# In[875]:


dfplt = pd.concat([odo1,age1],axis=1)


# In[876]:


#dfplt.head()


# In[877]:


dfplt['Miles'] = (dfplt['odo-price'] - odo_corr) / odo_corr


# In[878]:


dfplt['Age'] = (dfplt['age-price'] - age_corr) / age_corr


# In[879]:


dfplt = dfplt.sort_values(by='Miles',ascending=False)


# In[880]:


#dfplt.head()


# In[881]:


dfpltmax = dfplt.nlargest(10,columns='Age').sort_values(by='Age')
dfpltmin = dfplt.nsmallest(10,columns='Age').sort_values(by='Age',ascending=False)


# In[895]:


ax = dfpltmax[['Age','Miles']].plot(kind='barh',xlim=(-.6,.6),color=['b','g'],figsize=(10,10))
ax.set_xlabel("Normalized Difference From the Mean")
ax.set_title("Price Depreciation by Manufacturer (10 Highest)")


# In[890]:


ax = dfpltmin[['Age','Miles']].plot(kind='barh',xlim=(-1,1),color=['b','g'],figsize=(10,10))
ax.set_xlabel("Normalized Difference From the Mean")
ax.set_title("Price Depreciation by Manufacturer (10 Lowest)")


# In[884]:


fuelplt = fuelplt.sort_values(by='Age',ascending=True)


# In[885]:


fuelplt.drop(['other'],axis=0,inplace=True)


# In[894]:


ax = fuelplt[['Age','Miles']].plot(kind='barh',title="Price Depreciation by Fuel Type",xlim=(-.6,.6),color=['b','g']
                              ,figsize=(10,10))
ax.set_xlabel("Normalized Difference From the Mean")


# In[ ]:




