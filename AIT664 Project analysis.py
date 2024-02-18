#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from datetime import datetime


# In[2]:


# Loading the dataset
df = pd.read_csv("C:\\Users\\bhavi\\OneDrive\\Desktop\\AIT 664\\Crime.csv")
df


# In[3]:


df.info()


# In[4]:


df


# In[5]:


# Pre-processing step
# Checking null values and their counts
null_counts = df.isna().sum()
print(null_counts)


# In[6]:


# Replacing all null values with 0 in the entire dataset
df.fillna(0, inplace=True)


# In[7]:


null_counts = df.isna().sum()
print(null_counts)


# In[8]:


import pandas as pd
from pandasql import sqldf

query = "Select count(*) from df"

df_dql_query = sqldf(query)
df_dql_query.head()


# In[9]:


df.describe(include="all", exclude = None)


# In[10]:


df = df.rename(columns={'Incident ID':'I_ID'})
ax = df.groupby(['Crime Name1']).I_ID.count().reset_index()
print(ax)
plt.rcParams['figure.figsize'] = [8,6]
ax.plot.bar(x="Crime Name1",y="I_ID")
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Crime Name')
plt.ylabel('Count of Crime Incidents')
plt.title('Crime Vs Incident Occurences',fontsize = 16, fontweight = "bold")
plt.show()


# In[11]:


bx = df.groupby(['Crime Name2']).I_ID.count().reset_index()
print(bx)
plt.rcParams['figure.figsize'] = [12,8]
bx.plot.bar(x="Crime Name2",y="I_ID",color='purple')
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Crime Type')
plt.ylabel('Count of Crime Incidents')
plt.title('Crime (Types) Vs Incident Occurances',fontsize = 16, fontweight = "bold")
plt.show()


# In[12]:


df = df.rename(columns={'Dispatch Date / Time':'crime_year'})
zx = df['crime_year'].str[-4:]
#print(zx)
yx = df.groupby(zx).I_ID.count().reset_index()
print(yx)
plt.rcParams['figure.figsize'] = [8,6]
yx.plot.bar(x="crime_year",y="I_ID",color="green")
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Crime Years')
plt.ylabel('Count of Crime Incidents')
plt.title('Crime (Yrs) Vs Count of Crime Incidents',fontsize = 16, fontweight = "bold")
plt.show()


# In[13]:


wx = df.groupby(['Crime Name3']).I_ID.count().reset_index()
print(wx)
plt.rcParams['figure.figsize'] = [25,22]
wx.plot.line(x="Crime Name3",y="I_ID",color='brown')
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 10)
plt.xlabel('Crime Type')
plt.ylabel('Count of Crime Incidents')
plt.title('Type of Crime Vs Incident Occurances',fontsize = 16, fontweight = "bold")
plt.show()


# In[14]:


cx = df.groupby(['Victims']).I_ID.count().reset_index()
print(cx)
plt.rcParams['figure.figsize'] = [8,6]
cx.plot.line(x="Victims",y="I_ID",color="blue")
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Number of Victims')
plt.ylabel('Count of Crime Incidents')
plt.title('Victims Vs Crime Occurrences',fontsize = 16, fontweight = "bold")
plt.show()


# In[15]:


dx = df.groupby(['Place']).I_ID.count().reset_index()
print(dx)
plt.rcParams['figure.figsize'] = [18,15]
dx.plot.bar(x="Place",y="I_ID",color="pink")
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.xlabel('Crime Place')
plt.ylabel('Count of Crime Incidents')
plt.title('Crime Place Vs Incident Occurrences',fontsize = 16, fontweight = "bold")
plt.show()


# In[17]:


df = df.rename(columns={'Start_Date_Time':'crime_year'})
df = df.rename(columns={'Incident ID':'case_num'})
df
#gx = df['crime_year'].str[-4:]
#print(gx)


# In[18]:


# Loading the dataset
df_crime_data = pd.read_csv("C:\\Users\\bhavi\\OneDrive\\Desktop\\AIT 664\\Crime.csv")
df_crime_data


# In[19]:


df_crime_data.info()


# In[49]:


df_crime_data = df_crime_data.rename(columns={'Start_Date_Time':'crime_year'})
df_crime_data = df_crime_data.rename(columns={'Incident ID':'case_num'})
gx = df_crime_data['crime_year'].str[-4:]
print(gx)


# In[50]:


#pd.to_numeric(gx).astype(int)
gx = gx.astype(str).astype(int)
gx.dtypes


# In[51]:


gx.info()


# In[52]:


bx = df_crime_data.groupby(gx).case_num.count().reset_index()
print(bx)
yr = np.array(bx['crime_year']).reshape(-1,1)
rn = np.array(bx['case_num']).reshape(-1,1)


# In[53]:


plt.xlabel('Crime Years')
plt.ylabel('Count of Incident numbers')
plt.scatter(yr,rn)
plt.show()


# In[54]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
clf = model.fit(yr,rn)
predictions = model.predict(yr)
plt.plot(yr,predictions,color='green')
plt.xlabel('Crime Years')
plt.ylabel('Count of Incident IDs')
plt.show()


# In[55]:


df2 = [[2024],[2025],[2026],[2027]]
predicted_years = model.predict(df2)
print(predicted_years)


# In[ ]:




