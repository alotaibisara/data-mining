#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
from matplotlib import pyplot as plt
 


# In[2]:


df=pd.read_csv('AB_NYC_2019.csv')
df.head(2)


# # Show, which attributes (variables), have missing values.

# In[3]:


name =df.isnull().sum().index
counts=df.isnull().sum()
nul_values={}
for x,y in zip(name,counts):
    if y>0:
        nul_values[x]= y

pd.DataFrame(nul_values,index=['nul_values']) 
        


# # Print out on the screen the total number of unique values of the categorical
# attributes: (minimum_nights, neighbourhood, room_type, price, number_of_reviews,
# availability_365)
#  Print out unique

# In[4]:


attributes=['minimum_nights', 'neighbourhood', 'room_type', 'price', 'number_of_reviews','availability_365']
total_unique_number=0

for i in attributes:
    if df[i].dtype == 'object':
        total_unique_number+=df[i].nunique()
print('Total number of unique values  = ',total_unique_number)


#  # Print out unique values in neighbourhood_group with their frequencies

# In[5]:


df.neighbourhood_group.value_counts()


# # Plot the distribution of categorical attribute room type  

# In[6]:


# using seaborn library
distribution=df.room_type.value_counts(normalize=True)
 
sns.barplot(x=distribution.index,y=distribution)


# In[7]:


# using matplotlib.plt libraray
plt.bar(distribution.index,distribution ,color=['red','orange','cyan'])


# # Plot in a Pie Chart the distribution of neighbourhood_group using availability_365 attribute (with y='availability_365').
# 

# In[8]:


df.neighbourhood_group.value_counts().plot.pie(  y=df['availability_365'] )


# In[9]:


plt.figure(figsize=(16,6))
X=df.neighbourhood_group.value_counts()
plt.pie(x=X,  labels=df.neighbourhood_group.unique() )
 
plt.legend(df.neighbourhood_group.unique() ,
          title ="neighbourhood_group",
          loc ="upper right",
          bbox_to_anchor =(1, 0, 0.5, 1))
plt.show()


# # Check whether price and number_of_reviews have outliers or not

# In[10]:


plt.figure(figsize=(6,6))
plt.boxplot(x=df.price,labels=['price'] )


# In[11]:


plt.scatter(df.price,y=df.index)


# In[12]:


# from the aboive figure you can notice there  is outliers in price value 


# In[13]:


plt.boxplot(x=df.number_of_reviews,labels=['number_of_reviews'] )


# In[14]:


plt.figure(figsize=(6,6))
sns.boxplot(x=df.number_of_reviews)


# In[15]:


plt.scatter(df.number_of_reviews,y=df.index)


# In[16]:


# from the aboive figure you can notice there  is outliers in number_of_reviews value 


# # Plot the distribution of the categorical attribute room type vs calculated_host_listings_count

# In[17]:



plt.scatter(df.room_type.astype(str), df.calculated_host_listings_count)
plt.margins(x=0.5)
plt.show()


# In[ ]:





# In[18]:


sns.stripplot(x=df['room_type'] , y=df['calculated_host_listings_count'] )


# In[19]:


sns.catplot(x="room_type", y="calculated_host_listings_count", kind="box", data=df)


# # Since (reviews_per_month) is continuous, the mean will be used to handle missed data in this attribute.

# In[20]:


temp=pd.DataFrame()
temp['reviews_per_month']=df.reviews_per_month
temp['reviews_per_month_mean']=df.reviews_per_month.fillna(df.reviews_per_month.mean())


# In[21]:


temp


# In[22]:


#withe zero 


# In[23]:


pd.concat([df.reviews_per_month.fillna(0),df.reviews_per_month],axis=1)


# In[24]:


df.reviews_per_month.fillna(0,inplace=True)


# # To handle missed data of both name and host_name, which are categorical, a new class called (global constant) will be used.

# In[25]:


df[['name','host_name']].isnull().sum()


# In[26]:


df['name'].fillna('global constant',inplace=True)
df['host_name'].fillna('global constant',inplace=True)


# In[27]:


df.isnull().sum()


# # Check whether the category values of (room_type), is consistent. If they are inconsistent, unify categories.

# In[28]:


df.room_type.unique()


# # Since the (room_type) variable has only 3 levels ("Entire home/apt", "Private room",
# "Shared room"), you decided to convert the variable values into a numeric discrete
# attribute with the value 0 for "Entire home/apt", the value 1 for "Private room" and
# the value 2 for "Shared room".

# In[29]:


def convertor(x):
    if x== 'Entire home/apt':
        return 0
    elif x == 'Private room':
        return 1
    else:
        return 2


# In[30]:


df.room_type.apply(convertor)


# # Use Z-score to standardize/normalize price. Show your data after adding the Price Z
# score. Print out price with its Z-Score values (just print a sample not all values).

# In[31]:


mean=df.price.mean()
std=df.price.std()
def z_score(x):
    return (x -mean)/std
df['z_score']=df.price.apply(z_score)
    


# In[32]:


df['z_score'].head()


# # You decided to drop last_review variable because we are not going to use it.

# In[33]:


df.drop(columns='last_review',axis=1,inplace=True)


#  # Print out Correlation matrix in terms of heatmap between all variables.

# In[34]:


df.isnull().sum()


# In[ ]:





# In[35]:


from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()
temp_data=encoder.fit_transform(df)
temp_data=pd.DataFrame(temp_data,columns=df.columns)
plt.figure(figsize=(10,6))
sns.heatmap(temp_data.corr(),annot=True)


#  # Bouns

# ## Using the computed Z-score values for the price, write a code to guarantee that
# outliers in z-scores will be removed.

# In[36]:


plt.scatter(x=df.z_score,y=df.index )


# In[37]:


new_data_set=df[(df.z_score <3) & (df.z_score >=-3)]


# In[38]:


print('old size = ',df.shape[0],'\nnew size = ', new_data_set.shape[0])


# In[39]:


plt.figure(figsize=(12,6))
plt.scatter(x=df.z_score,y=df.index,color='red')
plt.scatter(x=new_data_set.z_score,y=new_data_set.index,color='green')
plt.legend(labels=['Outlaier' ,'Normal'] )


# In[ ]:





# # Determine the price range in both Manhattan and Bronx (neighbourhood attribute). To
# Hint: do this, you have to plot price Distribution in every neighbourhood_group based
# on the neighbourhood (s) (in other words, plot neighbourhood (s) and price for each
# neighbourhood_group )

# In[40]:


plt.figure(figsize=(10,6))
data_manha=(df[(df.neighbourhood_group == 'Manhattan')]) 
data_Bronx=(df[(df.neighbourhood_group == 'Bronx')]) 


plt.scatter(x=data_manha.price,y=data_manha.index,color='red')

plt.show()


# In[41]:


plt.scatter(x=data_Bronx.price,y=data_Bronx.index,color='green')


# In[ ]:





# In[ ]:





# In[42]:


temp_df =df[['neighbourhood_group','price']]


# In[43]:


grouped_df=temp_df.groupby('neighbourhood_group')


# In[44]:


plt.figure(figsize=(14,4))
sns.distplot(grouped_df.get_group('Bronx').price )


# In[ ]:





# In[45]:


plt.figure(figsize=(16,3))

sns.distplot(grouped_df.get_group('Manhattan').price )


# In[ ]:




