#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import sys
import numpy as np # linear algebra
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import glob as glob
from sklearn import model_selection, preprocessing, ensemble
from scipy import stats
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import image
from PIL import Image
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v


# In[2]:

train_df = pd.read_json("train.json")
train_df.head(10)


# In[3]:


test_df = pd.read_json("test.json")
train_df.info()
print("# of train rows : ", train_df.shape[0])
print("# of test rows : ", train_df.shape[0])


# In[4]:


# Plot histograms for the following numeric columns: Price, Latitude & Longitude
# None of the histograms are very useful due to outliers

sns.distplot(train_df['price'], kde = False)


# In[5]:


sns.distplot(train_df['latitude'], kde = False)


# In[6]:


sns.distplot(train_df['longitude'], kde = False)

# In[7]:


# Plot hour-wise listing trend and find out the top 5 busiest hours of postings

train_df['created'] = pd.to_datetime(train_df['created'])

sns.countplot(train_df.created.dt.hour)


# In[8]:


# Visualization to show the proportion of target variable values

sns.countplot(train_df['interest_level'], order=['low','medium','high'])


# In[9]:


# !!!Find the number of missing values in each variable

# number of null values in each variable

print(train_df.isnull().sum())


# In[10]:


# Find out the number of outliers in each variable

z = np.abs(stats.zscore(train_df[['bedrooms', 'bathrooms', 'latitude', 'longitude', 'price']]))

threshold = 3

outliers = np.where(z>threshold)
outliersRow = outliers[0]
outliersCol = outliers[1]

bedrooms_outliers = 0 
bathrooms_outliers = 0
latitude_outliers = 0
longitude_outliers = 0 
price_outliers = 0

for val in outliersCol:
    if val == 0:
        bedrooms_outliers += 1
    elif val == 1:
        bathrooms_outliers += 1
    elif val == 2:
        latitude_outliers += 1
    elif val == 3:
        longitude_outliers += 1
    else:
        price_outliers += 1

print('TOTAL OUTLIERS')
print('bedrooms : ', bedrooms_outliers)
print('bathrooms : ', bathrooms_outliers)
print('latitude : ', latitude_outliers)
print('longitude : ', longitude_outliers)
print('price : ', price_outliers)


# In[11]:


# Plot visualizations to demonstrate outliers

sns.boxplot(x = train_df['bedrooms'])


# In[12]:


sns.boxplot(x = train_df['bathrooms'])


# In[13]:

sns.boxplot(x = train_df['latitude'])

# In[14]:


sns.boxplot(x = train_df['longitude'])


# In[15]:


sns.boxplot(x = train_df['price'])


# In[16]:


# Find outliers using Z-score
# look through latitude, longitude, price

z = np.abs(stats.zscore(train_df[['latitude', 'longitude', 'price']]))
threshold = 3

# Remove outliers from these columns with a z-score greater than 3
train_df_o = train_df[(z<threshold).all(axis=1)]

z = np.abs(stats.zscore(train_df_o[['bedrooms']]))
# Remove outliers from bedroom with very high z-score
train_df_o = train_df_o[(z<5).all(axis=1)]

print(train_df.shape)
print(train_df_o.shape)

train_df = train_df_o


# In[17]:


# Plot histograms for the following numeric columns:
# Price

sns.distplot(train_df['price'], kde = False)


# In[18]:


# Latitude

sns.distplot(train_df['latitude'], kde = False)


# In[19]:


#Longitide

sns.distplot(train_df['longitude'], kde = False)


# In[20]:


# !!!Can we safely drop the missing values? If not, how will you deal with them?


# In[21]:


# !!!Extract features from the images and transform it into data that’s ready to be used in the model for classification.

# Feature 1: Grayscale pixel values      ~~~ Commented out due to memory issues ~~~
#grayscale_feat_list = []
#for filename in glob.glob('images_sample/*/*.jpg'):
#    img = imread(filename, as_gray=True)
#    imgFeature = np.reshape(img, (img.shape[0]*img.shape[1]))    
#    grayscale_feat_list.append(imgFeature)


# Feature 2: Channel mean pixel values
mean_pix_list = []
for filename in glob.glob('images_sample/*/*.jpg'):
    colorImg = imread(filename)
    print(filename)
    feat_matrix = np.zeros((colorImg.shape[0], colorImg.shape[1]))
    for i in range(0, colorImg.shape[0]):
        for j in range(0, colorImg.shape[1]):
                feat_matrix[i][j] = ((int(colorImg[i,j,0]) + int(colorImg[i,j,1]) + int(colorImg[i,j,2]))/3)
    feature = np.reshape(feat_matrix, (colorImg.shape[0]*colorImg.shape[1]))
    mean_pix_list.append(feature)

# In[22]:


# Extract features from the text data and transform it into data that's ready to be used in the model for classification

# number of photos
train_df['num_photos'] = train_df['photos'].apply(len)
test_df['num_photos'] = test_df['photos'].apply(len)

# number of features
train_df['num_features'] = train_df['features'].apply(len)
test_df['num_features'] = test_df['features'].apply(len)

# get hour, day, month, year
train_df['hour'] = train_df.created.dt.hour
train_df['day'] = train_df.created.dt.day
train_df['month'] = train_df.created.dt.month
train_df['year'] = train_df.created.dt.year

test_df['created'] = pd.to_datetime(test_df['created'])
test_df['hour'] = test_df.created.dt.hour
test_df['day'] = test_df.created.dt.day
test_df['month'] = test_df.created.dt.month
test_df['year'] = test_df.created.dt.year

# use label encoder to normalize labels
categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
    le = preprocessing.LabelEncoder()
    le.fit(list(train_df[f].values) + list(test_df[f].values))
    train_df[f] = le.transform(list(train_df[f].values))
    test_df[f] = le.transform(list(test_df[f].values))

train_df.info()


# 

# In[ ]:




