
# coding: utf-8

# ## Import the required packages

# In[1]:

import pandas as pd
import numpy as np


# ## Load the data using pandas package

# In[2]:

dataset = pd.read_csv('LocationData.csv')
dataset


# In[3]:

#Let us now add another column
#Function to convert HMS to Seconds
def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


# In[4]:

dataset['track_time_sec'] = ""
for i in range(0,len(dataset['track_time'])):
    dataset['track_time_sec'][i] = get_sec(dataset['track_time'][i][11:].split(".")[0])


# In[5]:

#With 3 clusters 
X = np.array(dataset[['latitude','longitude']])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_
dataset['labels_latlo'] = labels


# In[7]:

import matplotlib.pyplot as plt
plt.scatter(dataset['latitude'],dataset['longitude'], label='skitscat', c=dataset['labels_latlo'].astype(int), s=25, marker="o")

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('K Means')
plt.legend()
plt.show()


# In[8]:

# Let us remove the noise based on coordinate clustering
dataset  = dataset[dataset['labels_latlo'] == 0]


# In[9]:

plt.scatter(dataset['latitude'],dataset['longitude'], label='datapoint', c=dataset['labels_latlo'].astype(int), s=25, marker="o")

plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('K Means')
plt.legend()
plt.show()


# In[10]:

#Libraries and constants for calculating distance
from numpy import sin, cos, sqrt, radians, arctan2
R = 6371.0

#Function to calculate distance
def distance(lat1, lon1, lat2, lon2):
    dlon = radians(lon2) - radians(lon1)
    dlat = radians(lat2) - radians(lat1)

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))

    return R * c * 1000



# In[11]:

dataset['Distance'] = distance(dataset['latitude'], dataset['longitude'], dataset['latitude'].shift(1), dataset['longitude'].shift(1))
dataset



# In[12]:

#Time difference
dataset['Time_diff'] = dataset['track_time_sec']-dataset['track_time_sec'].shift(1)

# Fill the nulls in the in the first 2 rows of distance_covered, time_diff
dataset.fillna(0)


# In[13]:

print(len(dataset))
dataset = dataset[(dataset['Distance']>0) & (dataset['Time_diff']>0)]
print(len(dataset))


# In[14]:

# To fit the data fr modeling, we create an array
X = np.array(dataset[['Distance','Time_diff']])


# In[15]:

#Using KMeans to cluster the points to remove noise
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels = kmeans.labels_
dataset['labels'] = labels


# In[16]:



plt.scatter(dataset['Distance'],dataset['Time_diff'], label='skitscat', c=labels, s=25, marker="o")

plt.xlabel('Distance')
plt.ylabel('Time')
plt.title('K Means')
plt.legend()
plt.show()


# In[17]:

dataset = dataset[dataset['Distance'] < 5000]
dataset = dataset[dataset['Distance'] < 1000] #
dataset = dataset[dataset['Time_diff'] < 200]


# In[18]:

plt.scatter(dataset['Distance'],dataset['Time_diff'], label='skitscat', c=dataset['labels'].astype(int), s=25, marker="o")

plt.xlabel('Distance')
plt.ylabel('Time')
plt.title('K Means')
plt.legend()
plt.show()


# In[19]:

#With 4 clusters
#The diff between now and earler is we removed noise from data effectively removing two clusters in the previous graph
# Now we are modeling after removing the noise observed earlier
X = np.array(dataset[['Distance','Time_diff']])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
labels = kmeans.labels_
dataset['labels'] = labels


# In[20]:

plt.scatter(dataset['Distance'],dataset['Time_diff'], label='skitscat', c=dataset['labels'].astype(int), s=25, marker="o")

plt.xlabel('Distance')
plt.ylabel('Time')
plt.title('K Means')
plt.legend()
plt.show()


# In[21]:

#With 3 clusters 
X = np.array(dataset[['Distance','Time_diff']])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_
dataset['labels'] = labels


# In[22]:

plt.scatter(dataset['Distance'],dataset['Time_diff'], label='skitscat', c=dataset['labels'].astype(int), s=25, marker="o")

plt.xlabel('Distance')
plt.ylabel('Time')
plt.title('K Means')
plt.legend()
plt.show()


# In[37]:

# Load the data into a file to visualise using tableau on a map

dataset.to_csv('CleanedDataWithLabels.csv')
dataset = dataset[dataset['labels'] == 0]
dataset.drop(['labels','labels_latlo'], axis =1).to_csv('CleanData.csv')




