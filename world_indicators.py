#!/usr/bin/env python
# coding: utf-8

# # **Installing Necessary Dependencies**

# In[ ]:


get_ipython().system('pip install -q pycountry')


# # **Importing Necessary Packages**

# In[ ]:


import pycountry                                        #to analyze the countries data
import numpy as np                                      #package for handling arrays
import pandas as pd                                     #package for data analysis
import seaborn as sns                                   #package for data visualization
import matplotlib.pyplot as plt                         #package for data visualization and charts manipulations
from sklearn.preprocessing import MinMaxScaler          #Importing MinMaxScaker for data normalization
from sklearn.cluster import KMeans                      #Importing Kmean from sklearn
from scipy.optimize import curve_fit                    #Importing curve_fit from scipy
import statsmodels.api as sm                            #Package for implementaion of statistics
scaler=MinMaxScaler()                                   #Creating object of MinMaxScaler


# # **Reading the Data**

# In[ ]:


def load_data(file):
  """
  this function receives data file remove metadata and splits data into two
  chunks of year_data and country_data
  """

  df = pd.read_csv(file, skiprows=4)
  
  # Drop unnecessary columns
  df = df.iloc[:, :-1]
  
  year_data = df.copy()
  
  # Create a dataset with countries as columns
  country_data = df.set_index(["Country Name", "Indicator Name"])
  country_data.drop(["Country Code", "Indicator Code"], axis=1, inplace=True)
  
  # Transpose the countries dataframe
  country_data = country_data.T
  
  # Return the years and countries dataframes
  return year_data, country_data


# In[ ]:


data_frame, countries = load_data("climate_indicators.csv")


# # **Extracting Some Necessary Data of Indicators against Countries**

# In[ ]:


def extracted_data(data, indicators_list):
    # Filter the dataset for the required indicators
    extracted_data = data[data["Indicator Name"].isin(indicators_list)]
    # Extracting data for only countries we are interested in
    country_list = [country.name for country in list(pycountry.countries)]
    extracted_data = extracted_data[extracted_data["Country Name"].isin(country_list)]
    return extracted_data


# # **Indicators**

# In[ ]:



indicators_list = [
    'Access to electricity (% of population)',
    'Agricultural land (% of land area)',
    'Forest area (% of land area)',
    'CO2 emissions (metric tons per capita)',
    'Total greenhouse gas emissions (kt of CO2 equivalent)',
    'Renewable electricity output (% of total electricity output)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Electricity production from coal sources (% of total)',
    'Electricity production from oil sources (% of total)',   
]


# In[ ]:


data = extracted_data(data_frame, indicators_list)


# In[ ]:


data = data.fillna(method='ffill').fillna(method='bfill')
pivot_data = data.pivot_table(index='Country Name', columns='Indicator Name', values='2020')


# In[ ]:


plt.figure(figsize=(12,5))
plt.bar(x=pivot_data.sum().index,height=pivot_data.sum().values,color='purple')
plt.xticks(rotation=90)
plt.title('Sum of All the Selected Indicators for Year 2020')
plt.show()


# In[ ]:


co_index=pivot_data['CO2 emissions (metric tons per capita)'].sort_values(ascending=False)[0:10]
plt.figure(figsize=(15,5))
plt.plot(co_index.index,co_index.values)
plt.title('Top 10 CO2 Emitors')
plt.show()


# In[ ]:


electric_coal=pivot_data['Electricity production from coal sources (% of total)'].sort_values(ascending=False)[0:5]
plt.pie(electric_coal.values,labels=electric_coal.index,autopct='%.1f%%')
plt.title("Top 5 Countries Producing Electricity from Coal")
plt.show()


# In[ ]:


re_usage=pivot_data['Renewable energy consumption (% of total final energy consumption)'].sort_values(ascending=False)[0:5]
plt.bar(x=re_usage.index,height=re_usage.values)
plt.title("Top Renewable Energy Consumer Countries")
plt.show()


# # **Clustering**

# In[ ]:


# data Normalization
norm_data, min_value, max_value = scaler.fit_transform(pivot_data.values),np.min(pivot_data.values),np.max(pivot_data.values)

# number of clusters
cluster_count = 4

# Applying KMeans
kmeans = KMeans(n_clusters=cluster_count, random_state=0)
cluster_labels = kmeans.fit_predict(norm_data)


# In[ ]:


# Add the cluster labels to the dataset
pivot_data["Cluster"] = cluster_labels
pivot_data.groupby("Cluster").mean()
labels = indicators_list
