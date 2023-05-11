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