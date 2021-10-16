# Importing neccessary libraries 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

## Data pre-processing part starts from here

# Importing dataset into pandas dataframe
data = pd.read_csv("diabetes.csv")
data.head()
data.describe()
data.shape
data['Outcome'].value_counts() # will show the output measurements 
data.groupby('Outcome').mean() # will show the mean value for each column [O & 1]



