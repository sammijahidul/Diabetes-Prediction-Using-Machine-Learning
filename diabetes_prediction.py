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

# Separating the output column from dataset
X = data.drop('Outcome',axis = 1)
Y = data['Outcome']

# Need to Standardize the data 
scaler = StandardScaler()
scaler.fit(X)
standardize_data = scaler.transform(X)
print(standardize_data)
X = standardize_data
Y = data['Outcome']

# Splliting data into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, 
                                                    random_state=2)

print(X.shape, X_train.shape, X_test.shape)

## Training the model process start from here
# Training with support vector machine (svm)
classifier = svm.SVC(kernel='linear')

# traning the data with support vector machine classifier
classifier.fit(X_train, Y_train)

