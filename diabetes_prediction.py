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
X
# Model Evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("The accuracy score of training data ", training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("The accuracy score of test data", test_data_accuracy)

# Making a predictive system
new_data = (1,89,66,23,94,28.1,0.167,21)
new_data_as_numpy_array = np.asarray(new_data)
new_data_reshaped = new_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(new_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("This person is a diabetic")