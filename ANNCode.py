# Artificial Neural Network

# Installing Theano
# pip3 install theano

# Installing Tensorflow
# pip3 install tensorflow

# Installing Keras
# pip3 instal keras

# Part 1 - Data Preprocessing

import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

#Part 1: Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.head()
dataset.info()
dataset.describe()

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

print(X[:10,:], '\n')

print(y[:10])
# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Giving ordinal feature to our variables
# onehotencoder = OneHotEncoder(categorical_features = [1])
onehotencoder= ColumnTransformer([('encoder', OneHotEncoder(), [1])],     remainder='passthrough')
# X = np.array(onehotencoder.fit_transform(X), dtype=np.float)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]
# print('test')
print(X[:10,:], '\n')
print(y[:10])

X.shape


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling (very important)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_mean=False)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''X_train_sc = sc.fit(X_train)
X_train =X_train_sc.fit_transform(X_train)
X_test = X_train_sc.transform(X_test)'''

#Import Keras library and packages

import keras
from keras.models import Sequential #to initialize NN
from keras.layers import Dense #used to create layers in NN
from keras.layers import Dropout


#Initialising the ANN - Defining as a sequence of layers or a Graph

classifier = Sequential()

#Adding the input layer
#units - number of nodes to add to the hidden layer.
#Tip: Average of nodes in the input layer and the number of nodes in the output layer. 11+2/2 = 6
#kernel_initializer - randomly initialize the weight with small numbers close to zero, according to uniform distribution.
#activation - Activation function.
#input_dim - number of nodes in the input layer, that our hidden layer should be expecting
classifier.add(Dense(units = 6, kernel_initializer = 'uniform',  activation = 'relu', input_dim = 11 ))


#Adding Second hidden layer
# There is no need to specify the input dimensions since our network already knows.
classifier.add(Dense(units = 6, kernel_initializer = 'uniform',  activation = 'relu'))


#Adding Output layer
# There is no need to specify the input dimensions since our network already knows.
#Units - one node in the output layer
#activation - If there are more than two categories in the output we would use the softmax
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#Stochastic Gradient Descent
#Compiling the ANN
#optimizer - algorithm to use to find the best weights that will make our system powerful
#loss - Loss function within our optimizer algorithm
#metric - criteria to evaluate the model

classifier.compile(optimizer = 'adam',loss= "binary_crossentropy",metrics=["accuracy"])

classifier.summary()

#Fitting the ANN to the Training Set
#batch size : number of observations after which we update the weights
#nb_epoch : How many times you train your model
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


#Predicting the Test set results
y_pred  = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


'''Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000'''

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

print(new_prediction)









