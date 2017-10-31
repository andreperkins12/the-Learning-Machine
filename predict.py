#Naive Bayes is used to build classifiers to best describe probabiliy of ebvents occurring

import numpy as np 

import matplotlib.pyplot as plt 
from sklearn.Naive_bayes import GaussianNB
from sklearn import cross_validation
from utilities import visualize_classifier

#input file

thFile = 'data_multivar_nb.txt'

#load Data from file
data = np.loadtxt(thFile, delimiter = ',')

X, y = data[: , : -1] , data[:, -1]

# create Naive Bayes classifier

classifier = GaussianNB()

#train the classifier

classifier.fit(X, y)

#run the classifier
#predict the values of training the data

y_pred = classifier.predict(X)

#compute the accuracy

accuracy = 100.0 * ( y == y_pred).sum() / X.shape[0]

print("Accuracy of Naive Bayes classifier = ", round(accuracy, 2), "%" )



