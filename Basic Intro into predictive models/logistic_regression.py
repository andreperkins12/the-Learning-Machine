#Logistic Regression

import numpy as np 

from sklearn import linear_model

import matplotlib.pyplot as plt 


from utilities import visualize_classifier

#sample data

X = np.array([[3.1,7.2] , [4,6.7], [2.9, 8], [5.1, 4.5] , [6.3,4], [1,4] , [2.8,1], [3.3,2]])

y = np.array([0,0,1,1,2,2,3,3])

#train classifier, logisitic classifier object

classifier = linear_model.LogisticRegression(solver = 'liblenear', C=1)


#train classifier

classifier.fit(X,y)

#visualize performance

visualize_classifier(classifier, X,y)





