import numpy as np 
from sklearn import preprocessing



#input based array 

input_data = np.array([[5.1, -2.9 , 3.3], [-1.2, 7.8, -6.1], [3.9,0.4, 2.1], [7.3, -9.9, -4.5]])

#Binarize data 

#giving a threshold of 2.2, every value above 2.2 becomes a 1, the rest is 0 

data_binarized = preprocessing.Binarizer(threshold = 2.2).transform(input_data)


print("\n Binarized Data: \n" , data_binarized )


#add th Mean

print("\nBefore: ...")

print("Mean: ", input_data.mean(axis=0))

print("STD Deviation = " , input_data.std(axis=0))

                                                                                                                                                                              
#remove Mean

#you remove the mean so that each feature is centered around 0 for an even playing field

print("\nAfter")

data_scaled = preprocessing.scale(input_data)


print("Mean: " , data_scaled.mean(axis=0))

print("STD DEV: " , data_scaled.std(axis = 0))

#Scaling

#Min-max scaling 

data_scale_min = preprocessing.MinMaxScaler(feature_range = ( 0,1))

data_scaled_minMax = data_scale_min.fit_transform(input_data)

print("\nMin Max Scaled Data: \n" , data_scaled_minMax)


#each row is scaled so that the max value is 1 and other values are relative to its value

#Normalization

#modifies the values in vector so that you can measure on same scale
# two types l1 = least absolute deviations; absolute values are 1 nad l2 least squares, sum of squares is 1

data_normalize = preprocessing.normalize(input_data, norm='l1')
data_normalize2 = preprocessing.normalize(input_data, norm='l2')

print("Normalized PT1\n: ", data_normalize)
print("Normalized PT2\n: ", data_normalize2)
