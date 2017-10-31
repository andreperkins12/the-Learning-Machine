#label encoding, transforms word labels into numeral form to understand the data


import numpy as np 
from sklearn import preprocessing

input_lable = ['red', 'black', 'blue', 'green'];

#create label encoder and train it

encoder = preprocessing.LabelEncoder()
encoder.fit(input_lable)

#print mapping

print("\nLabel: \n")

for i, item in enumerate(encoder.classes_):
	print(item, '---> ', i)


	#encode set of labels


	encoded_values = encoder.transform(input_lable)

	print("\nLabels: \n", input_lable)
	print("\nEncoded Values\n", list(encoded_values))



	#decode random set of numbers

	decoded_list = encoder.inverse_transform(encoded_values)

	print("Encoded Val :\n" , encoded_values)

	print("Decoded val: \n ", list(decoded_list))