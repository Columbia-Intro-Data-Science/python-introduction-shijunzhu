import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# Load in data from the csv file
data = pd.read_csv('data/Cleveland.data.csv')
# We just removes all the instances which have missing values.
data = data[data['ca'] != '?']
data = data[data['thal'] != '?']

data['ca'] = data['ca'].astype(int)
data['thal'] = data['thal'].astype(int)

# Convert feature values from strings to integers.
from sklearn.preprocessing import LabelEncoder
labelencoders = []
for col in data.columns:

    labelencoder=LabelEncoder()
    data[col] = labelencoder.fit_transform(data[col])
    labelencoders.append(labelencoder)

# Set up the X and y variables

X = data.iloc[:,0:13]
y = data.iloc[:,13]

LogReg = LogisticRegression()

# Train the model using the training set
LogReg.fit(X, y)
yy = LogReg.predict(X)


# Save the trained decision tree into a pickle file
with open('pickledata/logres.pickle','wb') as f:

    pickle.dump((LogReg,labelencoders),f)

# pickle_in = open('pickledata/logres.pickle','rb')
# clf = pickle.load(pickle_in)