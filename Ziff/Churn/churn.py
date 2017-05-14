# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,auc,roc_auc_score

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold

# Importing the Dataset
data = pd.read_csv('churn.csv')

#Removing 'Unnamed: 0' and 'Phone' columns
# Area code doesn't appear to be useful
data.drop('Unnamed: 0',axis = 1, inplace = True)
data.drop('Phone',axis = 1, inplace = True)
data.drop('Area Code', axis = 1, inplace = True)
data['VMail Plan'].replace('?','no',inplace=True)
data['Int\'l Plan'].replace('?','no',inplace=True)

le = LabelEncoder()
data['VMail Plan'] = le.fit_transform(data['VMail Plan'])
data['Int\'l Plan'] = le.fit_transform(data['Int\'l Plan'])

# Checking the Question Mark and replacing with NaN (float value)
data.replace('?', 0, inplace=True)

# There are some large values which could be concerning
col_names = ['VMail Message', 'Account Length', 'Day Mins', 'Day Calls', 'Day Charge', 'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge']
data[col_names] = data[col_names].astype(float)
for col in col_names:
    data[col] = (data[col] - data[col].mean()) / data[col].std()

# Creating train and test data sets
data['Churn?'] = data['Churn?'].apply(lambda x: x.split('.')[0])
y = le.fit_transform(data['Churn?'])
data.drop('Churn?', axis=1, inplace=True)

X = data.drop('State', axis=1)
X = pd.DataFrame(X).as_matrix()

#10 kfold Cross Validation
ypred = np.zeros_like(y,dtype=float)
ypredc = np.zeros_like(y,dtype=float)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)    
for train, test in (kfold.split(X, y)):
    # Initial tests appear to indicate no overfitting, dropout layer unneccesary
    ann = Sequential()    
    ann.add(Dense(500, activation='tanh', kernel_initializer='random_normal', input_shape=(X[train].shape[1],)))
    ann.add(Dropout(rate=0.4))
    ann.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ann.fit(X[train], y[train], batch_size=100, epochs=150, verbose=0)
    fpr, tpr, _ = roc_curve(y[test],ann.predict_proba(X[test])) # If I add [:,1] error for index out of bounds
    ypred[test] = ann.predict_proba(X[test])
 
auc_score = roc_auc_score(y,ypred)
print('AUC Score: {:.3f}%'.format(auc_score))
