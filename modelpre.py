from fileinput import filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge 
from sklearn.preprocessing import LabelEncoder
import pickle



d_data=pd.read_csv('insurance.csv')

label_encode=LabelEncoder()
labels=label_encode.fit_transform(d_data.sex)
d_data['sex']=labels
#male-->1
#female-->0

d_data.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)

d_data.replace({'smoker':{'yes':0,'no':1}}, inplace=True)


X = d_data.drop(columns='charges', axis=1)
Y = d_data['charges']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=3)



treeRegressionModel = DecisionTreeRegressor(random_state=42, max_depth=6)
treeRegressionModel.fit(X_train.values, Y_train.values)
y_pred_dt = treeRegressionModel.predict(X_test.values)
treeRegressionModel.score(X_test.values, Y_test.values)

filename="model_pre.sav"
pickle.dump(treeRegressionModel,open(filename,'wb'))








