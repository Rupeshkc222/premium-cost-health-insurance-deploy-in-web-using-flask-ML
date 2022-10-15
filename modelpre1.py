#2nd  model 

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






dataset=pd.read_csv('insurance 1.csv')

label_encode=LabelEncoder()
labels=label_encode.fit_transform(dataset.sex)
dataset['sex']=labels
#male-->1
#female-->0
dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)
dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

Xx = dataset.drop('charges',axis=1).values
yy = dataset['charges'].values

Xx_train, Xx_test, yy_train, yy_test = train_test_split(Xx, yy, test_size=0.25 , random_state = 3 )

ridge=Ridge()
ridge.fit(Xx_train,yy_train)

filename="model_pre1.sav"
pickle.dump(ridge,open(filename,'wb'))