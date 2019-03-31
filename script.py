# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#training the model
from sklearn.linear_model import LogisticRegression

#measuring model performance
from sklearn.metrics import classification_report

#spliting data into test and training dataset
from sklearn.model_selection import train_test_split

# load data from train.csv file
data = pd.read_csv("train.csv")

#extract columns
df = pd.DataFrame(data, columns = ['PassengerId', 'Embarked', 'Survived'])

#split the array to independent array X
X = df.iloc[:, 1:2]

#split the array to dependent vector Y
Y = df.iloc[:,2:3]

#Encode categorial columns used
labelEncoder = LabelEncoder()
X_Label_Encoded = X.apply(labelEncoder.fit_transform)

ct = ColumnTransformer([('oh_enc', OneHotEncoder(sparse=False), [0]),],  remainder='passthrough')
X_Encoded = ct.fit_transform(X_Label_Encoded)

X_train, X_test, y_train, y_test = train_test_split(X_Encoded, Y, test_size=0.30, random_state=0)

regressor = LogisticRegression(random_state=0).fit(X_train, y_train.values.ravel())

#prediction
predicted = regressor.predict(X_test)

report = classification_report(y_test, predicted)
print(report)






