# Author: Nicole Hardy
# Last Edited: 30 November 2023
# This program will classify text using information gain

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#1 = ChatGPT, 0 = Dickens
col_names = ['Token', 'Tag', 'label']
data = pd.read_csv("combinedPartsOfSpeech.csv", header= None, names= col_names)

feature_cols = ['Token', 'Tag']
x = data[feature_cols]
y = data.label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

dTree = DecisionTreeClassifier()

dTree = dTree.fit(x_train, y_train)
prediction = dTree.predict(x_test)
