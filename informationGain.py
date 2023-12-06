# Author: Nicole Hardy
# Last Edited: 30 November 2023
# This program will classify text using information gain

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

#1 = ChatGPT, 0 = Dickens
col_names = ['Token', 'Tag', 'label']
col_types = {
     "Token": "string"
    ,"Tag": "string"
    ,"label": "float64"
}
# data = pd.read_csv("combinedPartsOfSpeech.csv", header= None, names= col_names)

data = pd.read_csv("combinedPartsOfSpeech.csv"
    ,dtype = col_types
    ,header = None
    ,names = col_names
)

def asciiSum(inputStr):
    val = 0
    for c in inputStr:
        val = val + ord(c)
    return float(val)

tokenAsciiList = []
for x in data["Token"]:
    tokenAsciiList.append(float(asciiSum(x)))

tagAsciiList = []
for x in data["Tag"]:
    tagAsciiList.append(float(asciiSum(x)))

data["Token_Ascii"] = tokenAsciiList
data["Tag_Ascii"] = tagAsciiList

feature_cols = ['Token_Ascii', 'Tag_Ascii']
x = data[feature_cols]
y = data.label


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
#print(x_train.max())
dTree = DecisionTreeClassifier()

dTree = dTree.fit(x_train, y_train)
prediction = dTree.predict(x_test)
#print(prediction)
print(metrics.accuracy_score(y_test, prediction))
