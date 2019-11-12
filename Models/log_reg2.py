import sklearn
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import csv

csv1 = pd.read_csv("./Datasets/Combo+Spotify.csv")
csv2 = pd.read_csv('./Datasets/complete_project_data.csv')
# print(csv.head())

data = csv1.iloc[1:, 2:-1]

target = csv1.iloc[1:, -1]
print(target)

clf = LogisticRegression().fit(data, target.values.ravel())
clf.predict(data)
clf.predict_proba(data)
score = clf.score(data, target)
print(score)