from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import csv
import sys 
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer, FunctionTransformer, StandardScaler


def main(): 
    importances = {} 
    data = pd.read_csv("./Datasets/Combo+Spotify+Followers.csv")
    all_X = data.iloc[:, 2:-1]
    all_Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=.2)

    clf1 = LogisticRegression() 
    rfe = RFE(clf1)
    rfe = rfe.fit(X_train, y_train)
    print(rfe.ranking_)
    # for i in range(len(rfe.ranking_)):
    #     rank = rfe.ranking_[i]
    #     col = X_train.columns[i]
    #     importances[col] = rank
    # print(importances)

main()