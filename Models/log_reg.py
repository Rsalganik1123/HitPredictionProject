
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import sys

def main(): 
    # *************** Load Data *****************
    data = pd.read_csv("./Datasets/Spotify/Rising.csv")
    data = shuffle(data, random_state = 44) 
    all_X = data[['Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']]
    #all_X = data[['Followers']]
    
    all_Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=.3, random_state = 44)
    
    # ************** Preprocessing ************** 
    sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    pca = PCA() 
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # explained_variance = pca.explained_variance_ratio_
    # print(explained_variance)

    #***************** Classifier****************
    classifier = LogisticRegression()

    # #***************** Pipeline ***************** 
    # pipeline = Pipeline([
    #     ('scaler', sc), 
    #     ('pca', pca),
    #     ('classifier', classifier)
    # ])

    # #**************** Grid Search ***************
    # print("Running grid search")
    # parameters_grid = {}
    # parameters_grid['classifier__penalty'] =  ('l2', 'l1')
    # parameters_grid['pca__n_components'] = (2,3,4,5)
    

    # #*************** Validation Pipeline ***********
    # grid_search = GridSearchCV(pipeline, parameters_grid, cv=5, n_jobs = 1, scoring='accuracy')
    # grid_search.fit(X_train, y_train)

    # cvres = grid_search.cv_results_
    
   
    # for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
    #     print('Mean accuracy: ', accuracy,'  using: ',params)

    # # ********** Accuracy Summaries *************
    
    # print("training score: ", grid_search.best_score_ , '\n', "best parameters: ", grid_search.best_params_,'\n')
    # best_model = grid_search.best_estimator_
    # y_true, y_pred = y_test, best_model.predict(X_test)
    # print(classification_report(y_true, y_pred))


    # classifier.fit(X_train, y_train)
    # print(classifier.coef_)
    
    # y_pred = classifier.predict(X_test)
    
    # print('Accuracy' , accuracy_score(y_test, y_pred))

main() 