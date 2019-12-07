
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, average_precision_score, accuracy_score, auc, f1_score, recall_score, precision_score

import sys

def main(): 
    musical = False
    #context = False
    # *************** Load Data *****************
    data = pd.read_csv("./Datasets/Spotify/B+F+P.csv")

    if musical: 
        all_X = data[['Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']]
    else:
        all_X = data[['Followers', 'Popularity']]
    
    all_Y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=.2, shuffle = True, stratify= all_Y, random_state = 44)
    
    # ************** Preprocessing ************** 
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # pca = PCA() 
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # explained_variance = pca.explained_variance_ratio_
    # print(explained_variance)

    #***************** Classifier****************
    classifier = SVC()

    # #***************** Pipeline ***************** 
    pipeline = Pipeline([
        # ('scaler', sc), 
        # ('pca', pca),
        ('classifier', classifier)
    ])

    # #**************** Grid Search ***************
    # print("Running grid search")
    parameters_grid = {}
    parameters_grid['classifier__C'] =  (0.1, 0.5, 1.0, 2.0)
    parameters_grid['classifier__kernel'] = ('linear', 'poly', 'rbf', 'sigmoid') 
    # parameters_grid['pca__n_components'] = (2,3,4,5)
    

    # #*************** Validation Pipeline ***********
    grid_search = GridSearchCV(pipeline, parameters_grid, cv=5, n_jobs = 1, scoring='f1')
    grid_search.fit(X_train, y_train)

    cvres = grid_search.cv_results_
    

    full_details = False 
    if full_details: 
        for sensitivity, train_auc_score, val_auc_score, std_train_auc, std_val_auc, params in zip(cvres['mean_test_sensitivity'],cvres['mean_train_auc'],cvres['mean_test_auc'],cvres['std_train_auc'],cvres['std_test_auc'],cvres['params']):
            print('Sensitivity: ', sensitivity)

    # ********** Accuracy Summaries *************
    
    print("training score: ", grid_search.scoring, grid_search.best_score_ , '\n', "best parameters: ", grid_search.best_params_,'\n')
    best_model = grid_search.best_estimator_
    y_true, y_pred = y_test, best_model.predict(X_test)
    print(classification_report(y_true, y_pred))
    print('Precision', precision_score(y_true, y_pred))
    print('Recall', recall_score(y_true, y_pred))
    print('F1', f1_score(y_true, y_pred))
    print('Accuracy' , accuracy_score(y_true, y_pred))
    print('AUC:',  roc_auc_score(y_true, y_pred))

main() 