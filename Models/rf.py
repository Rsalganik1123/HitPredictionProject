import csv
import sys 
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, average_precision_score, accuracy_score, auc, f1_score, recall_score, precision_score
from sklearn.preprocessing import Normalizer, FunctionTransformer, StandardScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA 

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def main():
    # ********** Load Data ************
    data = pd.read_csv("./Datasets/Spotify/B+F+P.csv")
    # data = shuffle(data, random_state=34)
    musical = True 
    if musical: 
        all_X = data[['Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']]
    else:
        all_X = data[['Followers', 'Popularity']]
    all_Y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=.2, shuffle=True, random_state = 34, stratify=all_Y)

    
    # ********* Ensemble Classifiers ************
    clf1 = ExtraTreesClassifier(random_state=34) 

    # ********** Preprocessing ****************
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    pca = PCA()

    # ********** Feature Pipeline *************
    pipeline = Pipeline([
        # ('scaler', sc),
        # ('pca', pca),
        ('classifier', clf1)])

    # ********** Grid Search *************
    print('Running Grid Search now')
    parameters_grid = {}
    parameters_grid['classifier__criterion']= ('gini', 'entropy')
    parameters_grid ['classifier__n_estimators'] = (10, 20, 100, 200, 300)
    # parameters_grid ['pca__n_components']= (2,3,5,6)

    # *********** Validation Pipeline ************
    grid_search = GridSearchCV(pipeline, parameters_grid, cv=2, n_jobs=-1, scoring='precision')
    grid_search.fit(X_train,y_train)

    cvres = grid_search.cv_results_
    grid_details = False 

    if grid_details: 
        for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
            print('Mean accuracy: ', accuracy,'  using: ',params)

    # ********** Accuracy Summaries *************
    
    print("training score: ", grid_search.scoring, grid_search.best_score_ , '\n', "best parameters: ", grid_search.best_params_,'\n')
    best_model = grid_search.best_estimator_
    y_true, y_pred = y_test, best_model.predict(X_test)
    print('test score: ', classification_report(y_true, y_pred))
    print('Precision', precision_score(y_true, y_pred))
    print('Recall', recall_score(y_true, y_pred))
    print('F1', f1_score(y_true, y_pred))
    print('Accuracy' , accuracy_score(y_true, y_pred))
    print('AUC:',  roc_auc_score(y_true, y_pred))
    # clf1.fit(X_train, y_train)
    # for i in range(len(clf1.feature_importances_)): 
    #     f = clf1.feature_importances_[i]
    #     # print(f)
    #     col = X_train.columns[i]
    #     importances[col] = f
    
    # print(sorted(importances.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))

    
main() 