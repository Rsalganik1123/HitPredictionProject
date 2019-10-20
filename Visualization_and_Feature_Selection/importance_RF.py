import csv
import sys 
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer, FunctionTransformer, StandardScaler

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def scaling(data): 
    scaler = MinMaxScaler()
    data[['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Followers']] = scaler.fit_transform(data[['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Followers']])
    return data

def main(): 
    importances = {} 
    # ********** Load Data ************
    data = pd.read_csv("../Datasets/Combo+Spotify+Followers.csv")
    all_X = data.iloc[:, 2:-1]
    all_Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=.2)
    # print(X_train)
    # ********* Ensemble Classifiers ************
    clf1 = ExtraTreesClassifier() 

    # ********** Preprocessing ****************
    # X_train = scaling(X_train)

    # ********** Feature Pipeline *************
    pipeline = Pipeline([
        # ('scaling', Pipeline([('minMax', MinMaxScaler(copy=False))])), #feature_range
        # ('scaling', Pipeline([('minMax', FunctionTransformer(scaling, validate=False))])), 
        # ('scaling2', Pipeline([('minMax', FunctionTransformer(scaling, validate=False))]))
        # ('normalization', Normalizer(copy=False)),
        # ('reduce_dim', None),
        ('classifier', clf1)])

    # ********** Grid Search *************
    print('Running Grid Search now')
    parameters_grid = {'classifier__criterion': ('gini', 'entropy')}

    # *********** Validation Pipeline ************
    grid_search = GridSearchCV(pipeline, parameters_grid, cv=2, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train,y_train)

    cvres = grid_search.cv_results_
    if(sys.argv[1] == "v"): 
        for accuracy, params in zip(cvres['mean_test_score'],cvres['params']):
            print('Mean accuracy: ', accuracy,'  using: ',params)

    # ********** Accuracy Summaries *************
    if (sys.argv[1] == "s"): 
        print("training score: ", grid_search.best_score_ , '\n', "best parameters: ", grid_search.best_params_,'\n')
        best_model = grid_search.best_estimator_
        y_true, y_pred = y_test, best_model.predict(X_test)
        print(classification_report(y_true, y_pred))
    
    clf1.fit(X_train, y_train)
    for i in range(len(clf1.feature_importances_)): 
        f = clf1.feature_importances_[i]
        # print(f)
        col = X_train.columns[i]
        importances[col] = f
    
    print(sorted(importances.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))

    
main() 