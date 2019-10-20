import csv
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
import numpy as np 
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF


def pipeline(csv): 
    pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', 'passthrough'),
    ('classify', LinearSVC(dual=False, max_iter=10000))
    ])
    N_FEATURES_OPTIONS = [2, 4, 8]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7), NMF()],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
    ]
    reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

    grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid, iid=False)
    
    # grid.fit(csv.iloc[1:, 3:-1].to_numpy(), csv.iloc[1:, -1].to_numpy())
    digits = load_digits()
    grid.fit(digits.data, digits.target)

    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
                (len(reducer_labels) + 1) + .5)

    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

    plt.title("Comparing feature reduction techniques")
    plt.xlabel('Reduced number of features')
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel('Digit classification accuracy')
    plt.ylim((0, 1))
    plt.legend(loc='upper left')

    plt.show()

def builtIn(csv): 
    X = csv.iloc[:, 3:-1]
    y = csv.iloc[:, -1]
    # X_new_anova = SelectKBest(f_classif, k=4).fit_transform(X, y)
    X_new_chi = SelectKBest(chi2, k=4).fit_transform(X, y)
    # print(X_new_chi, X_new_anova)

def wrapper(csv, mode): #Backward Elimination, Forward Selection, Bidirectional Elimination and RFE
    X = csv.iloc[:, :-1]
    y = csv.iloc[:, -1]
    X_1 = sm.add_constant(X)
    if mode == 0: #BE
        model = sm.OLS(y,X_1).fit()
        pvalues = model.pvalues
        print(pvalues)
    
    

def embedded(): 
    return 0 

def filter(csv): 
    #Using correlation matrix
    plt.figure(figsize=(12,10)) #Create matrix
    cor = csv.corr()
    sns.heatmap(cor, annot=True)
    plt.show()
    cor_target = abs(cor["Target"])
    relevant_features = cor_target[cor_target>0.3] #get relevant features 
    print(relevant_features)


def main(): 
    csv1 = pd.read_csv("./Datasets/Combo+Spotify.csv")
    # print(csv1.columns)
    filter(csv1)
    # wrapper(csv1, 0)
    # builtIn(csv1)
    # pipeline(csv1) 

main()