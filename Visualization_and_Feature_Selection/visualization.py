
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np 
import csv

def compareDatasets(mine, theirs): 
    my_means = mine.mean(axis = 0)
    print('my_means', my_means)
    their_means = theirs.mean(axis = 0)
    print('their_means', their_means)

    my_std = mine.std(axis = 0)
    print("my_std", my_std)
    their_std = theirs.std(axis = 0)
    print("their_std", their_std)

def scatterPlot(csv, v1, v2, class_label): 
    sns.scatterplot(x=v1, y=v2, hue=class_label,data=csv)
    plt.show()

def histogram(csv, v): 
    csv.hist() 
    # sns.distplot(csv[v], bins=10, kde=False) 
    plt.show()

def corrMatrix(csv): 
    sns.heatmap(csv.corr(), annot=True)
    plt.show()

def barPlot(csv, val, class_label): 
    billboard = csv.loc[csv[class_label] == 1]
    not_billboard = csv.loc[csv[class_label] == 0]
    print("bb", len(billboard), "nb", len(not_billboard) )

    sns.distplot(billboard[[val]], hist=False, rug=True, color="r", label="Billboard")
    sns.distplot(not_billboard[[val]], hist=False, rug=True, color="b", label="Not_billboard")
    plt.show() 

def main(): 

    csv1 = pd.read_csv("../Datasets/Spotify/Followers.csv")
    billboard = csv1.loc[csv1['Target'] == 1]
    not_billboard = csv1.loc[csv1['Target'] == 0]
    # print(csv1.iloc[:, -1])
    print("bb", len(billboard), "nb", len(not_billboard))

    # csv2 = pd.read_csv('./Datasets/complete_project_data.csv')
    # histogram(csv1,'Tempo')
    # scatterPlot(csv1, 'Followers', 'Liveness', 'Target')
    corrMatrix(csv1)
    # compareDatasets(csv1.iloc[1:, :-1], csv2.iloc[1: , :-1])
    # barPlot(csv1, 'Energy', 'Target')

main() 