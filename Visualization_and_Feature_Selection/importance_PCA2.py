
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
import numpy as np

def visualize(X_train, X_test, y_train, y_test, cols): 

    pd_X_train = pd.DataFrame(X_train, columns = cols)
    pd_X_test = pd.DataFrame(X_test, columns = cols)
    pd_y_train = pd.DataFrame(y_train, columns = ['Target'])
    pd_all_train = pd_X_train.join(y_train)
    X_train_0 = pd_all_train.loc[pd_all_train.Target == 0]['Weeks']
    X_train_1 = pd_all_train.loc[pd_all_train.Target == 1]['Weeks']

    plt.plot(X_train_0, X_train_1, 'o', label='')

def main(): 
    data = pd.read_csv("./Datasets/Spotify/Rising.csv")
    data = shuffle(data, random_state = 44) 
    all_X = data[['Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']]
    # all_X = data[['Followers']]
    print(all_X.columns)
    all_Y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=.3, random_state = 44)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #visualize(X_train, X_test, y_train, y_test, cols)

    # pca = PCA(n_components = 3) 
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)

    # explained_variance = pca.explained_variance_ratio_
    # print(explained_variance)
    classifier = LogisticRegression(penalty = 'l1')
    classifier.fit(X_train, y_train)
    print(classifier.coef_)
    
    y_pred = classifier.predict(X_test)
    
    print('Accuracy' , accuracy_score(y_test, y_pred))
    print("report", classification_report(y_test, y_pred))

main() 