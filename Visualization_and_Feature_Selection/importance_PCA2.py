
import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main(): 
    data = pd.read_csv("../Datasets/Combo+Spotify+Followers.csv")
    all_X = data.iloc[:, 2:-1]
    all_Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=.2)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    pca = PCA(n_components = 7) 
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    classifier = LogisticRegression(penalty = 'l2')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print('Accuracy' , accuracy_score(y_test, y_pred))

main() 