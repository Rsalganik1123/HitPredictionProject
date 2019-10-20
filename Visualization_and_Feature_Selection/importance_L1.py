import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV



def main(): 
    data = pd.read_csv("./Datasets/Combo+Spotify.csv")
    all_X = data.iloc[:, 2:-1]
    all_Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=.2)

main() 