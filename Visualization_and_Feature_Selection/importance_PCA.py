import csv
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np 
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
     
    # plt.show() 
    return f, ax, sc, txts

def PCA_implement(X, y, feat_cols): 
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y 
    df['label'] = df['y'].apply(lambda i: str(i))   
    rndperm = np.random.permutation(df.shape[0])   

    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(df[feat_cols].values)
    # pca_df = pd.DataFrame()
    for i in range(10): 
        # pca_df.insert(i, ('pca'+str(i)), pca_result[:, i])
        df['pca'+str(i)] = pca_result[:, i]
    
    sns.scatterplot(x='pca1', y='pca2', hue="y", data=df.loc[rndperm,:], legend="full", alpha=0.3)
    plt.show()
    

def TSNE_implement(X_train, y_train): 
    tsne = TSNE().fit_transform(X_train) 
    plot = fashion_scatter(tsne, y_train)
    plt.show(plot)

def main(): 

    data = pd.read_csv("./Datasets/Combo+Spotify.csv")
    all_X = data.iloc[:, 2:-1]
    all_Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size=.2)
    feat_cols = ['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    PCA_implement(all_X, all_Y, feat_cols)
    # TSNE_implement(X_train, y_train)

    
main() 