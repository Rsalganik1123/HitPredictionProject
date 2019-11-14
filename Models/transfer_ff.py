import csv 
import torch 
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.tensor as tnsr
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import torch.utils.data as data_utils


csv = pd.read_csv("./Datasets/Spotify/B+F+P.csv")
 
pd_data, pd_labels = csv.iloc[:, 2:-1], csv.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(pd_data, pd_labels, test_size=.2)

#Convert to Numpy
y_train = y_train.to_numpy().ravel() 
y_test = y_test.to_numpy().ravel() 
X_train = X_train.to_numpy()
X_test = X_test.to_numpy() 

# #convert to pytorch 
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

# #Load tensor dataset 
train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

# #Loaders 
train_loader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=4, shuffle=True)


PATH = './ff_1.pth'
net.load_state_dict(torch.load(PATH))

correct = 0 
total = 0 
with torch.no_grad():
    for data in test_loader: 
        vals,labels = data
        outputs = net(vals.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy : %d %%' % (
    100 * correct / total))