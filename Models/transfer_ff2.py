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
import os 
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score, auc 
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#*******************LOAD DATA *******************
csv = pd.read_csv("./Datasets/Spotify/Rising.csv")
csv = shuffle(csv, random_state = 2)
pd_data = csv[['Year','Month','Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']]
pd_labels = csv['Target']

X_train, X_test, y_train, y_test = train_test_split(pd_data, pd_labels, test_size=.2, random_state = 2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .2, random_state = 2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)

#Convert to Numpy
y_train = y_train.to_numpy().ravel() 
y_test = y_test.to_numpy().ravel() 
y_val = y_val.to_numpy().ravel() 
# X_train = X_train.to_numpy()
# X_test = X_test.to_numpy() 

 #convert to pytorch 
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
X_val = torch.from_numpy(X_val)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)
y_val = torch.from_numpy(y_val).type(torch.LongTensor)

 #Load tensor dataset 
train = torch.utils.data.TensorDataset(X_train, y_train)
val = torch.utils.data.TensorDataset(X_val, y_val)
test = torch.utils.data.TensorDataset(X_test, y_test)

 #Loaders 
train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

#********************CREATE NET**************
class Net(nn.Module): 
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net,self).__init__()
        
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Output layer, 10 units - one for each digit
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout() 
        x = self.fc2(x)

        return x
#Initialize 
net = Net(len(pd_data.columns),6, 2).float()

#Load parameters from previously trained NN 
PATH = './ff_1.pth'
net.load_state_dict(torch.load(PATH))

#Function returns which label is actually gonna be assigned
all_categories = [0, 1]
def categoryFromOutput(output): 
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item() 
    return all_categories[category_i]


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
y_score, y = [], [] 
v_y_score, v_y = [], []
t_auc, t_precision, t_acc = [], [], []  
v_acc, v_precision, v_auc = [], [], []
#**************TRAINING*****************
for epoch in range(7):
    running_loss = 0
    total = 0 
    correct = 0 
    for i, data in enumerate(train_loader): 
        vals,labels = data 

        #Forward + Backward + Optimize 
        optimizer.zero_grad() 
        outputs = net(vals.float())
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step()
        running_loss += loss.item() 

        #Compute Train Accuracy 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        y_score.append(categoryFromOutput(outputs.cpu()))
        y.append(labels.cpu().numpy())
        
        #VALIDATION LOOP  
    for data in val_loader: 
        vals,labels = data
        outputs = net(vals.float())
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        v_y_score.append(categoryFromOutput(outputs.cpu()))
        v_y.append(labels.cpu().numpy())

    fpr, tpr, threshold = roc_curve(v_y, v_y_score)
    roc_auc = auc(fpr, tpr)
    v_precision.append(average_precision_score(v_y, v_y_score))
    v_auc.append(roc_auc_score(v_y, v_y_score))
    v_acc.append(accuracy_score(v_y, v_y_score))

    print("***********EPOCH", epoch , "************")
    #print ("average_precision_score v: " + str(average_precision_score(v_y, v_y_score)))
    print ("roc_auc_score v: " + str(roc_auc_score(v_y, v_y_score)))
    print ("accuracy v: " + str(accuracy_score(v_y, v_y_score)) ) 
        

    fpr, tpr, threshold = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    t_precision.append(average_precision_score(y, y_score))
    t_auc.append(roc_auc_score(y, y_score))
    t_acc.append(accuracy_score(y, y_score))
    
    #print ("average_precision_score t: " + str(average_precision_score(y, y_score)))
    print ("roc_auc_score t: " + str(roc_auc_score(y, y_score)))
    print ("accuracy t: " + str(accuracy_score(y, y_score)) ) 


plt.plot(t_acc, label="Training Accuracy",  color = 'green')
plt.plot(v_acc, label = "Validation Accuracy", color='red')
# plt.plot(t_auc, label="Training AUC",  color = 'blue')
# plt.plot(v_auc, label = "Validation AUC", color='orange')
plt.xlabel("Number of Steps")
plt.legend(loc='lower left')
plt.show()

correct = 0 
total = 0 
test_y_score, test_y = [], [] 
 
with torch.no_grad():
    for data in test_loader: 
        vals,labels = data
        outputs = net(vals.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_y_score.append(categoryFromOutput(outputs.cpu()))
        test_y.append(labels.cpu().numpy())

print ("roc_auc_score v: " + str(roc_auc_score(test_y, test_y_score)))
print ("accuracy v: " + str(accuracy_score(test_y, test_y_score)) ) 

# print('Accuracy : %d %%' % (
#     100 * correct / total))




# classes = (0, 1)
# class_correct = list(0. for i in range(2))
# class_total = list(0. for i in range(2))
# with torch.no_grad(): 
#     for data in test_loader: 
#         vals,labels = data
#         #print(labels)
#         outputs = net(vals.float())
#         _, predicted = torch.max(outputs.data, 1)
#         c = (predicted == labels).squeeze() 
#         for i in range(4): 
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1

# for i in range(2): 
#     print('Accuracy of %5s: %2d %%' %(classes[i], 100*class_correct[i]/class_total[i]))
