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
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score, auc, f1_score, recall_score
from sklearn.model_selection import validation_curve, StratifiedKFold
import sys 
import statistics
import random  


os.environ['KMP_DUPLICATE_LIB_OK']='True'


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

#Used for Scoring Function below: converts tensor to single target scalar 
all_categories = [0, 1]
def categoryFromOutput(output): 
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item() 
    return all_categories[category_i]


#*******************LOAD DATA *******************
csv = pd.read_csv("./Datasets/Spotify/Rising.csv")

balance = True #Used for balancing dataset
transfer = True #Used for fine-tuning NN
train = False #Used for training NN on Task 2 dataset 
kfold = True #Used to implement k-fold validation on dataset 

if balance: #Used for balancing dataset
    class_1 = csv.loc[csv['Target'] == 1]
    class_0 = csv.loc[csv['Target'] == 0]
    class_0 = shuffle(class_0, random_state = 9)
    balanced = pd.concat([class_1, class_0[0:len(class_1)]])
    pd_data = balanced[['Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']]
    pd_labels = balanced[['Target']]
else: #Use original unbalanced dataset 
    pd_data = csv[['Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']]
    pd_labels = csv[['Target']]

#Initialize Neural Net architecture 
net = Net(len(pd_data.columns),6, 2).float() 

if transfer: #Loads pretrained NN and initializes param 
    PATH = './ff_1.pth'
    net.load_state_dict(torch.load(PATH))
    epochs = 10 
if train: epochs = 100 
if not train and not transfer: epochs = 0 


if kfold:
    #Convert dataframes to numpy arrays
    pd_data = pd_data.to_numpy()
    pd_labels = pd_labels.to_numpy().ravel()

    #Initialize overall seed values 
    seed_train_auc, seed_val_auc, seed_test_auc = [] , [], [] 
    seed_train_acc, seed_val_acc, seed_test_acc = [] , [], []  
    seed_val_auc, seed_val_auc, seed_val_auc = [] , [], [] 
    seed_val_acc, seed_val_acc, seed_val_acc = [] , [], [] 
    seed_test_auc, seed_test_auc, seed_test_auc = [] , [], [] 
    seed_test_acc, seed_test_acc, seed_test_acc = [] , [], []
    for j in range(101): 
        final_train_auc, final_val_auc, final_test_auc = [] , [], [] 
        final_train_acc, final_val_acc, final_test_acc = [] , [], []   

        #Create K-folds
        seed = random.randint(1, 200) #Generate random seed 
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
        
        for train_index, test_index in skf.split(pd_data, pd_labels): 
            # net = Net(11,6, 2).float()
            #Initialize Splits 
            X_train, X_test = pd_data[train_index], pd_data[test_index]
            y_train, y_test = pd_labels[train_index], pd_labels[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .2, random_state = 3, stratify=y_train)
            #Scale X values 
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_val = sc.transform(X_val)
            X_test = sc.transform(X_test)
            #convert to pytorch 
            X_train = torch.from_numpy(X_train)
            X_test = torch.from_numpy(X_test)
            X_val = torch.from_numpy(X_val)
            y_train = torch.from_numpy(y_train).type(torch.LongTensor)
            y_test = torch.from_numpy(y_test).type(torch.LongTensor)
            y_val = torch.from_numpy(y_val).type(torch.LongTensor)
            #Create tensor datasets
            train = torch.utils.data.TensorDataset(X_train, y_train)
            val = torch.utils.data.TensorDataset(X_val, y_val)
            test = torch.utils.data.TensorDataset(X_test, y_test)

            #Loaders 
            train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            y_score, y = [], [] 
            v_y_score, v_y = [], []
            t_auc, t_acc = [], []  
            v_acc, v_auc = [], []
            #**************TRAINING*****************
            for epoch in range(epochs):
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
                

                print('\n',"***********EPOCH", epoch , "************")
                print ("training accuracy: " + str(accuracy_score(y, y_score)) )
                print ("validation accuracy: " + str(accuracy_score(v_y, v_y_score)) )

                #Save training and validation accuracy for each epoch 
                t_auc.append(roc_auc_score(y, y_score))
                t_acc.append(accuracy_score(y, y_score))

                v_auc.append(roc_auc_score(v_y, v_y_score))
                v_acc.append(accuracy_score(v_y, v_y_score))

            #Save values for all epochs 
            final_train_auc.append(roc_auc_score(y, y_score))
            final_train_acc.append(accuracy_score(y, y_score))

            final_val_auc.append( roc_auc_score(v_y, v_y_score)) 
            final_val_acc.append( accuracy_score(y, y_score)) 
            #**************TESTING*****************
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

            
            #Save overall testing values 
            final_test_auc.append( roc_auc_score(test_y, test_y_score)) 
            final_test_acc.append( accuracy_score(test_y, test_y_score))
            
            print('\n'+ "TESTING INFORMATION" + '\n')
            print ("accuracy: " + str(accuracy_score(test_y, test_y_score)) ) 
            print ("roc_auc_score: " + str(roc_auc_score(test_y, test_y_score)))
            print ("average_precision_score: " + str(average_precision_score(test_y, test_y_score)))
            print('recall:' + str(recall_score(test_y, test_y_score)))
            print ('f1:' + str(f1_score(test_y, test_y_score))) 

            
        
        print("FINAL_TRAIN_AUC", statistics.mean(final_train_auc))
        print( "FINAL_TRAIN_ACC", statistics.mean(final_train_acc))
       
        print("FINAL_VAL_AUC", statistics.mean(final_val_auc))
        print("FINAL_VAL_ACC", statistics.mean(final_val_acc))

        print( "FINAL_TEST_AUC", statistics.mean(final_test_auc))
        print( "FINAL_TEST_ACC", statistics.mean(final_test_acc))

        #Save average for k-fold
        seed_train_acc.append(statistics.mean(final_train_acc))
        seed_train_auc.append(statistics.mean(final_train_auc))
        seed_val_acc.append(statistics.mean(final_val_acc))
        seed_val_auc.append(statistics.mean(final_val_auc))
        seed_test_acc.append(statistics.mean(final_test_acc))
        seed_test_auc.append(statistics.mean(final_test_auc))

    #Average for seeds
    print( "SEED_TRAIN_AUC", statistics.mean(seed_train_auc), statistics.stdev(seed_train_auc))
    print("SEED_VAL_AUC", statistics.mean(seed_val_auc), statistics.stdev(seed_val_auc))
    print( "SEED_TEST_AUC", statistics.mean(seed_test_auc), statistics.stdev(seed_test_auc))
    print( "SEED_TRAIN_ACC", statistics.mean(seed_train_acc), statistics.stdev(seed_train_acc))
    print( "SEED_VAL_ACC", statistics.mean(seed_val_acc), statistics.stdev(seed_val_acc))
    print( "SEED_TEST_ACC", statistics.mean(seed_test_acc), statistics.stdev(seed_test_acc))

    sys.exit() 

X_train, X_test, y_train, y_test = train_test_split(pd_data, pd_labels, test_size=.2, random_state = 4, stratify=pd_labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .2, random_state = 4, stratify=y_train)

#print(len(X_train), len(y_train), len(X_test), len(y_test))
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


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
y_score, y = [], [] 
v_y_score, v_y = [], []
t_auc, t_precision, t_acc = [], [], []  
v_acc, v_precision, v_auc = [], [], []
#**************TRAINING*****************
net = Net(len(pd_data.columns),6, 2).float()
for epoch in range(epochs):
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

    # fpr, tpr, threshold = roc_curve(v_y, v_y_score)
    # roc_auc = auc(fpr, tpr)
    v_precision.append(average_precision_score(v_y, v_y_score))
    v_auc.append(roc_auc_score(v_y, v_y_score))
    v_acc.append(accuracy_score(v_y, v_y_score))

    print("***********EPOCH", epoch , "************")
    #print ("average_precision_score v: " + str(average_precision_score(v_y, v_y_score)))
    print ("roc_auc_score v: " + str(roc_auc_score(v_y, v_y_score)))
    print ("accuracy v: " + str(accuracy_score(v_y, v_y_score)) ) 
        

    # fpr, tpr, threshold = roc_curve(y, y_score)
    # roc_auc = auc(fpr, tpr)
    #print(y, y_score)
    t_precision.append(average_precision_score(y, y_score))
    t_auc.append(roc_auc_score(y, y_score))
    t_acc.append(accuracy_score(y, y_score))
    
    #print ("average_precision_score t: " + str(average_precision_score(y, y_score)))
    print ("roc_auc_score t: " + str(roc_auc_score(y, y_score)))
    print ("accuracy t: " + str(accuracy_score(y, y_score)) ) 


# plt.plot(t_acc, label="Training Accuracy",  color = 'green')
# plt.plot(v_acc, label = "Validation Accuracy", color='red')
plt.plot(t_auc, label="Training AUC",  color = 'blue')
plt.plot(v_auc, label = "Validation AUC", color='orange')
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

print('\n'+ "TESTING INFORMATION" + '\n')
print ("accuracy: " + str(accuracy_score(test_y, test_y_score)) ) 
print ("roc_auc_score: " + str(roc_auc_score(test_y, test_y_score)))
print ("average_precision_score: " + str(average_precision_score(test_y, test_y_score)))
print('recall:' + str(recall_score(test_y, test_y_score)))
print ('f1:' + str(f1_score(test_y, test_y_score)))




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
