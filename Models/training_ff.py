import comet_ml
from comet_ml import Experiment
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
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score, auc, f1_score, recall_score
import os 

os.environ['KMP_DUPLICATE_LIB_OK']='True'



hyper_params = {
    "sequence_length": 28,
    "input_size": 18,
    "hidden_size": 6,
    "num_layers": 2,
    "num_classes": 2,
    "batch_size": 4,
    "num_epochs": 100,
    "learning_rate": 0.001
}

#experiment = Experiment(project_name="practice_ff")
#experiment = Experiment(api_key="6DEbIEUEuvzFpDeEnVuF3UEV9",
                        #project_name="Billboard_ff", workspace="rsalganik1123")
#experiment.log_parameters(hyper_params)

###################################################

csv = pd.read_csv("./Datasets/Spotify/Rising.csv")
csv = shuffle(csv, random_state=44) 
 
pd_data = csv[['Danceability','Energy','Key','Loudness','Mode','Speechiness','Acousticness','Instrumentalness','Liveness','Valence','Tempo']]
pd_labels =  csv['Target']

X_train, X_test, y_train, y_test = train_test_split(pd_data, pd_labels, test_size=.2, random_state = 44)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# print(type(X_train)) 
# print(type(X_test))

#Convert to Numpy
y_train = y_train.to_numpy().ravel() 
y_test = y_test.to_numpy().ravel() 
#X_train = X_train.to_numpy()
#X_test = X_test.to_numpy() 

# #convert to pytorch 
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

# #Load tensor dataset 
train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

# #Loaders 
train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

# dataiter = iter(train_loader)
# vals, lables = dataiter.next() 

all_categories = [0, 1]
def categoryFromOutput(output): 
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item() 
    return all_categories[category_i]

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

net = Net(len(pd_data.columns),6, 2).float() #18 normally for all features
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(net.parameters(), lr=0.001) #0.001 = 81% 
 
y_score, y = [], [] 
t_auc, t_precision, t_acc, t_recall, t_f1 = [], [], [], [], []    
for epoch in range(100): 
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
           
    print("***********EPOCH", epoch , "************")
    # print ("roc_auc_score: " + str(roc_auc_score(y, y_score)))
    # print ("accuracy: " + str(accuracy_score(y, y_score)) ) 
        
    t_precision.append(average_precision_score(y, y_score))
    t_auc.append(roc_auc_score(y, y_score))
    t_acc.append(accuracy_score(y, y_score))
    t_recall.append(recall_score(y,y_score))
    t_f1.append(f1_score(y,y_score))
print("TRAINING INFO")
print ("accuracy: " + str(accuracy_score(y, y_score)) ) 
print ("roc_auc_score: " + str(roc_auc_score(y, y_score)))
print ("average_precision_score: " + str(average_precision_score(y, y_score)))
print('recall:' + str(recall_score(y, y_score)))
print ('f1:' + str(f1_score(y, y_score)))
        
# plt.plot(train_loss)
# plt.show()

print('Finished Training')
#PATH = './ff_1.pth'
#torch.save(net.state_dict(), PATH)
test = True
if test: 
    y_score, y = [], [] 
    with torch.no_grad():
        for data in test_loader: 
            vals,labels = data
            outputs = net(vals.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_score.append(categoryFromOutput(outputs.cpu()))
            y.append(labels.cpu().numpy())
    print("TESTING INFO")
    print ("accuracy: " + str(accuracy_score(y, y_score)) ) 
    print ("roc_auc_score: " + str(roc_auc_score(y, y_score)))
    print ("average_precision_score: " + str(average_precision_score(y, y_score)))
    print('recall:' + str(recall_score(y, y_score)))
    print ('f1:' + str(f1_score(y, y_score)))