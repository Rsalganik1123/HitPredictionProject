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
import torch.utils.data as data_utils



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
experiment = Experiment(api_key="6DEbIEUEuvzFpDeEnVuF3UEV9",
                        project_name="Billboard_ff", workspace="rsalganik1123")
experiment.log_parameters(hyper_params)



###################################################

csv = pd.read_csv("./Datasets/Spotify/B+F+P.csv")
pd_data, pd_labels = csv.iloc[:, 2:-1], csv.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(pd_data, pd_labels, test_size=.2)

#Convert to Numpy
y_train = y_train.to_numpy().ravel() 
y_test = y_test.to_numpy().ravel() 
X_train = X_train.to_numpy()
X_test = X_test.to_numpy() 

#convert to pytorch 
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

#Load tensor dataset 
train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

#Loaders 
train_loader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=4, shuffle=True)

dataiter = iter(train_loader)
vals, lables = dataiter.next() 


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

#net = Net(18,10, 2).float() 
criterion = nn.CrossEntropyLoss()

net = Net(hyper_params['input_size'], hyper_params['hidden_size'], hyper_params['num_classes']).float()

#optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001) #0.001 = 81% 

with experiment.train(): 
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

            #Log to comet.ml
        experiment.log_metric("accuracy", correct / total, step=i)
        experiment.log_metric("loss", loss)
        if epoch % 10 == 0: 
            print(epoch, running_loss/len(train_loader))
        

print('Finished Training')
PATH = './ff_1.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(test_loader)
images = labels = dataiter.next()
net.load_state_dict(torch.load(PATH))
with experiment.test() : 
    correct = 0 
    total = 0 
    for vals, labels in test_loader: 
        outputs = net(vals.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    experiment.log_metric("accuracy", 100 * correct / total)
    print('Test Accuracy of the model : %d %%' % (100 * correct / total))

# correct = 0 
# total = 0 
# with torch.no_grad():
#     for data in test_loader: 
#         vals,labels = data
#         outputs = net(vals.float())
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print('Accuracy : %d %%' % (
#     100 * correct / total))

