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
import pyro
import datetime 
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


####################################
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

####################################
class NN(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size): 
        super(NN, self).__init__() 
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, x): 
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output 

net = NN(18, 10, 2).float() 
print(net)

#Optimization
params_to_update = net.parameters()
total_params = sum(p.numel() for p in net.parameters() )
print("Total number of parameters = ", total_params)

log_softmax = nn.LogSoftmax(dim=1)

def model(x_data, y_data): 
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,  'out.weight': outw_prior, 'out.bias': outb_prior}

    lifted_module = pyro.random_module("module", net, priors)
    lifted_reg_model = lifted_module()
    lhat = log_softmax(lifted_reg_model(x_data))
    
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)

softplus = nn.Softplus()

def guide(x_data, y_data):
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)
    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()

num_samples = 20 
print("Neural Network samples ", num_samples)

def predict(x): 
    sampled_models = [guide(None,None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats),0)
    return torch.argmax(mean, dim=1)

#Backprop
optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

epochs = 20
loss = 0
train_losses, test_losses = [] , [] 

for epoch in range(epochs):
    #print("time: ", datetime.now().time())
    running_loss = 0
    for batch_id, data in enumerate(train_loader):
        # calculate the loss and take a gradient step
        train = data[0].float()
        labels = data[1]
        loss = svi.step(train,labels)
        # loss.backward()
        # svi.step()

        running_loss += loss

        #loss += svi.step(data[0].view(-1,28*28), data[1])
    else:
        test_loss = 0
        total = 0
        correct = 0

        with torch.no_grad(): #add net.eval()
            net.eval()
            for batch_id, data in enumerate(test_loader):
                images = data[0].float()
                labels = data[1]
                test_loss += svi.step(images,labels)
                predicted = predict(images)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        net.train()
        train_losses.append(running_loss/len(train_loader.dataset))
        test_losses.append(test_loss/len(test_loader.dataset))

        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
            "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
            "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
            "Test Accuracy: %d %%" % (100 * correct / total))


