# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:57:06 2019

@author: mehmo
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
import sklearn
import math

import pdb
import matplotlib.pyplot as plt

class NN(nn.Module):
    """
    The LTC network.
    """

    def __init__(self, input_size,d, pretrained=False):
        super(NN, self).__init__()

       
        self.fc1 = nn.Linear(input_size, 64)
        
        self.dropout1 = nn.Dropout(d)
        self.fc2 = nn.Linear(64, 128)
        
        self.dropout2 = nn.Dropout(d)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256,64)
        self.fc5 = nn.Linear(64,8)
        self.fc6 = nn.Linear(8, 1)        
        self.relu = nn.ReLU()


    def forward(self, x):
       
      
        x = self.relu(self.fc1(x))
        #x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        #x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        result = self.fc6(x)

        return result
    
    
data = pd.read_csv('data.csv')
#
#data = data.drop(['imdb_id','original_language','original_title','overview','poster_path','production_countries','release_date','status','tagline','title','Keywords','cast','crew','all_genres','collection_name'], axis=1)

data = data.drop(columns=['revenue','imdb_id','original_language','original_title','overview','poster_path','production_countries','release_date','status','tagline','title','Keywords','cast','crew','all_genres','collection_name','id','index','homepage'])
Y = np.array(data['log_revenue']).reshape(-1,1)
data = data.drop(columns='log_revenue')
X = data
scaler = StandardScaler()
X = scaler.fit_transform(X)
#X = normalize(X,axis=0)
#Y = normalize(Y,axis=0)
Y = scaler.fit_transform(Y)
print(X.shape,Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

model = NN(input_size=X_train.shape[1],d = 0.5,pretrained=True) # i have to change that

#optimizer = optim.SGD(model.parameters(), lr=1e-14, momentum=0.7)
optimizer = optim.Adam(model.parameters(),  lr=1e-3, betas=(0.9, 0.999), eps=1e-08 )
loss = nn.MSELoss()

#def train(epoch,data):
#    model.train()
#    lossArray = []
#    for i in range(0,epoch):
#        print(i)
#        for d in data:
#            
#            output = model(torch.FloatTensor(d["frame"]))
##            labels = torch.zeros(class_num,dtype=torch.int64)
##            
##            labels[] = 1
##            pdb.set_trace() 
#            target = torch.tensor([d["label"] - 1])
#            L = loss(output.view(1,class_num), target)
##            pdb.set_trace()
#            L.backward()
#            optimizer.step()
#            optimizer.zero_grad()
#            break
#        lossArray.append(L)
#    return lossArray

def train(data,label):
    model.train()
    output = model(torch.FloatTensor(data))
    L = loss(output, torch.FloatTensor(label))
    L.backward()
    optimizer.step()
    optimizer.zero_grad()
    return L
    

def test(data,label):
    model.eval()
    output = model(torch.FloatTensor(data))
    L = loss(output, torch.FloatTensor(label))
    return L,output

trainLoss = []
testLoss = []
epochs = 100
for e in range(epochs):
   print(e)
   trainLoss.append(train(X_train,y_train))
   a,b = test(X_test,y_test)
   testLoss.append(a)

print(np.array(trainLoss[90:]),np.array(testLoss[90:]))

fig, axs = plt.subplots(1, 1, figsize=(20, 10))
fig.suptitle("Loss vs Epochs")
axs.plot(range(epochs),trainLoss,label="train")
axs.plot(range(epochs),trainLoss, "o")
axs.plot(range(epochs),testLoss,label="test")
axs.plot(range(epochs),testLoss, ".")
axs.set_ylabel('loss values')
plt.legend()
plt.show()
#print(b[90:],y_test[90:])
#    test(data)

        
#input = torch.randn(3, 1, requires_grad=True)
#print(input)
#target = torch.tensor([1, 0, 0])
#output = F.nll_loss(F.log_softmax(input), target)
#print(output)   
    
    
#m = nn.LogSoftmax(dim=1)
#loss = nn.NLLLoss()
## input is of size N x C = 3 x 5
#input = torch.randn(1, 3, requires_grad=True)
## each element in target has to have 0 <= value < C
#target = torch.tensor([0])
#output = loss(m(input), target)
#output.backward()

#
## 2D loss example (used, for example, with image inputs)
#N, C = 5, 4
#loss = nn.NLLLoss()
## input is of size N x C x height x width
#data = torch.randn(N, 16, 10, 10)
#conv = nn.Conv2d(16, C, (3, 3))
#m = nn.LogSoftmax(dim=1)
## each element in target has to have 0 <= value < C
#target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
#output = loss(m(conv(data)), target)
#output.backward() 
#    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    