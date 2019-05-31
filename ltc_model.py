# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:57:06 2019

@author: mehmo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import preprocessing
import pdb
import matplotlib.pyplot as plt
import numpy as np

class LTC(nn.Module):
    """
    The LTC network.
    """

    def __init__(self, num_classes,opt, d, pretrained=False):
        super(LTC, self).__init__()

        self.conv1 = nn.Conv3d(opt[0], 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
       
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        oT = math.floor(opt[1]/16); # 4 times max pooling of 1/2
        oH = math.floor(opt[2]/32); # 5 times max pooling of 1/2
        oW = math.floor(opt[3]/32); # 5 times max pooling of 1/2
#        pdb.set_trace()
        self.fc6 = nn.Linear(256*oT*oH*oW, 2048)
        
        self.dropout1 = nn.Dropout(d)
        self.fc7 = nn.Linear(2048, 2048)
        
        self.dropout2 = nn.Dropout(d)
        self.fc8 = nn.Linear(2048, num_classes)
        self.softmax = nn.LogSoftmax()
        self.relu = nn.ReLU()


    def forward(self, x):
        opt = [4, 20, 58, 58]   # Need to change this
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        x = self.relu(self.conv4(x))
        x = self.pool4(x)

        x = self.relu(self.conv5(x))
        x = self.pool5(x)

        oT = math.floor(opt[1]/16); # 4 times max pooling of 1/2
        oH = math.floor(opt[2]/32); # 5 times max pooling of 1/2
        oW = math.floor(opt[3]/32); # 5 times max pooling of 1/2
#        pdb.set_trace()
        x = x.view(x.size()[0],256*oT*oH*oW)
        x = self.relu(self.fc6(x))

        x = self.dropout1(x)

        x = self.relu(self.fc7(x))

        x = self.dropout2(x)
        
        result = self.fc8(x)
        
#        return result
        result = self.softmax(self.fc8(x))
        return result.exp()
    
    
#inputs = torch.rand(1, 3, 16, 112, 112)
#net = LTC(num_classes=3,opt =  [2, 100, 58, 58],d =0.5, pretrained=True)
#outputs = net.forward(inputs)
#print(outputs.size())
        
trainpath = "G:/DA - Hildeshim/Project/LTC/Dataset/Check/"
testpath = "G:/DA - Hildeshim/Project/LTC/Dataset/Test/"
labelpath = "G:/DA - Hildeshim/Project/LTC/Dataset/Label.txt"
class_num = 5
channels = 4
timeDepth,xSize,ySize = 20,58,58 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("Processing Dataset...")
traintensor,labels_train = preprocessing.processvideo(trainpath,labelpath,channels,timeDepth,xSize,ySize)
testtensor,labels_test = preprocessing.processvideo(testpath,labelpath,channels,timeDepth,xSize,ySize)
print("Dataset processed successfully...")
print("Training Files:",traintensor.shape[0])
print("Channels:",traintensor.shape[1])
print("Timedepth:",traintensor.shape[2])
print("xSize:",traintensor.shape[3])
print("ySize:",traintensor.shape[4])
#data_test,labels_test = preprocessing.processvideo(test,labelpath,channels,timeDepth,xSize,ySize)

model = LTC(num_classes=class_num, opt = [channels, timeDepth, xSize, ySize],d =0.5, pretrained=True) # i have to change that
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss = nn.NLLLoss()

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

def train(traintensor,label):
    model.train()
    index = np.arange(traintensor.shape[0])
    np.random.shuffle(index)
#    for i,t in enumerate(index):
#    tensor = torch.FloatTensor(1,channels, timeDepth, xSize, ySize)
#    tensor[:,:,:,:,:] = traintensor[t]
    optimizer.zero_grad()   
    output = model.forward(traintensor)
    L = loss(output, torch.tensor(label))
    print(L)
    L.backward()
    optimizer.step()
    return L
    

def test(testtensor,label):
    model.eval()
    output = model(testtensor)
    L = loss(output, torch.tensor(label))
    return L

trainLoss = []
testLoss = []
epochs = 10

for e in range(epochs):
   print("Epoch: ",e+1,"/",epochs)
   trainLoss.append(train(traintensor,labels_train))
#   print(trainLoss)
   testLoss.append(test(testtensor,labels_test))


fig, axs = plt.subplots(1, 1, figsize=(10, 10))
fig.suptitle("Loss vs Epochs")
   
axs.plot(range(epochs),trainLoss)
axs.plot(range(epochs),trainLoss, "o")
axs.plot(range(epochs),testLoss)
axs.plot(range(epochs),testLoss, "x")
axs.set_ylabel('loss values')
axs.set_xlabel('Epochs')
plt.show()
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    