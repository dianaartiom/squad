# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:57:06 2019

@author: mehmo
"""

import torch
import torch.nn as nn

import math

class LTC(nn.Module):
    """
    The LTC network.
    """

    def __init__(self, num_classes,opt, d, pretrained=False):
        super(LTC, self).__init__()

        self.conv1 = nn.Conv3d(opt[1], 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.t1 = nn.Threshold(0, 0,True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.t2 = nn.Threshold(0, 0,True)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.t3 = nn.Threshold(0, 0,True)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.t4 = nn.Threshold(0, 0,True)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.t5 = nn.Threshold(0, 0,True)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        oT = math.floor(opt[2]/16); # 4 times max pooling of 1/2
        oH = math.floor(opt[3]/32); # 5 times max pooling of 1/2
        oW = math.floor(opt[4]/32); # 5 times max pooling of 1/2
        
        self.fc6 = nn.Linear(256*oT*oH*oW, 2048)
        self.t6 = nn.Threshold(0, 0,True)
        self.dropout1 = nn.Dropout(d)
        self.fc7 = nn.Linear(2048, 2048)
        self.t7 = nn.Threshold(0, 0,True)
        self.dropout2 = nn.Dropout(d)
        self.fc8 = nn.Linear(2048, num_classes)
        self.softmax = nn.LogSoftmax()
        self.relu = nn.ReLU()


    def forward(self, x):
        opt = inputs.size()
        x = self.relu(self.conv1(x))
        x = self.t1(x)
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.t2(x)
        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.t3(x)
        x = self.pool3(x)

        x = self.relu(self.conv4(x))
        x = self.t4(x)
        x = self.pool4(x)

        x = self.relu(self.conv5(x))
        x = self.t5(x)
        x = self.pool5(x)
        
        oT = math.floor(opt[2]/16); # 4 times max pooling of 1/2
        oH = math.floor(opt[3]/32); # 5 times max pooling of 1/2
        oW = math.floor(opt[4]/32); # 5 times max pooling of 1/2

        x = x.view(256*oT*oH*oW)
        x = self.relu(self.fc6(x))
        x = self.t6(x)
        x = self.dropout1(x)
        x = self.relu(self.fc7(x))
        x = self.t7(x)
        x = self.dropout2(x)

        result = self.softmax(self.fc8(x))

        return result
    
    
inputs = torch.rand(1, 3, 16, 112, 112)
net = LTC(num_classes=101,opt = inputs.size(),d =0.5, pretrained=True)
outputs = net.forward(inputs)
print(outputs.size())

        

    