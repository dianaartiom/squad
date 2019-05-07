# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:57:06 2019

@author: mehmo
"""

import torch
import torch.nn as nn
from mypath import Path
import math

class LTC(nn.Module):
    """
    The LTC network.
    """

    def __init__(self, num_classes, pretrained=False,opt):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(opt.sampleSize[1], 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
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
        
        oT = math.floor(opt.sampleSize[2]/16); # 4 times max pooling of 1/2
        oH = math.floor(opt.sampleSize[3]/32); # 5 times max pooling of 1/2
        oW = math.floor(opt.sampleSize[4]/32); # 5 times max pooling of 1/2
        
        self.fc6 = nn.Linear(256*oT*oH*oW, 2048)
        self.t6 = nn.Threshold(0, 0,True)
        self.dropout = nn.Dropout(opt.dropout)
        self.fc7 = nn.Linear(2048, 2048)
        self.t7 = nn.Threshold(0, 0,True)
        self.dropout = nn.Dropout(opt.dropout)
        self.fc8 = nn.Linear(2048, num_classes)
        self.softmax = nn.LogSoftmax()


    def forward(self, x):

        

    