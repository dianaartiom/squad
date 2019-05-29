# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:00:10 2019

@author: mehmo
"""
#import video_dataset
import cv2
import torch.nn as nn
import torch
import numpy as np
import os

#
#video_dataset = video_dataset.videoDataset()

#video_dataset.videoDataset.display()
def processvideo(path):
    channels = 2
    depth = 20
    f = np.zeros((20,58,58))
    result = []
    listOfFolder = os.listdir(path)
    counter = 0
    input = []
    labels = []
    for folder in listOfFolder:
        
        video_list = os.listdir(path + folder)
        for video in video_list:
            
            cap = cv2.VideoCapture(path+folder+'/'+video)
            frames = []
            for ch in range(channels):
                i = 0
                while(i<depth):
                    ret,frame = cap.read()
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame,(58,58), interpolation = cv2.INTER_AREA)
                    f[i] = frame
                    i+=1
                frames.append(f)
            input.append(frames)
            labels.append(counter)
#            result.append({'frame': input, 'label': counter})
        counter =counter + 1
    return input,labels
#            inputs = torch.FloatTensor(input)
#            conv3layer1 = nn.Conv3d(channels,64,3,1,1)
#            layer1 = conv3layer1(inputs)
#            conv3layer2 = nn.Conv3d(64,128,3,1,1)
#            layer2 = conv3layer2(layer1)
#            print(layer2.shape,layer1.shape,inputs.shape)

#data = processvideo('C:/Users/mehmo/Downloads/UCF-102/')
#torch.FloatTensor(frames)
#conv3layer1 = nn.Conv3d(channels,64,3,1,1)
#layer1 = conv3layer1(inputs)
#conv3layer2 = nn.Conv3d(64,128,3,1,1)
#layer2 = conv3layer2(layer1)
#layer2.shape,layer1.shape,inputs.shape