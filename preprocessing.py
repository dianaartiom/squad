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
import fnmatch
import re
import pandas as pd

def getlabel(labelpath):
    label = pd.read_csv(labelpath,header = None)
    labeldict = {}
    for i in range(len(label)):
        labeldict[label[0][i]] = i
    return labeldict
    
def processvideo(videopath,labelpath,channels,timeDepth,xSize,ySize):
    f = np.zeros((timeDepth,xSize,ySize))    
    labeldict = getlabel(labelpath)
    labels = []
    filelist = []
    inputs = []
    for root,dirs,files in os.walk(videopath):
        file_types = ['*.avi','*.mp4']
        file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if re.match(file_types, f)]
        for file in files:
            filelist.append(file)
            
    frametensor = torch.FloatTensor(len(filelist),channels, timeDepth, xSize, ySize)
    for num, videofile in enumerate(filelist):
        labelname = os.path.basename(os.path.dirname(videofile))
        cap = cv2.VideoCapture(videofile)
        frames = []
        for ch in range(channels):
            d = 0
            while(d<timeDepth):
                ret,frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame,(xSize,ySize), interpolation = cv2.INTER_AREA)
                    frame = torch.from_numpy(frame)
                    frametensor[num,ch,d, :, :] = frame
#                    f[i] = frame
                    d+=1
#            frames.append(f)
#        inputs.append(frames)
        labels.append(labeldict[labelname])
    return frametensor,labels

#channels = 4
#timeDepth = 20
#xSize, ySize = 58,58
#videopath,labelpath = "G:/DA - Hildeshim/Project/LTC/Dataset/One/","G:/DA - Hildeshim/Project/LTC/Dataset/Label.txt"
#frametensor,labels = processvideo(videopath,labelpath,channels,timeDepth,xSize, ySize)
#print("here")
#data = frametensor
#for m in range(1):
#    for i in range(4):
#        for j in range(4):
#            print(i,j,torch.all(torch.eq(data[m][i],data[m][j])))
    
