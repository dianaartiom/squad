#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:53:58 2019

@author: dianaartiom
"""

import pandas as pd
import numpy as np

X_train = pd.read_csv("../data/train.csv", sep=',')
X_test = pd.read_csv("../data/test.csv", sep=',')

movie_names = ["InSight", "The Princess Diaries 2: Royal Engagement", "Whiplash",
               "	Kahaani", "Marine Boy", "The Possession", "A Mighty Wind", "Rocky", 
               "American Beauty", "Minority Report", "Skinning", "The Invisible Woman",
               "Chalet Girl", "Transporter 2", "Lost in Space", "Black Sheep",
               "The Spanish Prisoner", "The Transformers: The Movie", 
               "Changing Lanes", "The Intouchables"]

movie_trailers = ["https://www.youtube.com/watch?v=aU3DwqwpIk0"]

A = X_train.loc[X_train['title'].isin(movie_names)]
A.to_csv(path_or_buf="../data/20Train.csv")