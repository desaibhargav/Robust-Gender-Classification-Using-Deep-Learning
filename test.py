#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 13:50:28 2019

@author: bhargavdesai
"""
import numpy as np 
import librosa
from keras.models import load_model
import os
import soundfile as sf

test_path="/Users/bhargavdesai/Desktop/Mono/"

Model = load_model('/Users/bhargavdesai/Desktop/Projects/Music Recommender System/Code/MR/mr2_(800-800)_trained-on-mac.h5')
Model.summary()
x_in= np.empty((3,640000))
for audiofile in os.listdir(test_path):
    number=0
    print(audiofile)
    try:
        y, sr = sf.read(os.path.join(test_path,audiofile),always_2d=False)
        y = librosa.resample(y, sr, 8000)
        y = y[(8000*2):(8000*82)]
        print(y.shape)
        x_in[number,:] = y 
        number=number+1
    except RuntimeError:
        print(".DS_Store file detected and dismissed")
        pass

x_in = x_in.reshape(x_in.shape[0],800,800)
result = Model.predict(x_in)
m=0
for r in result:
    print(r)
    







