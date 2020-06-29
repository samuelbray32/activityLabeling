#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:27:41 2020

@author: sam
"""

from waveletFunct import haar_wavelet
from waveletFunct import db2_wavelet
import numpy as np
from ssm import LDS

class activityClassifier_LDS:
     
    def __init__(self, classifier = None, a = np.array([2,3,4,5,6]),
                 square = [], nx=None, positive=False, D=1):
        self.classifier = classifier  #note: leaves option to pass pre-fit classifier
        self.a = a.astype(int)
        self.square = square
        self.nx = nx
        self.positive=positive
        self.D=D
        
    def waveletTransform(self, data, pc = None):
        """
        Converts timeseries into wavelet 
        """
        #If more than single dimensional timetrace, pas to _mult version
        if len(data.shape)>1:
            return self.waveletTransform_mult(data)
        Q = np.zeros((data.size,self.a.size))+.01
        for i,aa in enumerate(self.a):
            if pc in self.square:
                trans = haar_wavelet(np.abs(data),aa)
                Q[:,i] = trans 
            else:
                trans = haar_wavelet(data,aa)
                Q[:,i] = trans 
        if self.positive:
            return np.abs(Q)
        return Q
    
    def waveletTransform_mult(self, data):
        """
        Converts timeseries into wavelet 
        Uses multiple pc features
        """
        Q = np.zeros((data.shape[0],self.a.size*data.shape[1]))+.01
        #for each PC
        for j in range(data.shape[1]):
        #for each wavelet dilation
            for i,aa in enumerate(self.a):
                if j in self.square:
                    trans = haar_wavelet(np.abs(data[:,j]),aa)
                    Q[:,j*self.a.size+i] = trans
                else:
                    trans = haar_wavelet(data[:,j],aa)
                    Q[:,j*self.a.size+i] = trans
        if self.positive:
            return np.abs(Q)
        return Q   
    
    def predict_wav(self, wav, ):
        elbos, posteriors = self.classifier.approximate_posterior(wav,num_iters=1)
        z = posteriors.mean_continuous_states[0]
        y_smooth = self.classifier.smooth(z, wav)
        return z, y_smooth
    
    def fit_raw(self, data, sub=1, num_iters=10, dynamicsOnly=False):
        if type(data)==np.ndarray:
            self.fit_raw_list([data,], sub=sub,num_iters=num_iters, dynamicsOnly=dynamicsOnly)
        else:
            self.fit_raw_list(data, sub=sub,num_iters=num_iters, dynamicsOnly=dynamicsOnly) #deprecated
#        if not self.nx==None:
#            data = data[:,:self.nx]
#        if type(data) == list:
#            self.fit_raw_list(data, n=n, sub=sub,num_iters=num_iters)
#            return
#        wav = self.waveletTransform(data)
#        self.fit_wav(wav, n=n, sub=sub)
        return
    
    def fit_raw_list(self, data_list, sub=1,num_iters=10, dynamicsOnly=False):
        wav = []
        lengths=[]
        for data in data_list:
            if not self.nx==None:
                data = data[:,:self.nx]
            wav.append(self.waveletTransform(data))
            lengths.append(len(data))
        lengths = np.array(lengths)
        if self.classifier==None:
            self.classifier = LDS(self.nx*self.a.size, self.D, emissions="gaussian")
        self.classifier.fit(wav, method="laplace_em", 
                            variational_posterior="structured_meanfield",
                            num_iters=num_iters, initialize=True,
                            learn_emissions=(not dynamicsOnly))
        return
    
    def predict_raw(self, data,):
        if not self.nx==None:
            data = data[:,:self.nx]
        wav = self.waveletTransform(data)
        return self.predict_wav(wav,)
    
    def mean_z(self):
        if self.classifier==None:
            return None
        return np.matmul(np.linalg.inv(np.eye(1)-self.classifier.dynamics.A),self.classifier.dynamics.b)
    

class activityClassifier_LDS_db2:
     
    def __init__(self, classifier = None, a = np.array([2,3,4,5,6]),
                 square = [], nx=None, positive=False, D=1):
        self.classifier = classifier  #note: leaves option to pass pre-fit classifier
        self.a = a.astype(int)
        self.square = square
        self.nx = nx
        self.positive=positive
        self.D=D
        
    def waveletTransform(self, data, pc = None):
        """
        Converts timeseries into wavelet 
        """
        #If more than single dimensional timetrace, pas to _mult version
        if len(data.shape)>1:
            return self.waveletTransform_mult(data)
        Q = np.zeros((data.size,self.a.size))+.01
        for i,aa in enumerate(self.a):
            if pc in self.square:
                trans, dil = haar_wavelet(np.abs(data),aa)
                Q[:,i] = trans 
            else:
                trans, dil = haar_wavelet(data,aa)
                Q[:,i] = trans 
        if self.positive:
            return np.abs(Q)
        return Q
    
    def waveletTransform_mult(self, data):
        """
        Converts timeseries into wavelet 
        Uses multiple pc features
        """
        Q = np.zeros((data.shape[0],self.a.size*data.shape[1]))+.01
        #for each PC
        for j in range(data.shape[1]):
        #for each wavelet dilation
            for i,aa in enumerate(self.a):
                if j in self.square:
                    trans, dil = haar_wavelet(np.abs(data[:,j]),aa)
                    Q[:,j*self.a.size+i] = trans
                else:
                    trans, dil = haar_wavelet(data[:,j],aa)
                    Q[:,j*self.a.size+i] = trans
        if self.positive:
            return np.abs(Q)
        return Q   
    
    def predict_wav(self, wav, ):
        elbos, posteriors = self.classifier.approximate_posterior(wav,num_iters=1)
        z = posteriors.mean_continuous_states[0]
        y_smooth = self.classifier.smooth(z, wav)
        return z, y_smooth
    
    def fit_raw(self, data, sub=1, num_iters=10, dynamicsOnly=False):
        if type(data)==np.ndarray:
            self.fit_raw_list([data,], sub=sub,num_iters=num_iters, dynamicsOnly=dynamicsOnly)
        else:
            self.fit_raw_list(data, sub=sub,num_iters=num_iters, dynamicsOnly=dynamicsOnly) #deprecated
#        if not self.nx==None:
#            data = data[:,:self.nx]
#        if type(data) == list:
#            self.fit_raw_list(data, n=n, sub=sub,num_iters=num_iters)
#            return
#        wav = self.waveletTransform(data)
#        self.fit_wav(wav, n=n, sub=sub)
        return
    
    def fit_raw_list(self, data_list, sub=1,num_iters=10, dynamicsOnly=False):
        wav = []
        lengths=[]
        for data in data_list:
            if not self.nx==None:
                data = data[:,:self.nx]
            wav.append(self.waveletTransform(data))
            lengths.append(len(data))
        lengths = np.array(lengths)
        if self.classifier==None:
            self.classifier = LDS(self.nx*self.a.size, self.D, emissions="gaussian")
        self.classifier.fit(wav, method="laplace_em", 
                            variational_posterior="structured_meanfield",
                            num_iters=num_iters, initialize=True,
                            learn_emissions=(not dynamicsOnly))
        return
    
    def predict_raw(self, data,):
        if not self.nx==None:
            data = data[:,:self.nx]
        wav = self.waveletTransform(data)
        return self.predict_wav(wav,)
    
    def mean_z(self):
        if self.classifier==None:
            return None
        return np.matmul(np.linalg.inv(np.eye(1)-self.classifier.dynamics.A),self.classifier.dynamics.b)
    