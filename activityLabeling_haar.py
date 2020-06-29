#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:23:10 2020

@author: sam
"""

from waveletFunct import haar_wavelet
import numpy as np
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm

class activityClassifier_HMM_haar:
     
    def __init__(self, classifier = None, a = np.array([2,3,4,5,6]), merge_states = [], square = [], nx=None):
        self.classifier = classifier  #note: leaves option to pass pre-fit classifier
        self.a = a.astype(int)
        self.merge_states = merge_states
        self.square = square
        self.nx = nx
        
    def waveletTransform(self, data, pc = None):
        """
        Converts timeseries into wavelet 
        """
        #If more than single dimensional timetrace, pas to _mult version
        if len(data.shape)>1:
            return self.waveletTransform_mult(data)
        Q = np.zeros((data.size,self.a.size))+.01
        for i,aa in enumerate(self.a):
#            Q[aa//2:-aa//2+1,i] = sin_wavelet(data,aa,normalize=True)
#            Q[:,i] = sin_wavelet(data,aa,normalize=True)
            if pc in self.square:
                trans = haar_wavelet(np.abs(data),aa)
                Q[:,i] = trans 
            else:
                trans = haar_wavelet(data,aa)
                Q[:,i] = trans 
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
#            print(j)
            for i,aa in enumerate(self.a):
#                print(i)
    #            Q[aa//2:-aa//2+1,i] = sin_wavelet(data,aa,normalize=True)
    #            Q[:,i] = sin_wavelet(data,aa,normalize=True)
                if j in self.square:
                    trans = haar_wavelet(np.abs(data[:,j]),aa)
                    Q[:,j*self.a.size+i] = trans
                else:
                    trans = haar_wavelet(data[:,j],aa)
                    Q[:,j*self.a.size+i] = trans
        return Q
    
    def fit_wav(self, wav, n=2, sub=1):
        if self.classifier==None:
            self.classifier = hmm.GaussianHMM(n_components=n, covariance_type='full')
        self.classifier.fit(wav[::sub])
        return
    
    def fit_wav_list(self, wav_list, n=2, sub=1):
        wav = []
        lengths=[]
        for x in wav_list:
            wav.extend(x)
            lengths.append(x.shape[0])
        wav = np.array(wav)
        lengths = np.array(lengths)
        if self.classifier==None:
            self.classifier = hmm.GaussianHMM(n_components=n, covariance_type='full')
        self.classifier.fit(wav[::sub],lengths)
        return
    
    def predict_wav(self, wav, prob=False):
        if prob:
            return self.smush_states(self.classifier.decode(wav)[1]), self.classifier.predict_proba(wav)
        return self.smush_states(self.classifier.decode(wav)[1])
    
    def fit_raw(self, data, n=2, sub=1):
        if not self.nx==None:
            data = data[:,:self.nx]
        if type(data) == list:
            self.fit_raw_list(data, n=n, sub=sub)
            return
        wav = self.waveletTransform(data)
        self.fit_wav(wav, n=n, sub=sub)
        return
    
    def fit_raw_list(self, data_list, n=2, sub=1):
        wav = []
        lengths=[]
        for data in data_list:
            if not self.nx==None:
                data = data[:,:self.nx]
            wav.extend(self.waveletTransform(data))
            lengths.append(len(data))
        wav = np.array(wav)
        lengths = np.array(lengths)
        sp = np.ones(n)*.01
        sp = sp/np.sum(sp)       
        if self.classifier==None:
            self.classifier = hmm.GaussianHMM(n_components=n, covariance_type='full',n_iter=300, min_covar = .01)
        self.classifier.fit(wav[::sub],lengths)
#        self.fit_wav(wav, n=n, sub=sub)
        return
    
    def predict_raw(self, data, prob=False):
        if not self.nx==None:
            data = data[:,:self.nx]
        wav = self.waveletTransform(data)
        return self.predict_wav(wav, prob=prob)
    
    def order_states(self,):
        """
        orders states by two norm of covariance matrix
        """
        val = np.sum(self.classifier.covars_**2,(1,2))
        order = np.argsort(val)
        # put necessary components in correct order
        self.classifier.startprob_ = self.classifier.startprob_[order]
        self.classifier.transmat_ = self.classifier.transmat_[order,:][:,order]
        self.classifier.means_ = self.classifier.means_[order]
        self.classifier.covars_ = self.classifier.covars_[order,:,:] 
        return
    
    def smush_states(self, labels):
        """
        Merges labels in merge states and shifts all to sequential from 0
        """
        if len(self.merge_states)>1:
            print('WARNING: smush states not equipped to resolve multiple merges') # TODO if needed
        for pair in self.merge_states:
            labels[(labels == pair[1])] = pair[0]
            labels[(labels > pair[1])] -= 1
        return labels
            
    def transmat(self, full = False):#Needs fixin'
        if full:
            return self.classifier.transmat_
        t_new = self.classifier.transmat_.copy()
        #add from states to delete
        for pair in self.merge_states:
            t_new[pair[0],:] += t_new[pair[1],:]
            t_new[:,pair[0]] += t_new[:,pair[1]]
        #delete merged states
        for pair in self.merge_states:
            t_new = np.delete(t_new, pair[1], 0)
            t_new = np.delete(t_new, pair[1], 1)
        return t_new
    
class activityClassifier_HMM_haar_normalized:
     
    def __init__(self, classifier = None, a = np.array([2,3,4,5,6]), merge_states = [], square = [], nx=None):
        self.classifier = classifier  #note: leaves option to pass pre-fit classifier
        self.a = a.astype(int)
        self.a_norm = 1
        self.merge_states = merge_states
        self.square = square
        self.nx = nx
        
    def waveletTransform(self, data, pc = None):
        """
        Converts timeseries into wavelet 
        """
        #If more than single dimensional timetrace, pas to _mult version
        if len(data.shape)>1:
            return self.waveletTransform_mult(data)
        Q = np.zeros((data.size,self.a.size))+.01
        for i,aa in enumerate(self.a):
#            Q[aa//2:-aa//2+1,i] = sin_wavelet(data,aa,normalize=True)
#            Q[:,i] = sin_wavelet(data,aa,normalize=True)
            if pc in self.square:
                trans = haar_wavelet(np.abs(data),aa)
                norm = np.mean(haar_wavelet(np.abs(data[:,pc]),self.a_norm))
                Q[:,i] = trans/norm
            else:
                trans = haar_wavelet(data,aa)
                norm = np.mean(haar_wavelet(data[:,pc],self.a_norm))
                Q[:,i] = trans/norm 
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
#            print(j)
            for i,aa in enumerate(self.a):
#                print(i)
    #            Q[aa//2:-aa//2+1,i] = sin_wavelet(data,aa,normalize=True)
    #            Q[:,i] = sin_wavelet(data,aa,normalize=True)
                if j in self.square:
                    trans = haar_wavelet(np.abs(data[:,j]),aa)
                    norm = np.mean(haar_wavelet(np.abs(data[:,j]),self.a_norm))
                    Q[:,j*self.a.size+i] = trans/norm
                else:
                    trans = haar_wavelet(data[:,j],aa)
                    norm = np.mean(haar_wavelet(data[:,j],self.a_norm))
                    Q[:,j*self.a.size+i] = trans/norm
        return Q
    
    def fit_wav(self, wav, n=2, sub=1):
        if self.classifier==None:
            self.classifier = hmm.GaussianHMM(n_components=n, covariance_type='full')
        self.classifier.fit(wav[::sub])
        return
    
    def fit_wav_list(self, wav_list, n=2, sub=1):
        wav = []
        lengths=[]
        for x in wav_list:
            wav.extend(x)
            lengths.append(x.shape[0])
        wav = np.array(wav)
        lengths = np.array(lengths)
        if self.classifier==None:
            self.classifier = hmm.GaussianHMM(n_components=n, covariance_type='full')
        self.classifier.fit(wav[::sub],lengths)
        return
    
    def predict_wav(self, wav, prob=False):
        if prob:
            return self.smush_states(self.classifier.decode(wav)[1]), self.classifier.predict_proba(wav)
        return self.smush_states(self.classifier.decode(wav)[1])
    
    def fit_raw(self, data, n=2, sub=1):
        if not self.nx==None:
            data = data[:,:self.nx]
        if type(data) == list:
            self.fit_raw_list(data, n=n, sub=sub)
            return
        wav = self.waveletTransform(data)
        self.fit_wav(wav, n=n, sub=sub)
        return
    
    def fit_raw_list(self, data_list, n=2, sub=1):
        wav = []
        lengths=[]
        for data in data_list:
            if not self.nx==None:
                data = data[:,:self.nx]
            wav.extend(self.waveletTransform(data))
            lengths.append(len(data))
        wav = np.array(wav)
        lengths = np.array(lengths)
        sp = np.ones(n)*.01
        sp = sp/np.sum(sp)       
        if self.classifier==None:
            self.classifier = hmm.GaussianHMM(n_components=n, covariance_type='full',n_iter=300, min_covar = .01)
        self.classifier.fit(wav[::sub],lengths)
#        self.fit_wav(wav, n=n, sub=sub)
        return
    
    def predict_raw(self, data, prob=False):
        if not self.nx==None:
            data = data[:,:self.nx]
        wav = self.waveletTransform(data)
        return self.predict_wav(wav, prob=prob)
    
    def order_states(self,):
        """
        orders states by two norm of covariance matrix
        """
        val = np.sum(self.classifier.covars_**2,(1,2))
        order = np.argsort(val)
        # put necessary components in correct order
        self.classifier.startprob_ = self.classifier.startprob_[order]
        self.classifier.transmat_ = self.classifier.transmat_[order,:][:,order]
        self.classifier.means_ = self.classifier.means_[order]
        self.classifier.covars_ = self.classifier.covars_[order,:,:] 
        return
    
    def smush_states(self, labels):
        """
        Merges labels in merge states and shifts all to sequential from 0
        """
        if len(self.merge_states)>1:
            print('WARNING: smush states not equipped to resolve multiple merges') # TODO if needed
        for pair in self.merge_states:
            labels[(labels == pair[1])] = pair[0]
            labels[(labels > pair[1])] -= 1
        return labels
            
    def transmat(self, full = False):#Needs fixin'
        if full:
            return self.classifier.transmat_
        t_new = self.classifier.transmat_.copy()
        #add from states to delete
        for pair in self.merge_states:
            t_new[pair[0],:] += t_new[pair[1],:]
            t_new[:,pair[0]] += t_new[:,pair[1]]
        #delete merged states
        for pair in self.merge_states:
            t_new = np.delete(t_new, pair[1], 0)
            t_new = np.delete(t_new, pair[1], 1)
        return t_new