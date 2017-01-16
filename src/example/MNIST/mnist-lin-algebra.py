'''
Created on Jul 14, 2016

@author: robertday
'''

import numpy as np

data=np.loadtxt("/Users/robertday/Dropbox/Phenotypic-Tunneling/Alex_Python_Code/datasets/mnist-train-norm.dat")
labels=np.loadtxt("/Users/robertday/Dropbox/Phenotypic-Tunneling/Alex_Python_Code/datasets/mnist-train_label.dat",dtype=int)
pos=[np.where(labels==i)[0] for i in range(10)]
data0=data[pos[0]]

def median_number(data,pos):
    median_v=[]
    for i in range(10):
        datai=data[pos[i]]
        median_i=[np.median(datai[:,j]) for j in range(784)]
        median_v.append(median_i)            
    return median_v

mv=median_number(data,pos)

# Prediction :

def predict_label(image_vector,median_vector):
    return np.argmax(np.array([np.dot(image_vector,median_vector[i]) for i in range(10)]))

def hamming_distance(v1,v2):
    l1=len(v1)
    dist=0
    
    for i in range(l1):
        if v1[i]!=v2[i]:
            dist+=1
    return dist

print("Performing predictions")
prediction_all=np.array([predict_label(d,mv) for d in data],dtype=int)

print("Accuracy of :",hamming_distance(prediction_all,labels)*1.0/len(labels))