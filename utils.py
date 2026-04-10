import numpy as np 
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def normalize(X) : 
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_train_norm = (X - mu) / sigma
   