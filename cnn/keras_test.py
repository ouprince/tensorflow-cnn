from __future__ import print_function
from keras.preprocessing import sequence  
from keras.models import Sequential  
from keras.layers import Dense,Dropout,Activation  
from keras.layers import Embedding  
from keras.layers import Conv1D,MaxPooling1D
from keras.datasets import imdb  
from keras.models import model_from_json  
import numpy as np  
# set parameters:  
max_features = 5001  
maxlen = 100  
batch_size = 32  
embedding_dims = 50  
filters = 250  
kernel_size = 3  
hidden_dims = 250  
epochs = 10  

x_train=np.loadtxt("x_train",dtype=int)  
y_train=np.loadtxt("y_train",dtype=int) 
