import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
np.random.seed(22)
seed_simulator = 47

iris = load_iris()
X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print('train dataset shape', X_train.shape)
print('test dataset shape', X_test.shape)
n=9
X_train = X_train[:n,:]
y_train = y_train[:n]
print(y_train)

l1 = X_train.shape[1]
l2 = 4


y_rotation = YRotation(bias=True)
layer1 = AnsatzRotationLinear(l1,l2,n_weights_a= int(np.ceil(np.log2(l1))),n_weights_r = int(np.ceil(np.log2(l1)))+1,rotation=y_rotation,ansatz=y_rotation_ansatz,seed_simulator=seed_simulator,n_parallel=1)
layer2 = IntermediateAnsatzRotationLinear(l2,3,n_weights_a= l2,n_weights_r = l2+1,rotation=y_rotation,ansatz_i=y_rotation_ansatz,ansatz_a=y_rotation_ansatz,seed_simulator=seed_simulator,n_parallel=1)

layers = [layer1,layer2]
loss_fn = cross_entropy()
model = QDNN(layers,loss_fn,classification=True)

model.fit(X=X_train,y=y_train,method='Powell')
y_pred = model.forward(X_test)
y_pred = np.argmax(y_pred,axis=1)

print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))

