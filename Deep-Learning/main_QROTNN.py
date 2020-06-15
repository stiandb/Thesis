import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

"""
iris = load_iris()
X = iris['data']
y = iris['target']
np.random.seed(5)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print('train dataset shape', X_train.shape)
print('test dataset shape', X_test.shape)

l1 = X_train.shape[1]
l2 = 3
n = 5
X_train = X_train[:n,:]
y_train = y_train[:n]
layers = [AnsatzLinear(l1,l2,int(np.ceil(np.log2(l1))),y_rotation_ansatz),AnsatzLinear(l2,3,int(np.ceil(np.log2(l2))),y_rotation_ansatz)]
loss_fn = cross_entropy()
model = QDNN(layers,loss_fn)

np.save('model_params.npy',model.fit(X=X_train,y=y_train,method='Powell'))



w = np.load('model_params.npy')
X = iris['data']
y = iris['target']
model.set_weights(w)
np.save('loss.npy',model.loss)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)

print('accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
"""

iris = load_iris()
X = iris['data']
y = iris['target']
np.random.seed(7)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print('train dataset shape', X_train.shape)
print('test dataset shape', X_test.shape)

l1 = X_train.shape[1]
l2 = 8
n = 5
X_train = X_train[:n,:]
y_train = y_train[:n]
y_rotation = YRotation(bias=True)
layer1 = AnsatzRotationLinear(l1,l2,n_weights_a= int(np.ceil(np.log2(l1))),n_weights_r = int(np.ceil(np.log2(l1)))+1,rotation=y_rotation,ansatz=y_rotation_ansatz)
layer2 = AnsatzRotationLinear(l2,3,n_weights_a= int(np.ceil(np.log2(l2))),n_weights_r = int(np.ceil(np.log2(l2)))+1,rotation=y_rotation,ansatz=y_rotation_ansatz)
layers = [layer1,layer2]
loss_fn = cross_entropy()
model = QDNN(layers,loss_fn)

np.save('model_params_bias.npy',model.fit(X=X_train,y=y_train,method='Powell'))



w = np.load('model_params_bias.npy')
X = iris['data']
y = iris['target']
model.set_weights(w)
np.save('loss_bias.npy',model.loss)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)

print('accuracy:', accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))