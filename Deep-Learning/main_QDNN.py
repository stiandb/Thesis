import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split





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
layers = [Linear(l1,l2),Linear(l2,3)]
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
print(confusion_matrix(y_test,y_pred))"""

