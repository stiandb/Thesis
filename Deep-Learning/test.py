import numpy as np
import matplotlib.pylab as plt

k=10
p = range(100,1001,900)

def classical(k,p):
	return(k*p)
def quantum(k,p):
	return(k*np.log(p))

clas = np.zeros(len(p))
quant = np.zeros(len(p))
for i,p_ in enumerate(p):
	clas[i] = classical(k,p_)
	quant[i] = quantum(k,p_)

plt.plot(p,clas,label='Dense Layer')
plt.plot(p,quant,label='Quantum Dense Layer')
plt.yscale('log')
plt.xlabel('Inputs')
plt.ylabel('Parameters')
plt.title('Number of classical versus quantum parameters for 10 nodes')
plt.show()
