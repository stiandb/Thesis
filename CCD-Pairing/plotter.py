import numpy as np
import sys
sys.path.append('../')
from CCD_pairing import *
from utils import PairingFCIMatrix
import matplotlib.pyplot as plt
import sys

n = 10
E_ccd = np.zeros(n)
g_array = np.linspace(1,5,n)
for i,g in enumerate(g_array):
    particles = 4
    delta = 1
    alpha = 0.4
    n = 4 #basis
    E,E_ref = run(particles,n,delta,g,alpha=alpha)
    E_ccd[i] = E + E_ref

np.save('E_ccd.npy',E_ccd)



