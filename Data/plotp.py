from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from numpy import linalg as LA





R=np.loadtxt("GateL16q3lay3fsim.txt")
xq2l6=R[:,0]
yq2l6=R[:,1]
errorq2l6=R[:,2]

R=np.loadtxt("GateL16q3lay3su4.txt")
xq3l8=R[:,0]
yq3l8=R[:,1]
errorq3l8=R[:,2]





plt.loglog( xq2l6, yq2l6, '>', color = '#0b8de3', label='L=16, q=3, lay=3, U(1) gates')

plt.loglog( xq3l8, yq3l8, '+', color = '#e3570b', label='L=16, q=3, lay=3, su(4) gates')



plt.title('qmps')
plt.ylabel(r'$\delta$ E')
plt.ylabel(r'$1-F$')
plt.xlabel(r'$n$')
#plt.axhline(0.0391549,color='black', label='D=4')
plt.axhline(0.005, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('qmps-plot-ladder.pdf')
plt.clf()
