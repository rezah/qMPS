from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from numpy import linalg as LA





R=np.loadtxt("QmpsPointsq2lay6p.txt")
xq2l6=R[:,0]
yq2l6=R[:,1]
errorq2l6=R[:,2]

R=np.loadtxt("QmpsPointsq3lay8p.txt")
xq3l8=R[:,0]
yq3l8=R[:,1]
errorq3l8=R[:,2]


R=np.loadtxt("QmpsPointsq4lay18p.txt")
xq4l18=R[:,0]
yq4l18=R[:,1]
errorq4l18=R[:,2]




plt.loglog( xq2l6, errorq2l6, '>', color = '#0b8de3', label='q=2, lay=6')

plt.loglog( xq3l8, errorq3l8, '+', color = '#e3570b', label='q=3, lay=8')


plt.loglog( xq4l18, errorq4l18, 'o', color = '#72cfbb', label='q=4, lay=18')


plt.title('qmps')
plt.ylabel(r'$\delta$ E')
plt.xlabel(r'$n$')
plt.axhline(0.0391549,color='black', label='D=4')
plt.axhline(0.00647, color='black', label='D=9')
plt.axhline(0.000355, color='black', label='D=16')

plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('qmps-plot-ladder.pdf')
plt.clf()
