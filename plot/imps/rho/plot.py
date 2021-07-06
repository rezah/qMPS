from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib as mpl



mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 1




R=np.loadtxt("rhoiMPSq2l2.txt")
l22=R[:,0]
l22r=R[:,1]


R=np.loadtxt("rhoiMPSq2l3.txt")
l23=R[:,0]
l23r=R[:,1]



R=np.loadtxt("rhoiMPSq2l4.txt")
l24=R[:,0]
l24r=R[:,1]

R=np.loadtxt("rhoiMPSq3l4.txt")
l34=R[:,0]
l34r=R[:,1]


R=np.loadtxt("rhoiMPSq1l4.txt")
l14=R[:,0]
l14r=R[:,1]



plt.figure(figsize=(8, 6))

r'$\tau=10$'
plt.plot( l14r,'--', lw=4,color = '#cf729d', label=r'q=1,$\tau=4$')
plt.plot( l22r,'--', lw=4,color = '#ce5c00', label=r'q=2,$\tau=2$')
plt.plot( l23r, '--', lw=4,color = '#0b8de3', label=r'q=2,$\tau=3$')
plt.plot( l24r, '--', lw=4,color = '#5c3566', label=r'q=2,$\tau=4$')
plt.plot( l34r, '--', lw=4,color = '#e91e7a', label=r'q=3,$\tau=4$')




plt.yscale('log')
plt.xscale('log')



#plt.title('qmps')
plt.ylabel(r'$\mathcal{F}$',fontsize=21)
plt.xlabel(r'$iterations$',fontsize=21)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')


#plt.xlim([1,2000])
#plt.ylim([1.e-1, 1.2e0])

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.legend(loc="upper right", prop={'size': 18})

plt.grid(True)
plt.savefig('Brickwall.pdf')
plt.clf()
