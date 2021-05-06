from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as LA

mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 1


def sort_low(error):
 val_min=1.e8
 error_f=[]
 for i in range(len(error)-1):
   if error[i]>=error[i+1] and  error[i]<=val_min :
    error_f.append(error[i])
    val_min=error[i+1]*1.0
    pass

 return error_f


R=np.loadtxt("cg.txt")
cg=R[:,1]
R=np.loadtxt("cg1.txt")
cg1=R[:,1]
#R=np.loadtxt("cg2.txt")
#cg2=R[:,1]


R=np.loadtxt("dmrgF.txt")
dmrgF=R[:,1]
R=np.loadtxt("dmrgF1.txt")
dmrgF1=R[:,1]
#R=np.loadtxt("dmrgF2.txt")
#dmrgF2=R[:,1]


R=np.loadtxt("bfgs.txt")
lbfgs=R[:,1]
R=np.loadtxt("bfgsQ.txt")
lbfgsQ=R[:,1]

R=np.loadtxt("bfgsQQ.txt")
lbfgsQQ=R[:,1]

lbfgs=list(lbfgs)+list(lbfgsQ)+list(lbfgsQQ)


R=np.loadtxt("bfgs1.txt")
lbfgs1=R[:,1]
R=np.loadtxt("bfgsQ1.txt")
lbfgsQ1=R[:,1]


lbfgs1=list(lbfgs1)+list(lbfgsQ1)
#lbfgs1=list(lbfgs1)#+list(lbfgsQ1)


x_lbfgs=[ i for i in range(1,len(list(lbfgs))+1 ) ]
x_lbfgs1=[ i for i in range(1,len(list(lbfgs1))+1 ) ]
x_dmrgF=[ i for i in range(1,len(list(dmrgF))+1 ) ]
x_dmrgF1=[ i for i in range(1,len(list(dmrgF1))+1 ) ]
x_cg=[ i for i in range(1,len(list(cg))+1 ) ]
x_cg1=[ i for i in range(1,len(list(cg1))+1 ) ]


plt.figure(figsize=(8, 6))


plt.plot( x_cg1,cg1, '-', lw=4, color = '#0b8de3', label='cg')
plt.plot( x_cg,cg, '--', lw=4,color = '#0b8de3', label='cg')

#plt.plot( cg2, 'o', color = '#0b8de3', label='cg-3')
plt.plot( x_dmrgF,dmrgF,'-', lw=4,color = '#cf729d', label='dmrg')
plt.plot( x_dmrgF1,dmrgF1,'--', lw=4,color = '#cf729d', label='dmrg')
#plt.plot( dmrgF2,'o', color = '#cf729d', label='dmrg-3')

plt.plot( x_lbfgs,lbfgs,'-', lw=4,color = '#c22a0c', label='l-bfgs-b')
plt.plot( x_lbfgs1,lbfgs1,'--', lw=4,color = '#c22a0c', label='l-bfgs-b')
#plt.plot( lbfgs2,'o', color = '#c22a0c', label='l-bfgs-b-3')



#plt.yscale('log',base=10)
#plt.xscale('log',base=10)
plt.yscale('log')
plt.xscale('log')



#plt.title('qmps')
plt.ylabel(r'$1-\mathcal{F}$',fontsize=21)
plt.xlabel(r'$iterations$',fontsize=21)
#plt.axhline(0.00422,color='black', label='D=4')



plt.xlim([1,15000])
plt.ylim([1.2e-2, 1.2e0])

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="upper right", prop={'size': 18})

plt.grid(True)
plt.savefig('Brickwall.pdf')
plt.clf()
