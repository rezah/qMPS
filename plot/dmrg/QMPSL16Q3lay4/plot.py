from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from numpy import linalg as LA

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
R=np.loadtxt("cg2.txt")
cg2=R[:,1]


R=np.loadtxt("dmrgF.txt")
dmrgF=R[:,1]
R=np.loadtxt("dmrgF1.txt")
dmrgF1=R[:,1]
R=np.loadtxt("dmrgF2.txt")
dmrgF2=R[:,1]


R=np.loadtxt("lbfgs.txt")
lbfgs=R[:,1]
R=np.loadtxt("lbfgs1.txt")
lbfgs1=R[:,1]
R=np.loadtxt("lbfgs1C.txt")
lbfgs1C=R[:,1]
lbfgs1=list(lbfgs1)+list(lbfgs1C)

R=np.loadtxt("lbfgs2.txt")
lbfgs2=R[:,1]




plt.figure(figsize=(9, 5))


#plt.plot( cg, '>', color = '#0b8de3', label='cg-1')
plt.plot( cg1, '2', color = '#191970', label='cg')
#plt.plot( cg2, 'o', color = '#0b8de3', label='cg-3')
#plt.plot( dmrgF,'>', color = '#cf729d', label='dmrg-1')
plt.plot( dmrgF1,'2', color = '#cf729d', label='dmrg')
#plt.plot( dmrgF2,'o', color = '#cf729d', label='dmrg-3')

#plt.plot( lbfgs,'>', color = '#c22a0c', label='l-bfgs-b-1')
plt.plot( lbfgs1,'2', color = '#c22a0c', label='l-bfgs-b')
#plt.plot( lbfgs2,'o', color = '#c22a0c', label='l-bfgs-b-3')



#plt.yscale('log',base=10)
#plt.xscale('log',base=10)



plt.yscale('log')
plt.xscale('log')



#plt.title('qmps')
plt.ylabel(r'$1-F$')
plt.xlabel(r'$iterations$')
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('Brickwall.pdf')
plt.clf()
