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


R=np.loadtxt("l6.txt")
l6=R[:,1]
l6r=R[:,2]


R=np.loadtxt("l14.txt")
l14=R[:,1]
l14r=R[:,2]



R=np.loadtxt("l24.txt")
l24=R[:,1]
l24r=R[:,2]


R=np.loadtxt("l44.txt")
l44=R[:,1]
l44r=R[:,2]



plt.figure(figsize=(9, 5))


plt.plot( l6, '+', color = '#191970', label='Q=3, L=6')
plt.plot( l14,'2', color = '#cf729d', label='Q=3, L=14')
plt.plot( l24,'h', color = '#c22a0c', label='Q=3, L=24')
plt.plot( l44,'o', color = '#c30be3', label='Q=3, L=44')



#plt.yscale('log',base=10)
#plt.xscale('log',base=10)



#plt.yscale('log')
#plt.xscale('log')



#plt.title('qmps')
plt.ylabel(r'$E$')
plt.xlabel(r'$iterations$')
plt.axhline(-0.886294376,color='black', label='Exact')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('Brickwall.pdf')
plt.clf()
