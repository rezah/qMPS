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


R=np.loadtxt("L8.txt")
x=R[:,1]
y=R[:,2]



plt.plot( x, y, '>', color = '#0b8de3', label='qmps, q=4, brickwall')
plt.yscale("log")
#plt.title('qmps')
plt.ylabel(r'$\delta$ E')
plt.xlabel(r'$parameters$')
plt.axhline(0.00122,color='black', label='D=16')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('qmps.pdf')
plt.clf()
