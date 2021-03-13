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


R=np.loadtxt("l8rand.txt")
x8=R[:,0]
y8=R[:,1]
error8=R[:,2]
error8=sort_low(error8)   


R=np.loadtxt("l6rand.txt")
x6=R[:,0]
y6=R[:,1]
error6=R[:,2]
error6=sort_low(error6)   

R=np.loadtxt("l12best.txt")
x12=R[:,0]
y12=R[:,1]
error12=R[:,2]
error12=sort_low(error12)   


R=np.loadtxt("l12good.txt")
x12g=R[:,0]
y12g=R[:,1]
error12g=R[:,2]
error12g=sort_low(error12g)   






#fig=plt.figure(figsize=(5,8))

plt.loglog( error6, '>', color = '#0b8de3', label='lay=6, random')

plt.loglog( error8, '+', color = '#e3570b', label='lay=8, random')

plt.loglog( error12, 'x', color = '#e30b69', label='lay=12, good')
plt.loglog( error12g, '4', color = '#e3360b', label='lay=12, best')




plt.ylabel(r'$\delta$ E')
plt.xlabel(r'$n$')
plt.axhline(4.1178e-06, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('qmps-plot.pdf')
plt.clf()
