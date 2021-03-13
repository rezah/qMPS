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


R=np.loadtxt("q4lay14rand.txt")
xr10=R[:,0]
yr10=R[:,1]
errorr10=R[:,2]
errorr10=sort_low(errorr10)   


R=np.loadtxt("q4Lay6Random.txt")
xg10=R[:,0]
yg10=R[:,1]
errorg10=R[:,2]
errorg10=sort_low(errorg10)   

R=np.loadtxt("q4Lay10good.txt")
x11=R[:,0]
y11=R[:,1]
error11=R[:,2]
error11=sort_low(error11)   


R=np.loadtxt("q4Lay14good.txt")
x6=R[:,0]
y6=R[:,1]
error6=R[:,2]
error6=sort_low(error6)   



R=np.loadtxt("q4Lay18good.txt")
x4=R[:,0]
y4=R[:,1]
error4=R[:,2]
error4=sort_low(error4)   





#fig=plt.figure(figsize=(5,8))

plt.loglog( errorg10, '>', color = '#0b8de3', label='q=4, lay=6, random')

plt.loglog( error11, '+', color = '#e3570b', label='q=4, lay=10, good')

plt.loglog( error6, 'x', color = '#e30b69', label='q=4, lay=14, good')
plt.loglog( errorr10, '4', color = '#e3360b', label='q=4, lay=14, random')



plt.loglog(  error4, '2', color = '#729fcf', label='q=4, lay=18, good')

#plt.loglog( error8, 'o', color = '#72cfbb', label='lay=8')

#plt.loglog( xq4l24, errorq4l24, '2', color = '#cf729d', label='q=4, lay=24')

plt.title('qmps')
plt.ylabel(r'$\delta$ E')
plt.xlabel(r'$n$')
plt.axhline(0.00422,color='black', label='D=4')
plt.axhline(0.0001435, color='black', label='D=8')
plt.axhline(4.1178e-06, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('qmps-plot.pdf')
plt.clf()
