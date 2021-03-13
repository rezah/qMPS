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


R=np.loadtxt("lay10rand.txt")
xr10=R[:,0]
yr10=R[:,1]
errorr10=R[:,2]
errorr10=sort_low(errorr10)   


R=np.loadtxt("lay10good.txt")
xg10=R[:,0]
yg10=R[:,1]
errorg10=R[:,2]
errorg10=sort_low(errorg10)   

R=np.loadtxt("lay11good.txt")
x11=R[:,0]
y11=R[:,1]
error11=R[:,2]
error11=sort_low(error11)   


R=np.loadtxt("lay6good.txt")
x6=R[:,0]
y6=R[:,1]
error6=R[:,2]
error6=sort_low(error6)   



R=np.loadtxt("lay4good.txt")
x4=R[:,0]
y4=R[:,1]
error4=R[:,2]
error4=sort_low(error4)   



R=np.loadtxt("lay8good.txt")
x8=R[:,0]
y8=R[:,1]
error8=R[:,2]
error8=sort_low(error8)   

R=np.loadtxt("lay9good.txt")
x9=R[:,0]
y9=R[:,1]
error9=R[:,2]
error9=sort_low(error9)   



R=np.loadtxt("lay8rand.txt")
x8r=R[:,0]
y8r=R[:,1]
error8r=R[:,2]
error8r=sort_low(error8r)   


R=np.loadtxt("lay12good.txt")
x12=R[:,0]
y12=R[:,1]
error12=R[:,2]
error12=sort_low(error12)   


R=np.loadtxt("lay13good.txt")
x13=R[:,0]
y13=R[:,1]
error13=R[:,2]
error13=sort_low(error13)   


#fig=plt.figure(figsize=(5,8))


#plt.loglog( errorr10, '4', color = '#e3360b', label='lay=10, random')
plt.loglog( error8, 'o', color = '#72cfbb', label='lay=8, good')
#plt.loglog( error9, '3', color = '#cf729d', label='lay=9, good')
plt.loglog( errorg10, '>', color = '#0b8de3', label='lay=10, good')
plt.loglog( error11, '+', color = '#e3570b', label='lay=11, good')
plt.loglog( error12, '2', color = '#729fcf', label='lay=12, good')
plt.loglog( error13, '4', color = '#96068a', label='lay=13, good')

#plt.loglog( error6, 'x', color = '#e30b69', label='lay=6')

#plt.loglog(  error4, '2', color = '#729fcf', label='lay=4')


#plt.loglog( error8r, '2', color = '#cf729d', label='lay=8, random')

plt.title('brickwall circuit')
plt.ylabel(r'$\delta$ E')
plt.xlabel(r'$n$')
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('qmps-plot.pdf')
plt.clf()
