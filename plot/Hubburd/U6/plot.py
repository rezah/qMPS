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

def log_f(x, y):
 logx = np.log(x)
 logy = np.log(y)
 yfit = lambda y: np.exp(poly(np.log(y)))
 coeffs = np.polyfit(logx,logy,deg=1)
 poly = np.poly1d(coeffs)
 return yfit


R=np.loadtxt("X.txt")
val_r=R[:,0]
c_r=R[:,1]



R=np.loadtxt("corrspinU6D16.txt")
x=R[:,0]
y=abs(R[:,1])

R=np.loadtxt("corrparticleU6D16.txt")
xp=R[:,0]
yp=abs(R[:,1])



R=np.loadtxt("corrXU6D16.txt")
xx=R[:,0]
yx=abs(R[:,1])


R=np.loadtxt("corrXU6D16.txt")
xy=R[:,0]
yy=abs(R[:,1])







R=np.loadtxt("corrspinU6.txt")
xc=R[:,0]
yc=abs(R[:,1])

R=np.loadtxt("corrparticleU6.txt")
xpc=R[:,0]
ypc=abs(R[:,1])



R=np.loadtxt("corrXU6.txt")
xxc=R[:,0]
yxc=abs(R[:,1])


R=np.loadtxt("corrXU6.txt")
xyc=R[:,0]
yyc=abs(R[:,1])



y_rand=np.arange(1, 4.0e1) 
fig=plt.figure(figsize=(8,7))



plt.loglog( val_r, c_r, '--', lw=2,color = '#0b8de3', label='daoheng')

plt.loglog( x, y, '>', color = '#0b8de3', markersize=15,label='spin')
plt.loglog( xc, yc, 'x', color = '#5c3566',markersize=15, label='spin')
plt.loglog( xy, yy, 's', color = '#08c25f',markersize=15, label='Y')
plt.loglog( xyc, yyc, '4', color = '#5c3566',markersize=15, label='Y')
plt.loglog( xxc, yxc, 'h', color = '#e90ff5',markersize=15, label='X')
plt.loglog( xx, yx, 'o', color = '#cf729d',markersize=15, label='X')


plt.loglog( xpc, ypc, '>', color = '#c4a000',markersize=15, label='charge')
plt.loglog( xp, yp, '<', color = '#a40000',markersize=15, label='charge')




#plt.loglog( xh, yh, 'H', color = '#9820e3', label='spin, h')
#plt.loglog( xhh, yhh, 's', color = '#9820e3', label='charge, h')

#plt.loglog( yp, errorp, 'o', color = '#e30b69', label='ladder')
#plt.loglog( yqmpsb, errorqmpsb,'s', color = '#cf729d', label='qmps-bk, q=4')
#plt.loglog( yqmpsp, errorqmpsp,'P', color = '#9820e3', label='qmps-ld, q=4')
#plt.loglog( yqmpsbq5, errorqmpsbq5,'H', color = '#08c25f', label='qmps-bk, q=5')


#plt.title('qmps')
plt.ylabel(r'$C(r)$',fontsize=21)
plt.xlabel(r'$r$',fontsize=21)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')



plt.legend(loc='lower left', prop={'size': 16})

plt.grid(True)
plt.savefig('corr.pdf')
plt.clf()
