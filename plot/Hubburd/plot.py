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



R=np.loadtxt("corrspinD32.txt")
x32=R[:,0]
y32=abs(R[:,1])

R=np.loadtxt("corrspinD16.txt")
x=R[:,0]
y=abs(R[:,1])

# R=np.loadtxt("corrparticleD16.txt")
# xN=R[:,0]
# yN=abs(R[:,1])


R=np.loadtxt("corrspinD16R.txt")
xr=R[:,0]
yr=abs(R[:,1])

R=np.loadtxt("corrXD16R.txt")
xxr=R[:,0]
yxr=abs(R[:,1])

R=np.loadtxt("corrYD16R.txt")
xyr=R[:,0]
yyr=abs(R[:,1])




R=np.loadtxt("corrspinD32R.txt")
xr32=R[:,0]
yr32=abs(R[:,1])

R=np.loadtxt("corrXD32R.txt")
xxr32=R[:,0]
yxr32=abs(R[:,1])

R=np.loadtxt("corrYD32R.txt")
xyr32=R[:,0]
yyr32=abs(R[:,1])




R=np.loadtxt("corrspinq4.txt")
xq4=R[:,0]
yq4=abs(R[:,1])

R=np.loadtxt("corrXq4.txt")
xxq4=R[:,0]
yxq4=abs(R[:,1])

R=np.loadtxt("corrYq4.txt")
xyq4=R[:,0]
yyq4=abs(R[:,1])



R=np.loadtxt("corrspinQ4U4.txt")
xq4U=R[:,0]
yq4U=abs(R[:,1])

R=np.loadtxt("corrXQ4U4.txt")
xxq4U=R[:,0]
yxq4U=abs(R[:,1])

R=np.loadtxt("corrYQ4U4.txt")
xyq4U=R[:,0]
yyq4U=abs(R[:,1])



y_rand=np.arange(1, 4.0e1) 
fig=plt.figure(figsize=(8,7))

# plt.loglog( y_rand, yfit(y_rand)  , '--', color = '#0b8de3' )
# plt.loglog(y_rand,yfitN(y_rand),'-.', color = '#e30b69')
# plt.loglog(y_rand,yfitU(y_rand) ,'--',  color = '#cf729d' )
# plt.loglog(y_rand,yfitUU(y_rand) ,'-.',  color = '#9820e3' )
#plt.loglog(y_rand,yfitqmpsbq5(y_rand) ,'-',  color = '#08c25f' )


plt.loglog( x, y, '--', color = '#0b8de3', label='daoheng, D=16')
plt.loglog( x32, y32, '-', color = '#0b8de3', label='daoheng, D=32')
#plt.loglog( xN, yN, '4', color = '#0b8de3', label='charge, D=16')
#plt.loglog( xU4, yU4, 'o', color = '#9820e3', label='spin, Q=4')

plt.loglog( xq4U, yq4U, '^', color = '#cf729d',markersize=12, label='Z, q=4')
plt.loglog( xxq4U, yxq4U, '>', color = '#cf729d',markersize=12, label='X, q=4')
plt.loglog( xyq4U, yyq4U, '<', color = '#cf729d',markersize=12, label='Y, q=4')

plt.loglog( xq4, yq4, 'o', color = '#08c25f',markersize=10, label='Z, q=4')
plt.loglog( xxq4, yxq4, 'x', color = '#08c25f',markersize=10, label='X, q=4')
plt.loglog( xyq4, yyq4, '+', color = '#08c25f',markersize=10, label='Y, q=4')


#plt.loglog( xr, yr, '-.', color = '#08c25f',markersize=15, label='Z, D=16')
#plt.loglog( xxr, yxr, '-.', color = '#cf729d',markersize=15, label='X, D=16')
#plt.loglog( xyr, yyr, '-.', color = '#c20856',markersize=15, label='Y, D=16')

#plt.loglog( xr32, yr32, '-', color = '#4d32e6', label='Z, vMPS')
#plt.loglog( xxr32, yxr32, '-', color = '#096beb', label='X, vMPS')
#plt.loglog( xyr32, yyr32, '-', color = '#e30b69', label='Y, vMPS')


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

plt.legend(frameon=False)
plt.legend(loc='upper right',prop={'size': 16})

plt.grid(True)
plt.savefig('corr.pdf')
plt.clf()
