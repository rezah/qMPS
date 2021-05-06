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



R=np.loadtxt("corrspinD16.txt")
x=R[:,0]
y=abs(R[:,1])

R=np.loadtxt("corrparticleD16.txt")
xN=R[:,0]
yN=abs(R[:,1])



R=np.loadtxt("corrspinQ4.txt")
xU=R[:,0]
yU=abs(R[:,1])

R=np.loadtxt("corrparticleQ4.txt")
xUU=R[:,0]
yUU=abs(R[:,1])







y_rand=np.arange(1, 4.0e1) 
#fig=plt.figure(figsize=(5,8))

# plt.loglog( y_rand, yfit(y_rand)  , '--', color = '#0b8de3' )
# plt.loglog(y_rand,yfitN(y_rand),'-.', color = '#e30b69')
# plt.loglog(y_rand,yfitU(y_rand) ,'--',  color = '#cf729d' )
# plt.loglog(y_rand,yfitUU(y_rand) ,'-.',  color = '#9820e3' )
#plt.loglog(y_rand,yfitqmpsbq5(y_rand) ,'-',  color = '#08c25f' )


plt.loglog( x, y, '>', color = '#0b8de3', label='spin, D=16')
plt.loglog( xN, yN, '4', color = '#0b8de3', label='charge, D=16')
plt.loglog( xU, yU, 'o', color = '#9820e3', label='spin, Q=4')
plt.loglog( xUU, yUU, 'P', color = '#08c25f', label='charge, Q=4')


#plt.loglog( xh, yh, 'H', color = '#9820e3', label='spin, h')
#plt.loglog( xhh, yhh, 's', color = '#9820e3', label='charge, h')

#plt.loglog( yp, errorp, 'o', color = '#e30b69', label='ladder')
#plt.loglog( yqmpsb, errorqmpsb,'s', color = '#cf729d', label='qmps-bk, q=4')
#plt.loglog( yqmpsp, errorqmpsp,'P', color = '#9820e3', label='qmps-ld, q=4')
#plt.loglog( yqmpsbq5, errorqmpsbq5,'H', color = '#08c25f', label='qmps-bk, q=5')


#plt.title('qmps')
plt.ylabel(r'$C(r)$')
plt.xlabel(r'$r$')
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('corr.pdf')
plt.clf()
