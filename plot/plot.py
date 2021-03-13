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


R=np.loadtxt("brick.txt")
x=R[:,0]
y=R[:,1]
error=R[:,2]
#error=sort_low(errorr10)   
logx = np.log(y)
logy = np.log(error)

yfit = lambda y: np.exp(poly(np.log(y)))
coeffs = np.polyfit(logx,logy,deg=1)
poly = np.poly1d(coeffs)


R=np.loadtxt("dmrg.txt")
xg=R[:,0]
yg=R[:,1]
errorg=R[:,2]
   


R=np.loadtxt("poll.txt")
xp=R[:,0]
yp=R[:,1]
errorp=R[:,2]
logx = np.log(yp)
logy = np.log(errorp)
yfitp = lambda yp: np.exp(poly2(np.log(yp)))
coeffs = np.polyfit(logx,logy,deg=1)
poly2 = np.poly1d(coeffs)





R=np.loadtxt("qmpsb.txt")
xqmpsb=R[:,0]
yqmpsb=R[:,2]
errorqmpsb=R[:,3]
logx = np.log(yqmpsb)
logy = np.log(errorqmpsb)
yfitqmpsb = lambda yqmpsb: np.exp(poly1(np.log(yqmpsb)))
coeffs = np.polyfit(logx,logy,deg=1)
poly1 = np.poly1d(coeffs)




R=np.loadtxt("qmpsp.txt")
xqmpsp=R[:,0]
yqmpsp=R[:,2]
errorqmpsp=R[:,3]
logx = np.log(yqmpsp)
logy = np.log(errorqmpsp)
yfitqmpsp = lambda yqmpsp: np.exp(poly3(np.log(yqmpsp)))
coeffs = np.polyfit(logx,logy,deg=1)
poly3 = np.poly1d(coeffs)



R=np.loadtxt("qmpsbq5.txt")
xqmpsbq5=R[:,0]
yqmpsbq5=R[:,2]
errorqmpsbq5=R[:,3]
logx = np.log(yqmpsbq5)
logy = np.log(errorqmpsbq5)
yfitqmpsbq5 = lambda qmpsbq5: np.exp(poly4(np.log(qmpsbq5)))
coeffs = np.polyfit(logx,logy,deg=1)
poly4 = np.poly1d(coeffs)





y_rand=np.arange(390, 5.0e4) 
#fig=plt.figure(figsize=(5,8))

plt.loglog( y_rand, yfit(y_rand)  , '--', color = '#0b8de3' )
plt.loglog(y_rand,yfitp(y_rand),'-.', color = '#e30b69')
plt.loglog(y_rand,yfitqmpsb(y_rand) ,'--',  color = '#cf729d' )
plt.loglog(y_rand,yfitqmpsp(y_rand) ,'-.',  color = '#9820e3' )
plt.loglog(y_rand,yfitqmpsbq5(y_rand) ,'-',  color = '#08c25f' )


plt.loglog( y, error, '>', color = '#0b8de3', label='brickwall')
plt.loglog( yg, errorg, '4', color = '#e3360b', label='mps')
plt.loglog( yp, errorp, 'o', color = '#e30b69', label='ladder')
plt.loglog( yqmpsb, errorqmpsb,'s', color = '#cf729d', label='qmps-bk, q=4')
plt.loglog( yqmpsp, errorqmpsp,'P', color = '#9820e3', label='qmps-ld, q=4')
plt.loglog( yqmpsbq5, errorqmpsbq5,'H', color = '#08c25f', label='qmps-bk, q=5')


#plt.title('qmps')
plt.ylabel(r'$\delta$ E')
plt.xlabel(r'$parameters$')
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.legend(frameon=False)
plt.legend(loc='upper right')

plt.grid(True)
plt.savefig('qmps-plot.pdf')
plt.clf()
