from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as LA
import scipy.optimize as opt

def func(x, a, b):
     return b*(x**a)

mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 5
mpl.rcParams['ytick.minor.width'] = 1




def sort_low(error):
 val_min=1.e8
 error_f=[]
 for i in range(len(error)-1):
   if error[i]>=error[i+1] and  error[i]<=val_min :
    error_f.append(error[i])
    val_min=error[i+1]*1.0
    pass

 return error_f




R=np.loadtxt("dmrgH.txt")
xg=R[:,0]
yg=R[:,2]
errorg=R[:,3]
logx = np.log(yg)
logy = np.log(errorg)
yfitDMRG = lambda yg: np.exp(polydmrg(np.log(yg)))
coeffs = np.polyfit(logx,logy,deg=1)
polydmrg = np.poly1d(coeffs)
print ( "dmrgH", coeffs)


ParamDMRG, pcov = opt.curve_fit(func, logx, logy);
print (*ParamDMRG)











R=np.loadtxt("dmrgF.txt")
xp=R[:,0]
yp=R[:,2]
errorp=R[:,3]
logx = np.log(yp)
logy = np.log(errorp)
yfitp = lambda yp: np.exp(poly2(np.log(yp)))
coeffs = np.polyfit(logx,logy,deg=1)
poly2 = np.poly1d(coeffs)
print ( "dmrgF", coeffs)


ParamDMRGF, pcov = opt.curve_fit(func, logx, logy);
print (*ParamDMRGF)




R=np.loadtxt("qmpsbq5.txt")
xqmpsbq5=R[:,0]
yqmpsbq5=R[:,2]
errorqmpsbq5=R[:,3]
logx = np.log(yqmpsbq5)
logy = np.log(errorqmpsbq5)
yfitqmpsbq5 = lambda qmpsbq5: np.exp(poly4(np.log(qmpsbq5)))
coeffs = np.polyfit(logx,logy,deg=1)
poly4 = np.poly1d(coeffs)
print ("qMPS-b", coeffs)

ParamQMPS, pcov = opt.curve_fit(func, logx, logy);
print (*ParamQMPS)




y_rand=np.arange(1500, 6.0e5) 
y_rand1=np.arange(1500, 6.0e5) 
y_rand2=np.arange(1500, 6.0e5) 

fig=plt.figure(figsize=(7,7))

plt.loglog( y_rand, yfitDMRG(y_rand) , '--', lw=4,color = '#a40000' )
plt.loglog(y_rand1,yfitp(y_rand1),'--', lw=4, color = '#a40000' )
plt.loglog(y_rand2,yfitqmpsbq5(y_rand2),'--', lw=4, color = '#a40000' )


#plt.loglog(y_rand, func(y_rand, *optimizedParameters), '-', lw=4,color = '#5c3566',label='Fit')
plt.loglog(y_rand, np.exp(func(np.log(y_rand), *ParamDMRG)), '-', lw=4,color = '#5c3566',label='DMRG')
plt.loglog(y_rand, np.exp(func(np.log(y_rand), *ParamDMRGF)), '-', lw=4,color = '#5c3566',label='DMRGF')
plt.loglog(y_rand, np.exp(func(np.log(y_rand), *ParamQMPS)), '-', lw=4,color = '#5c3566',label='QMPS')


plt.loglog( yg, errorg, 'H', markersize=15,color = '#204a87', label='DMRG-H')
plt.loglog( yp, errorp, 'o', markersize=15,color = '#e90ff5', label='DMRG-F')
plt.loglog( yqmpsbq5, errorqmpsbq5,'s',markersize=15, color = '#a40000', label=r'$qMPS, q=5$')



#plt.title('qmps')
plt.ylabel(r'$\delta$ E',fontsize=21)
plt.xlabel(r'$parameters$',fontsize=21)
#plt.axhline(0.00422,color='black', label='D=4')


#plt.xlim([600,40000])
#plt.ylim([1.e-7, 1.e-1])

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="lower left", prop={'size': 22})




plt.grid(True)
plt.savefig('qmpsB-plot.pdf')
plt.clf()
