from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib as mpl


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
logx = np.log(yg)
logy = np.log(errorg)
yfitDMRG = lambda yg: np.exp(polydmrg(np.log(yg)))
coeffs = np.polyfit(logx,logy,deg=1)
polydmrg = np.poly1d(coeffs)



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




# R=np.loadtxt("qmpsp.txt")
# xqmpsp=R[:,0]
# yqmpsp=R[:,2]
# errorqmpsp=R[:,3]
# logx = np.log(yqmpsp)
# logy = np.log(errorqmpsp)
# yfitqmpsp = lambda yqmpsp: np.exp(poly3(np.log(yqmpsp)))
# coeffs = np.polyfit(logx,logy,deg=1)
# poly3 = np.poly1d(coeffs)



R=np.loadtxt("qmeraQ2.txt")
xqmpsbq5=R[:,0]
yqmpsbq5=R[:,2]
errorqmpsbq5=R[:,3]
logx = np.log(yqmpsbq5)
logy = np.log(errorqmpsbq5)
yfitqmpsbq5 = lambda qmpsbq5: np.exp(poly4(np.log(qmpsbq5)))
coeffs = np.polyfit(logx,logy,deg=1)
poly4 = np.poly1d(coeffs)
print ("qmera-q2", coeffs)



R=np.loadtxt("qmeraQ3.txt")
xqmpsbq8=R[:,0]
yqmpsbq8=R[:,2]
errorqmpsbq8=R[:,3]
logx = np.log(yqmpsbq8)
logy = np.log(errorqmpsbq8)
yfitqmpsbq8 = lambda qmpsbq8: np.exp(poly8(np.log(qmpsbq8)))
coeffs = np.polyfit(logx,logy,deg=1)
poly8 = np.poly1d(coeffs)
print ("qmera-q3", coeffs)


R=np.loadtxt("qmeraQ4.txt")
#print (R)
xmera=R[:,0]
ymera=R[:,2]
errormera=R[:,3]
logx = np.log(ymera)
logy = np.log(errormera)
yfitmera = lambda qmpsmera: np.exp(poly9(np.log(qmpsmera)))
coeffs = np.polyfit(logx,logy,deg=1)
poly9 = np.poly1d(coeffs)
print ("qmera-q4", coeffs)


y_rand=np.arange(390, 5.0e4) 
y_rand1=np.arange(1400, 5.0e4) 
y_rand2=np.arange(1500, 5.0e4) 

fig=plt.figure(figsize=(7,7))

plt.loglog(y_rand1,yfitqmpsb(y_rand1) ,'--',lw=4,  color = '#cf729d' )
plt.loglog(y_rand2,yfitqmpsbq8(y_rand2) ,'-.',lw=4,  color = '#a40000' )
plt.loglog(y_rand2,yfitqmpsbq5(y_rand2) ,'--',lw=4,  color = '#f57900' )



plt.loglog( yqmpsb, errorqmpsb,'s', markersize=14,color = '#cf729d', label=r'$qMPS, n_q=4$')
plt.loglog( yqmpsbq5, errorqmpsbq5,'o', markersize=14,color = '#f57900', label=r'$qMERA, q=2$')
plt.loglog( yqmpsbq8, errorqmpsbq8,'D', markersize=14,color = '#a40000', label=r'$qMERA, q=3$')
plt.loglog( ymera, errormera,'v', markersize=14,color = '#e3360b', label=r'$qMERA, q=4$')

plt.loglog( yg, errorg, 'H', markersize=14,color = '#204a87', label='MPS-DMRG')


#plt.title('qmps')
plt.ylabel(r'$\delta$ E',fontsize=21)
plt.xlabel(r'$parameters$',fontsize=21)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.xlim([1000,40000])
plt.ylim([1.e-5, 1.e-1])

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="upper right", prop={'size': 20})


plt.grid(True)
plt.savefig('qmera-plot.pdf')
plt.clf()
