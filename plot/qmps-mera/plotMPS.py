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


R=np.loadtxt("AF4.txt")
x=list(R[:,0])
y=list(R[:,2])

R=np.loadtxt("AFF4.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x=x+xp
y=y+yp


R=np.loadtxt("AFFF4.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x=x+xp
y=y+yp


R=np.loadtxt("AFFFF4.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x=x+xp
y=y+yp

y=sort_low(y)


R=np.loadtxt("BF4.txt")
x1=list(R[:,0])
y1=list(R[:,2])

R=np.loadtxt("BFF4.txt")
xp1=list(R[:,0])
yp1=list(R[:,2])
x1=x1+xp1
y1=y1+yp1


y1=sort_low(y1)










R=np.loadtxt("AF10.txt")
x2=list(R[:,0])
y2=list(R[:,2])

R=np.loadtxt("AFF10.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x2=x2+xp
y2=y2+yp


R=np.loadtxt("AFFF10.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x2=x2+xp
y2=y2+yp

R=np.loadtxt("AFFFF10.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x2=x2+xp
y2=y2+yp


y2=sort_low(y2)







R=np.loadtxt("CF4.txt")
x4=list(R[:,0])
y4=list(R[:,2])

R=np.loadtxt("CFF4.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x4=x4+xp
y4=y4+yp


R=np.loadtxt("CFFF4.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x4=x4+xp
y4=y4+yp

R=np.loadtxt("CFFFF4.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x4=x4+xp
y4=y4+yp


y4=sort_low(y4)






R=np.loadtxt("BF10.txt")
x3=list(R[:,0])
y3=list(R[:,2])

R=np.loadtxt("BFF10.txt")
xp=list(R[:,0])
yp=list(R[:,2])
x3=x3+xp
y3=y3+yp

y3=sort_low(y3)





plt.figure(figsize=(8, 6))
plt.axhline(0.00008422,color='black', label=r'qMPS-b, $\tau=4$')

plt.loglog( y, '--', lw=4, color = '#e3360b', label=r'qMPS-m, $\tau=4$')
plt.loglog( y1, '--', lw=4, color = '#72cfbb', label=r'qMPS-m, $\tau=4$')
plt.loglog( y4, '--', lw=4, color = '#cf729d', label=r'qMPS-m, $\tau=4$')
plt.loglog( y2, '-', lw=4, color = '#c74294', label=r'qMPS-m, $\tau=10$')
plt.loglog( y3, '-', lw=4, color = '#e3570b', label=r'qMPS-m, $\tau=10$')
#plt.loglog(  error4, '--', lw=4, color = '#a40000', label=r'$ \tau=18,good$')
#plt.loglog( error8, 'o', color = '#72cfbb', label='lay=8')
#plt.loglog( xq4l24, errorq4l24, '2', color = '#cf729d', label='q=4, lay=24')




#plt.title('brickwall circuit')
plt.ylabel(r'$\delta$ E',fontsize=20)
plt.xlabel(r'$iterations$',fontsize=20)
#plt.axhline(0.00422,color='black', label='D=4')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc="lower left", prop={'size': 16})


plt.xlim([0.94,1.e5])
plt.ylim([2.0e-5, 30.e-1])

plt.grid(True)
plt.savefig('qmpsMERA.pdf')
plt.clf()
