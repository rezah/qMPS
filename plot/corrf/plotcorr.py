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


R=np.loadtxt("corrZDMRG.txt")
x=R[:,2]
y=R[:,3]


R=np.loadtxt("corrZDMRGD16.txt")
xD16=R[:,2]
yD16=R[:,3]
errorD16=[    abs((y[i]-yD16[i])/ y[i])     for i in range(len(y))]

print (errorD16)

R=np.loadtxt("corrZDMRGD32.txt")
xD32=R[:,2]
yD32=R[:,3]
errorD32=[    abs((y[i]-yD32[i])/ y[i])     for i in range(len(y))]



R=np.loadtxt("corrZPQMPSq5l12.txt")
xq5p=R[:,2]
yq5p=R[:,3]
errorq5p=[    abs((y[i]-yq5p[i])/ y[i])     for i in range(len(y))]



R=np.loadtxt("corrZqMPSq5lay14.txt")
xq5b=R[:,2]
yq5b=R[:,3]
errorq5b=[    abs((y[i]-yq5b[i])/ y[i])     for i in range(len(y))]

R=np.loadtxt("corrZQMPSq4l20.txt")
xq4b=R[:,2]
yq4b=R[:,3]
errorq4b=[    abs((y[i]-yq4b[i])/ y[i])     for i in range(len(y))]





fig=plt.figure(figsize=(8,6))



#plt.plot( x, errorD16, '--', color = '#e30b69',markersize=10, label=r'$D=16$')

plt.plot( x, errorq4b, '3', color = '#e30b69',markersize=15, label=r'$qMPS, q=4$')
#plt.loglog( x, errorq5p,'3', color = '#cf729d',markersize=15, label=r'$qMPS, n_q=5$')
#plt.loglog( x, errorq5b,'2',markersize=15, color = '#c74294', label=r'$qMPS, n_q=5$')
plt.loglog( x, errorD32, '2', markersize=15,color = '#204a87', label='MPS, D=32')








plt.yscale("log")

#plt.title('qmps')
plt.ylabel(r'$\delta$ C',fontsize=16)
plt.xlabel(r'$r$',fontsize=16)
#plt.axhline(0.00422,color='black', label='D=4')
#plt.axhline(0.000143, color='black', label='D=8')
#plt.axhline(0.000355, color='black', label='D=16')

plt.xlim([4,26])
plt.ylim([ 1.e-1, 5.e-6])


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc="upper right", prop={'size': 20})


plt.grid(True)
plt.savefig('qmps-p.pdf')
plt.clf()
