import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


key = ['200', '100', '50', '30', '25']
clr = ['b', 'r', 'k', 'y', 'g']
count = 0
"""
plt.figure(1, figsize=(8, 6))
for fn in key:
    s1 = '/dataspace/sandeep/Bispectrum_data/Bispectrum_value/'
    s2 = 'Analysis_Bispectrum_512_%s.txt' % fn

    name = s1+s2
    data = ascii.read(name, guess=False, delimiter='\t')
    Bis = data['Bis']

    i = data['i']
    j = data['j']
    k = data['k']
    Bis_1 = []
    l3 = []

    for ii in xrange(len(Bis)):
        if j[ii] == k[ii] == i[ii]:
            Bis_1.append(Bis[ii])
            l3.append(i[ii])
    Bis_1 = np.asarray(Bis_1)
    plt.plot(l3, abs(Bis_1), '-', color=clr[count], linewidth=2, label=fn)
    count += 1

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlabel(r"$l$", fontsize=18)
plt.ylabel(r"$B_{lll}$", fontsize=18)

count1 = 0
plt.figure(2, figsize=(8, 6))
for fn in key:
    s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_%sK_test/'%(fn)
    s2 = 'Analysis_%sKBin_Bispectrum_512_%s.txt' % (fn, fn)

    name = s1+s2
    data = ascii.read(name, guess=False, delimiter='\t')
    Bis = data['Bis']

    i = data['i']
    j = data['j']
    k = data['k']
    count = data['count']
    Bis_even = []
    Bis_1 = []
    l3 = []
    I3 = []
    I2 = []
    I1 = []

    for ii in xrange(len(Bis)):
        if count[ii] != 0:
            Bis_even.append(Bis[ii]/count[ii])
            I3.append(i[ii])
            I2.append(j[ii])
            I1.append(k[ii])

            if j[ii] == k[ii] == i[ii]:
                Bis_1.append(Bis[ii]/count[ii])
                l3.append(i[ii])

    Bis1 = np.asarray(Bis_even)
    Bis2 = np.asarray(Bis_1)

    plt.plot(l3, abs(Bis2), '-', color=clr[count1], linewidth=2, label=fn)
    count1 += 1

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlabel(r"$l$", fontsize=18)
plt.ylabel(r"$B_{lll}$", fontsize=18)
"""


Indx = 0
Indy = 0
lmax = 251

Actual = np.zeros((5, lmax), dtype=np.float32)
Mean = np.zeros((5, lmax), dtype=np.float32)
Std_Dev = np.zeros((5, lmax), dtype=np.float32)

for fn in key:
    esti_cl = np.zeros((1000, lmax), dtype=np.float32)
    s1 = "/dataspace/sandeep/Bispectrum_data"
    s2 = "/Gaussian_%sK_test/haslam_%sK_cl.txt"%(fn, fn)
    name = s1+s2
    cl = np.genfromtxt(name)
    Actual[count, :] = cl
    for i in xrange(0, 1000):
        s1 = "/dataspace/sandeep/Bispectrum_data"
        s2 = "/Gaussian_%sK_test/Gaussian_%sK_cl/haslam_%sKgaussMap_cl_%d.txt" % (fn, fn, fn, i)
        filename = s1+s2
        Map_cl = np.genfromtxt(filename)
        esti_cl[i, :] = Map_cl

    Mean[count, :] = np.mean(esti_cl, 0)
    Std_Dev[count, :] = np.std(esti_cl, 0)
    count += 1


l = np.arange(lmax)
plt.figure(1, figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)

ax1 = plt.subplot(gs[0, 0])
ax1.fill_between(l, l*(l+1)*(Mean[0,:]-Std_Dev[0, :]), l*(l+1)*(Mean[0, :]+Std_Dev[0, :]), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
ax1.plot(l, l*(l+1)*Mean[0, :], '-', color='crimson', linewidth=2, label='mean Cl')
ax1.plot(l, l*(l+1)*Actual[0, :], '-', color='orange', linewidth=2, label='original Cl')
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_title('200K')


plt.tight_layout()
plt.grid(which='both')
plt.legend(loc=3)
plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)


ax1 = plt.subplot(gs[0, 1])
ax1.fill_between(l, l*(l+1)*(Mean[1, :]-Std_Dev[1, :]), l*(l+1)*(Mean[1, :]+Std_Dev[1, :]), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
ax1.plot(l, l*(l+1)*Mean[1, :], '-', color='crimson', linewidth=2, label='mean Cl')
ax1.plot(l, l*(l+1)*Actual[1, :], '-', color='orange', linewidth=2, label='original Cl')
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_title('100K')

plt.grid(which='both')
plt.legend(loc=3)
plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)

plt.tight_layout()
ax1 = plt.subplot(gs[1, 0])
ax1.fill_between(l, l*(l+1)*(Mean[2, :]-Std_Dev[2, :]), l*(l+1)*(Mean[2, :]+Std_Dev[2, :]), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
ax1.plot(l, l*(l+1)*Mean[2, :], '-', color='crimson', linewidth=2, label='mean Cl')
ax1.plot(l, l*(l+1)*Actual[2, :], '-', color='orange', linewidth=2, label='original Cl')
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_title('50K')


plt.grid(which='both')
plt.legend(loc=3)
plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)

plt.tight_layout()

ax1 = plt.subplot(gs[1, 1])
ax1.fill_between(l, l*(l+1)*(Mean[3,:]-Std_Dev[3, :]), l*(l+1)*(Mean[3, :]+Std_Dev[3, :]), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
ax1.plot(l, l*(l+1)*Mean[3, :], '-', color='crimson', linewidth=2, label='mean Cl')
ax1.plot(l, l*(l+1)*Actual[3, :], '-', color='orange', linewidth=2, label='original Cl')
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_title('30K')


plt.grid(which='both')
plt.legend(loc=3)
plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)

plt.tight_layout()

plt.savefig("/dataspace/sandeep/Bispectrum_data/1000Realization_sim.eps", dpi=100)
plt.figure(2, figsize=(8, 6))
plt.fill_between(l, l*(l+1)*(Mean[4,:]-Std_Dev[4, :]), l*(l+1)*(Mean[4, :]+Std_Dev[4, :]), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
plt.plot(l, l*(l+1)*Mean[4, :], '-', color='crimson', linewidth=2, label='mean Cl')
plt.plot(l, l*(l+1)*Actual[4, :], '-', color='orange', linewidth=2, label='original Cl')
plt.yscale("log")
plt.xscale("log")
plt.title('25K')

plt.grid(which='both')
plt.legend(loc=3)
plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)

plt.tight_layout()

plt.savefig("/dataspace/sandeep/Bispectrum_data/1000_25K_Realization_sim.eps", dpi=100)



plt.show()