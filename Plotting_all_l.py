import matplotlib.pyplot as plt
import numpy as np
import astropy.io.ascii as ascii
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

lmax = 250

s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_50K_test/'
s2 = 'Analysis_50KBin_Bispectrum_512_50.txt'

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
I3 = np.asarray(I3)
I2 = np.asarray(I2)
I1 = np.asarray(I1)

l3 = np.asarray(l3)

print len(l3), len(Bis1), len(Bis2)

esti_bis = np.zeros((1000, len(Bis1)), dtype=np.float64)
esti_bis_1 = np.zeros((1000, len(Bis2)), dtype=np.float64)

for ii in xrange(0, 1000):

    s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_50K_test/Gaussian_Bin_Bispectrum/'
    s2 = 'BinnedBispectrum_GaussianMaps_512_50k_%d.txt' % ii

    name = s1+s2
    data = ascii.read(name, guess=False, delimiter='\t')
    Gauss_Bis = data['Bis']
    I = data['i']
    J = data['j']
    K = data['k']
    count = data['count']
    Bis_even = []
    Bis_1 = []
    for nn in xrange(len(Gauss_Bis)):
        if count[nn] != 0:
            Bis_even.append(Gauss_Bis[nn]/count[nn])
            if J[nn] == K[nn] == I[nn]:
                Bis_1.append(Gauss_Bis[nn]/count[nn])

    Gauss_Bis = np.asarray(Bis_even, dtype=np.float64)
    Gauss_Bis1 = np.asarray(Bis_1, dtype=np.float64)

    esti_bis[ii, :] = Gauss_Bis
    esti_bis_1[ii, :] = Gauss_Bis1


mean = np.mean(esti_bis, 0, dtype=np.float64)
std_dev = np.std(esti_bis, 0, dtype=np.float64)

mean1 = np.mean(esti_bis_1, 0, dtype=np.float64)
std_dev1 = np.std(esti_bis_1, 0, dtype=np.float64)

print Bis2
print mean1

nbin = 12
#x = 10 ** np.linspace(np.log10(2), np.log10(251), nbin)

x = 10**np.linspace(np.log10(11), np.log10(251), nbin)


def plot_data(count1):
    data = np.zeros((nbin-1, nbin-1), dtype=np.float64)
    for ii in xrange(len(I3)):
        if I3[ii] == count1:
            index, index1 = I2[ii], I1[ii]
            data[index, index1] = (Bis1[ii]-mean[ii])/std_dev[ii]
    return data


fig = plt.figure(1, figsize=(9, 8))

gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, 0])
im = ax1.imshow(plot_data(0), cmap='RdBu', origin='lower', interpolation='nearest')
ax1.set_xlabel(r'$l_{1}$', fontsize=14)
ax1.set_ylabel(r'$l_{2}$', fontsize=14)
ax1.set_title(r'$l_{3}\in [2], l_{3}=I_{1}$')
#ax1.set_title(r'$l_{3}\in [10, 13], l_{3}=I_{1}$')
plt.colorbar(im, fraction=0.046, pad=0.04)
ax2 = plt.subplot(gs[0, 1])
im = ax2.imshow(plot_data(1), cmap='RdBu', origin='lower', interpolation='nearest')
ax2.set_xlabel(r'$l_{1}$', fontsize=14)
ax2.set_ylabel(r'$l_{2}$', fontsize=14)
ax2.set_title(r'$l_{3}\in [3], l_{3}=I_{2}$')
#ax2.set_title(r'$l_{3}\in [14, 18], l_{3}=I_{2}$')
plt.colorbar(im, fraction=0.046, pad=0.04)

ax3 = plt.subplot(gs[0, 2])
im = ax3.imshow(plot_data(2), cmap='RdBu', origin='lower', interpolation='nearest')
ax3.set_xlabel(r'$l_{1}$', fontsize=14)
ax3.set_ylabel(r'$l_{2}$', fontsize=14)
ax3.set_title(r'$l_{3}\in [4, 6], l_{3}=I_{3}$')
#ax3.set_title(r'$l_{3}\in [19, 24], l_{3}=I_{3}$')
plt.colorbar(im, fraction=0.046, pad=0.04)

ax4 = plt.subplot(gs[1, 0])
im = ax4.imshow(plot_data(3), cmap='RdBu', origin='lower', interpolation='nearest')
ax4.set_xlabel(r'$l_{1}$', fontsize=14)
ax4.set_ylabel(r'$l_{2}$', fontsize=14)
ax4.set_title(r'$l_{3}\in [7, 10], l_{3}=I_{4}$')
#ax4.set_title(r'$l_{3}\in [25, 33], l_{3}=I_{4}$')
plt.colorbar(im, fraction=0.046, pad=0.04)


ax5 = plt.subplot(gs[1, 1])
im = ax5.imshow(plot_data(4), cmap='RdBu', origin='lower', interpolation='nearest')
ax5.set_xlabel(r'$l_{1}$', fontsize=14)
ax5.set_ylabel(r'$l_{2}$', fontsize=14)
ax5.set_title(r'$l_{3}\in [11, 16], l_{3}=I_{5}$')
#ax5.set_title(r'$l_{3}\in [34, 44], l_{3}=I_{5}$')
plt.colorbar(im, fraction=0.046, pad=0.04)


ax6 = plt.subplot(gs[1, 2])
im = ax6.imshow(plot_data(5), cmap='RdBu', origin='lower', interpolation='nearest')
ax6.set_xlabel(r'$l_{1}$', fontsize=14)
ax6.set_ylabel(r'$l_{2}$', fontsize=14)
ax6.set_title(r'$l_{3}\in [17, 26], l_{3}=I_{6}$')
#ax6.set_title(r'$l_{3}\in [45, 59], l_{3}=I_{6}$')
plt.colorbar(im, fraction=0.046, pad=0.04)


ax7 = plt.subplot(gs[2, 0])
im = ax7.imshow(plot_data(6), cmap='RdBu', origin='lower', interpolation='nearest')
ax7.set_xlabel(r'$l_{1}$', fontsize=14)
ax7.set_ylabel(r'$l_{2}$', fontsize=14)
ax7.set_title(r'$l_{3}\in [27, 42], l_{3}=I_{7}$')
#ax7.set_title(r'$l_{3}\in [60, 79], l_{3}=I_{7}$')
plt.colorbar(im, fraction=0.046, pad=0.04)


ax8 = plt.subplot(gs[2, 1])
im = ax8.imshow(plot_data(7), cmap='RdBu', origin='lower', interpolation='nearest')
ax8.set_xlabel(r'$l_{1}$', fontsize=14)
ax8.set_ylabel(r'$l_{2}$', fontsize=14)
ax8.set_title(r'$l_{3}\in [43, 66], l_{3}=I_{8}$')
#ax8.set_title(r'$l_{3}\in [80, 105], l_{3}=I_{8}$')
plt.colorbar(im, fraction=0.046, pad=0.04)


ax9 = plt.subplot(gs[2, 2])
im = ax9.imshow(plot_data(8), cmap='RdBu', origin='lower', interpolation='nearest')
ax9.set_xlabel(r'$l_{1}$', fontsize=14)
ax9.set_ylabel(r'$l_{2}$', fontsize=14)
ax9.set_title(r'$l_{3}\in [67, 103], l_{3}=I_{9}$')
#ax9.set_title(r'$l_{3}\in [106, 141], l_{3}=I_{9}$')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.tight_layout()  # Or equivalently,  "plt.tight_layout()"
plt.savefig("/dataspace/sandeep/Bispectrum_data/Gaussian_50K_test/plots/"
            "50K_2d_Binnedplots_data-mean_stdDev_1.eps", dpi=100)


plt.figure(2, figsize=(8, 6))
gs = gridspec.GridSpec(2, 2)
ax10 = plt.subplot(gs[0, 0])
im = ax10.imshow(plot_data(9), cmap='RdBu', origin='lower', interpolation='nearest')
ax10.set_xlabel(r'$l_{1}$', fontsize=14)
ax10.set_ylabel(r'$l_{2}$', fontsize=14)
ax10.set_title(r'$l_{3}\in [104, 160], l_{3}=I_{10}$')
#ax10.set_title(r'$l_{3}\in [142, 187], l_{3}=I_{10}$')
plt.colorbar(im, fraction=0.046, pad=0.04)

ax11 = plt.subplot(gs[0, 1])
im = ax11.imshow(plot_data(10), cmap='RdBu', origin='lower', interpolation='nearest')
ax11.set_xlabel(r'$l_{1}$', fontsize=14)
ax11.set_ylabel(r'$l_{2}$', fontsize=14)
ax11.set_title(r'$l_{3}\in [161, 249], l_{3}=I_{11}$')
#ax11.set_title(r'$l_{3}\in [188, 249], l_{3}=I_{11}$')
plt.colorbar(im, fraction=0.046, pad=0.04)

fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
plt.savefig("/dataspace/sandeep/Bispectrum_data/Gaussian_50K_test/plots/"
            "50K_2d_Binnedplots_data-mean_stdDev_2.eps", dpi=100)


plt.figure(3, figsize=(8, 6))
plt.plot(l3, Bis2, 'b-', linewidth=2, label='data')
plt.plot(l3, mean1, '-', color='orange', linewidth=2, label='mean')
plt.fill_between(l3, (mean1 - std_dev1),  (mean1 + std_dev1), alpha=0.5, edgecolor='c',
                 facecolor='paleturquoise')
plt.legend()
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlabel(r"$l$", fontsize=18)
plt.ylabel(r"$B_{lll}$", fontsize=18)
#plt.xlim(6,)
plt.yscale('symlog', linthreshy=0.001)
plt.savefig('/dataspace/sandeep/Bispectrum_data/Gaussian_50K_test/plots/Bispectrum_lll_1.eps', dpi=100)
plt.show()
