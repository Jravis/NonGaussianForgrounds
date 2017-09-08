import matplotlib.pyplot as plt
import numpy as np
import astropy.io.ascii as ascii
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
lmax = 250


s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_50K_test/'
s2 = 'All_mode/Analysis_Bin_Bispectrum_128_50K.txt'


name = s1+s2
data = ascii.read(name, guess=False, delimiter='\t')
Bis = data['Bis']

i = data['I1']
j = data['I2']
k = data['I3']
Bis_even = []
Bis_1 = []
l3 = []
I3 = []
I2 = []
I1 = []

print "Do you have just even modes[Y/N]"
key = raw_input("")

if key == 'Y':
    count = data['count']
    for ii in xrange(len(Bis)):
        if count[ii] != 0:
            Bis_even.append(Bis[ii]/count[ii])
            I1.append(i[ii])
            I2.append(j[ii])
            I3.append(k[ii])

            if j[ii] == k[ii] == i[ii]:
                Bis_1.append(Bis[ii]/count[ii])
                l3.append(k[ii])

    Bis1 = np.asarray(Bis_even)
    Bis2 = np.asarray(Bis_1)
    I3 = np.asarray(I3)
    I2 = np.asarray(I2)
    I1 = np.asarray(I1)
    l3 = np.asarray(l3)
else:
    for ii in xrange(len(Bis)):
        if j[ii] == k[ii] == i[ii]:
            Bis_1.append(Bis[ii])
            l3.append(k[ii])

    Bis1 = np.asarray(Bis, dtype=np.float64)
    Bis2 = np.asarray(Bis_1, dtype=np.float64)
    I3 = np.asarray(k)
    I2 = np.asarray(j)
    I1 = np.asarray(i)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
esti_bis = np.zeros((1000, len(Bis1)), dtype=np.float64)
esti_bis_1 = np.zeros((1000, len(Bis2)), dtype=np.float64)


for ii in xrange(0, 1000):
    s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_50K_test/Gaussian_Bin_Bispectrum/'
    s2 = 'BinnedBispectrum_Bin_GaussianMaps_linCorr_128_50K_%d.txt' % ii

    name = s1+s2
    data = ascii.read(name, guess=False, delimiter='\t')
    Gauss_Bis = data['Bis']
    I = data['I1']
    J = data['I2']
    K = data['I3']
    Bis_even = []
    Bis_1 = []
    if key == 'Y':
        count = data['count']
        for nn in xrange(len(Gauss_Bis)):
            if count[nn] != 0:
                Bis_even.append(Gauss_Bis[nn]/count[nn])
                if J[nn] == K[nn] == I[nn]:
                    Bis_1.append(Gauss_Bis[nn]/count[nn])
        Gauss_Bis = np.asarray(Bis_even, dtype=np.float64)
        Gauss_Bis1 = np.asarray(Bis_1, dtype=np.float64)

    else:
        for nn in xrange(len(Gauss_Bis)):
            if J[nn] == K[nn] == I[nn]:
                Bis_1.append(Gauss_Bis[nn])

        Gauss_Bis = np.asarray(Gauss_Bis, dtype=np.float64)
        Gauss_Bis1 = np.asarray(Bis_1, dtype=np.float64)

    esti_bis[ii, :] = Gauss_Bis
    esti_bis_1[ii, :] = Gauss_Bis1


mean = np.mean(esti_bis, 0, dtype=np.float64)
std_dev = np.std(esti_bis, 0, dtype=np.float64)

mean1 = np.mean(esti_bis_1, 0, dtype=np.float64)
std_dev1 = np.std(esti_bis_1, 0, dtype=np.float64)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

lmax = 256
nbin = 11
index = np.logspace(np.log10(10), np.log10(256), nbin, endpoint=True, dtype=np.int32)
ind = (index != 11)
index = index[ind]
print index


bin_arr = np.zeros((nbin - 1, 2), dtype=np.int32)

for i in xrange(0, nbin):
    ini = index[i]
    if i + 1 < nbin:
        final = index[i + 1]
        bin_arr[i, 0] = ini
        bin_arr[i, 1] = final - 1


cmap = plt.cm.RdBu
cmap.set_bad(color='grey')


def plot_data(count1):

    data = np.zeros((nbin-1, nbin-1), dtype=np.float64)
    for ii in xrange(len(I3)):
        if I3[ii] == count1:
            index, index1 = I2[ii], I1[ii]
    #        temp = (Bis1[ii]-mean[ii])/std_dev[ii]
            #if -2.0 > temp or temp > 2.0:
            data[index, index1] = (Bis1[ii]-mean[ii])/std_dev[ii]
    #data = np.ma.masked_where(data == 0.0, data)
    return data, [bin_arr[count1, 0], bin_arr[count1, 1]]


Indx = 0
Indxy = 0
nn = 0

fig = plt.figure(1, figsize=(12, 15))
gs = gridspec.GridSpec(4, 4, hspace=0.5, wspace=0.6)
#gs.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
for Indx in xrange(0, 3):
    for Indy in xrange(0, 3):
        a, b = plot_data(nn)
        ax1 = plt.subplot(gs[Indx, Indy])
        im = ax1.imshow(a, cmap=cmap, origin='lower', interpolation='none')
        ax1.set_xlabel(r'$I_{1}$', fontsize=14)
        ax1.set_ylabel(r'$I_{2}$', fontsize=14)
        ax1.set_title(r'$I_{3}\in [%d, %d]$' % (b[0], b[1]))
        ax1.set_aspect('equal')
        plt.colorbar(im,  spacing='proportional',  fraction=0.046, pad=0.04)
        nn += 1
a, b = plot_data(nn)
ax1 = plt.subplot(gs[3, 0])
im = ax1.imshow(a, cmap=cmap, origin='lower', interpolation='none')
ax1.set_xlabel(r'$I_{1}$', fontsize=14)
ax1.set_ylabel(r'$I_{2}$', fontsize=14)
ax1.set_title(r'$I_{3}\in [%d, %d]$' % (b[0], b[1]))
ax1.set_aspect('equal')
plt.colorbar(im,  spacing='proportional',  fraction=0.046, pad=0.04)
#plt.tight_layout()  # Or equivalently,  "plt.tight_layout()"
plt.savefig("/dataspace/sandeep/Bispectrum_data/Gaussian_50K_test/plots/Allmode_2dplot_Bin_Bispectrum_nbin-11.png", dpi=600)


plt.figure(3, figsize=(8, 6))
plt.plot(l3, Bis2, 'b-', linewidth=2, label='data')
plt.plot(l3, mean1, '-', color='orange', linewidth=2, label='mean')
plt.fill_between(l3, (mean1 - std_dev1),  (mean1 + std_dev1), alpha=0.5, edgecolor='c',
                 facecolor='paleturquoise')
plt.legend()
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlabel(r"$I$", fontsize=18)
plt.ylabel(r"$B_{III}$", fontsize=18)
#plt.xlim(6,)
plt.yscale('symlog', linthreshy=0.001)
plt.savefig('/dataspace/sandeep/Bispectrum_data/Gaussian_50K_test/plots/Allmode_Bin_Bispectrum_lll_nbin-11.png', dpi=600)
plt.show()
