import numpy as np
import healpy as hp
from astroML.plotting import hist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'
Haslam_512 = hp.fitsfunc.read_map(name)
Haslam_128 = hp.pixelfunc.ud_grade(Haslam_512, nside_out=128)

nside_f_est = 128
f_name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_binary_128/Mask_60K_binary_ns_128.fits'
ap_map = hp.fitsfunc.read_map(f_name, verbose=False)
npix = hp.nside2npix(nside_f_est)
haslam = Haslam_128 * ap_map

lmax = 256
nbin = 11
index = np.logspace(np.log10(10), np.log10(256), nbin, endpoint=True, dtype=np.int32)
ind = (index != 11)
index = index[ind]

bin_arr = [[] for i in range(11)]
bin_arr1 = np.zeros((nbin-1, 2), dtype=int)
for i in xrange(0, nbin):
    ini = index[i]
    if i + 1 < nbin:
        final = index[i + 1]
        if ini+5 > final:
            tmp = abs(final-ini)
            bin_arr1[i, 0] = ini
            bin_arr1[i, 1] = final+tmp
            bin_arr[i].append(range(ini, final+tmp))
            index[i+1] = final+tmp
        else:
            bin_arr1[i, 0] = ini
            bin_arr1[i, 1] = final
            bin_arr[i].append(range(ini, final))

print bin_arr1

alm_obs_actual = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)
skewr_actual = []
skewi_actual = []

kurti_actual = []
kurtr_actual = []


for i in xrange(0, nbin-1):

    alm = []
    temp = bin_arr[i][0]
    for l in temp:
        for m in xrange(-l, l+1):
            if m < 0.0:
                indx = hp.sphtfunc.Alm.getidx(lmax, l, -m)
                alm.append(np.conjugate(alm_obs_actual[indx]))
            else:
                indx = hp.sphtfunc.Alm.getidx(lmax, l, m)
                alm.append(alm_obs_actual[indx])

    alm = np.asarray(alm)
    index = (alm.real != 0.0)
    index1 = (alm.imag != 0.0)

    almr = alm.real[index]
    almi = alm.imag[index1]

    #real_alm, real_bin_edges = np.histogram(almr, bins='auto')
    #imag_alm, imag_bin_edges = np.histogram(almi, bins='auto')

    skewr_actual.append(stats.skew(almr, axis=0, bias=True))

    skewi_actual.append(stats.skew(almi, axis=0, bias=True))

    kurtr_actual.append(stats.kurtosis(almr, axis=0, fisher=False, bias=True))
    kurti_actual.append(stats.kurtosis(almi, axis=0, fisher=False, bias=True))


# ==============================================================================

hist_almr = np.zeros(nbin-1, dtype=np.float32)
hist_almi = np.zeros(nbin-1, dtype=np.float32)

knuth_almr = np.zeros(nbin-1, dtype=np.float32)
knuth_almi = np.zeros(nbin-1, dtype=np.float32)


skewr = np.zeros((1000, nbin-1), dtype=np.float32)
skewi = np.zeros((1000, nbin-1), dtype=np.float32)
kurtr = np.zeros((1000, nbin-1), dtype=np.float32)
kurti = np.zeros((1000, nbin-1), dtype=np.float32)


for fn in xrange(0, 1000):
    s1 = '/dataspace/sandeep/Bispectrum_data'
    s2 = '/Gaussian_60K_test/Gaussian_60K_Maps/haslam_60KgaussMap_%d.fits' % fn

    filename = s1+s2
    haslam = hp.fitsfunc.read_map(filename, verbose=False)*ap_map
    alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)

    for i in xrange(0, nbin-1):

        alm = []
        temp = bin_arr[i][0]
        for l in temp:
            for m in xrange(-l, l+1):
                if m < 0.0:
                    indx = hp.sphtfunc.Alm.getidx(lmax, l, -m)
                    alm.append(np.conjugate(alm_obs[indx]))
                else:
                    indx = hp.sphtfunc.Alm.getidx(lmax, l, m)
                    alm.append(alm_obs[indx])

        alm = np.asarray(alm)
        index = (alm.real != 0.0)
        index1 = (alm.imag != 0.0)

        almr = alm.real[index]
        almi = alm.imag[index1]

 #       real_alm, real_bin_edges = np.histogram(almr, bins='auto')
 #       imag_alm, imag_bin_edges = np.histogram(almi, bins='auto')

        skewr[fn, i] = stats.skew(almr, axis=0, bias=True)

        skewi[fn, i] = stats.skew(almi, axis=0, bias=True)

        kurtr[fn, i] = stats.kurtosis(almr, axis=0, fisher=False, bias=True)
        kurti[fn, i] = stats.kurtosis(almi, axis=0, fisher=False, bias=True)

skewr_mean = np.mean(skewr, 0, dtype=np.float32)
skewr_std_dev = np.std(skewr, 0, dtype=np.float32)
skewi_mean = np.mean(skewi, 0, dtype=np.float32)
skewi_std_dev = np.std(skewi, 0, dtype=np.float32)


kurtr_mean = np.mean(kurtr, 0, dtype=np.float32)
kurtr_std_dev = np.std(kurtr, 0, dtype=np.float32)
kurti_mean = np.mean(kurti, 0, dtype=np.float32)
kurti_std_dev = np.std(kurti, 0, dtype=np.float32)

I = np.arange(nbin-1)


fig = plt.figure(1, figsize=(10, 6))
gs = gridspec.GridSpec(2, 2)


ax1 = plt.subplot(gs[0, 0])
ax1.fill_between(I, (skewr_mean-skewr_std_dev), (skewr_mean+skewr_std_dev), alpha=0.8, edgecolor='k',
                 facecolor='blue')
ax1.fill_between(I, (skewr_mean-2.*skewr_std_dev), (skewr_mean + 2.*skewr_std_dev), alpha=0.2, edgecolor='k',
                 facecolor='green')

ax1.plot(I, (skewr_mean), '-', color='m', linewidth=2, label='mean')
ax1.plot(I, skewr_actual, '-', color='r', linewidth=2, label='Actual')


ax1.set_xlabel(r"$I$", fontsize=18)
ax1.set_ylabel(r"$S_{3}$", fontsize=18)
ax1.set_title("60K Real alm")
ax1.set_xlim(0, 10)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)


ax2 = plt.subplot(gs[0, 1])

ax2.fill_between(I, (skewi_mean-skewi_std_dev), (skewi_mean+skewi_std_dev), alpha=0.8, edgecolor='k',
                 facecolor='blue')
ax2.fill_between(I, (skewi_mean-2.*skewi_std_dev), (skewi_mean+2.*skewi_std_dev), alpha=0.2, edgecolor='k',
                 facecolor='green')

ax2.plot(I, (skewi_mean), '-', color='m', linewidth=2, label='mean')
ax2.plot(I, skewi_actual, '-', color='r', linewidth=2, label='Actual')

ax2.set_xlabel(r"$I$", fontsize=18)
ax2.set_ylabel(r"$S_{3}$", fontsize=18)
ax2.set_title("60K Imag alm")

ax2.set_xlim(0, 10)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)

plt.tight_layout()


ax3 = plt.subplot(gs[1, 0])

ax3.fill_between(I, (kurtr_mean-kurtr_std_dev), (kurtr_mean+kurtr_std_dev), alpha=0.8, edgecolor='k',
                 facecolor='blue')
ax3.fill_between(I, (kurtr_mean-2.*kurtr_std_dev), (kurtr_mean+2.*kurtr_std_dev), alpha=0.2, edgecolor='k',
                 facecolor='green')

ax3.plot(I, (kurtr_mean), '-', color='m', linewidth=2, label='mean')
ax3.plot(I, kurtr_actual, '-', color='r', linewidth=2, label='Actual')

#ax3.set_title("60K Real alm")
ax3.set_xlabel(r"$I$", fontsize=18)
ax3.set_ylabel(r"$S_{4}$", fontsize=18)
ax3.set_xlim(0, 10)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)

ax4 = plt.subplot(gs[1, 1])

ax4.fill_between(I, (kurti_mean-kurti_std_dev), (kurti_mean+kurti_std_dev), alpha=0.8, edgecolor='k',
                 facecolor='blue')
ax4.fill_between(I, (kurti_mean-2.*kurti_std_dev), (kurti_mean+2.*kurti_std_dev), alpha=0.2, edgecolor='k',
                 facecolor='green')

ax4.plot(I, (kurti_mean), '-', color='m', linewidth=2, label='mean')
ax4.plot(I, kurti_actual, '-', color='r', linewidth=2, label='Actual')

#ax4.set_title("60K Imag alm")
ax4.set_xlabel(r"$I$", fontsize=18)
ax4.set_ylabel(r"$S_{4}$", fontsize=18)
ax4.set_xlim(0, 10)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)

plt.tight_layout()

fig.savefig("/dataspace/sandeep/Bispectrum_data/Gaussian_60K_test/plots/bin_skewness_Kurtosis_60K.png", dpi=600)

plt.show()








"""

indX = 0
indY = 0
Count = 0

plt.figure(1, figsize=(17, 15))
gs = gridspec.GridSpec(4, 3)
for indX in xrange(0, 4):
    for indY in xrange(0, 3):
        if Count < 11:
            Count += 1
        else:
            break
        for i in xrange(Count-1, Count):
            alm = []
            temp = bin_arr[i][0]
            for l in temp:
                for m in xrange(-l, l+1):
                    if m < 0.0:
                        indx = hp.sphtfunc.Alm.getidx(lmax, l, -m)
                        alm.append(np.conjugate(alm_obs[indx]))
                    else:
                        indx = hp.sphtfunc.Alm.getidx(lmax, l, m)
                        alm.append(alm_obs[indx])

        alm = np.asarray(alm)

        index = (alm.real != 0.0)
        index1 = (alm.imag != 0.0)

        almr = alm.real[index]
        almi = alm.imag[index1]

        real_alm, real_bin_edges = np.histogram(almr, bins='auto')
        imag_alm, imag_bin_edges = np.histogram(almi, bins='auto')

        ax1 = plt.subplot(gs[indX, indY])
        h1 = hist(almr, bins='knuth', histtype='stepfilled', ec='k', fc='#AAAAAA', label='Knuth')
        ax1.plot(real_bin_edges[:-1], real_alm, "bo", ms=5, label='auto')

        #h1 = hist(almi, bins='knuth', histtype='stepfilled', ec='k', fc='skyblue', label='Knuth')
        #ax1.plot(imag_bin_edges[:-1], imag_alm, "ro", ms=5, label='auto')

        ax1.set_ylabel(r'$N$', fontsize='large', fontstyle='italic', weight='extra bold')
        ax1.set_title(r'25K Haslam ($I_{%d}$)' % (i+1))
        #ax1.set_xlabel(r'$Imag(a_{lm})$', fontsize='large', fontstyle='italic', weight='extra bold')
        ax1.set_xlabel(r'$Real(a_{lm})$', fontsize='large', fontstyle='italic', weight='extra bold')
        ax1.set_yscale("log")
        plt.minorticks_on()
        plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
        plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)
        plt.tight_layout()
        plt.legend()


#plt.savefig("/dataspace/sandeep/Bispectrum_data/Gaussian_25K_test/alm_25K_stat/GaussianMap_Bin_Imag_alm_log.eps",
#            dpi=100)
plt.savefig("/dataspace/sandeep/Bispectrum_data/Gaussian_25K_test/alm_25K_stat/ActualMap_Bin_real_alm_log.eps",
            dpi=100)
plt.show()
"""

#Voilin Plot
"""
test = np.array([skewr[:, 4], skewr[:, 5], skewr[:, 6], skewr[:, 7], skewr[:, 8], skewr[:, 9]])
test = test.T


pvalue = []
for i in xrange(4, 10):
    ind = (skewr[:, i] < skewr_actual[i])
    test1 = skewr[ind, i]
    print test
    print skewr_actual[i]
    P = (len(test1)+1)/100.
    pvalue.append(P)

fig = plt.figure(1, figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
violin_parts = ax1.violinplot(test, showmedians=True)
ax1.plot(1.0, skewr_actual[4], 'bo', ms=10, label="p-value-%0.4f" % pvalue[0])
ax1.plot(2.0, skewr_actual[5], 'mo', ms=10, label="p-value-%0.4f" % pvalue[1])
ax1.plot(3.0, skewr_actual[6], 'ro', ms=10, label="p-value-%0.4f" % pvalue[2])
ax1.plot(4.0, skewr_actual[7], 'co', ms=10, label="p-value-%0.4f" % pvalue[3])
ax1.plot(5.0, skewr_actual[8], 'yo', ms=10, label="p-value-%0.4f" % pvalue[4])
ax1.plot(6.0, skewr_actual[9], 'go', ms=10, label="p-value-%0.4f" % pvalue[5])

ax1.set_xticklabels(['', r'$l_{4}$', r'$l_{5}$', r'$l_{6}$', r'$l_{7}$', r'$l_{8}$', r'$l_{9}$'],
                     weight='extra bold', fontsize='x-large')

for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    vp = violin_parts[partname]
    vp.set_edgecolor('k')
    vp.set_linewidth(2)
for vp in violin_parts['bodies']:
    vp.set_facecolor('#C71585')
    vp.set_edgecolor('k')
    vp.set_linewidth(1)
    vp.set_alpha(0.7)
plt.legend()
plt.minorticks_on()
"""







