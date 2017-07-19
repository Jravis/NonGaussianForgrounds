import numpy as np
import healpy as hp
from astroML.plotting import hist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'
Haslam_512 = hp.fitsfunc.read_map(name)

nside_f_est = 512

f_name = "/dataspace/sandeep/Bispectrum_data/Input_Maps/ApodizeBinaryMask_%s_%0.1fdeg_apodi.fits" % ('30K', 2.0)
ap_map = hp.fitsfunc.read_map(f_name, verbose=False)
npix = hp.nside2npix(nside_f_est)
haslam = Haslam_512 * ap_map

lmax = 251
nbin = 12

#index = 10 ** np.linspace(np.log10(2), np.log10(251), nbin)  # logrithmic bins
index = 10 ** np.linspace(np.log10(11), np.log10(251), nbin)


bin_arr = [[] for i in range(12)]
bin_arr1 = np.zeros((nbin-1, 2), dtype=int)
for i in xrange(0, nbin):
    ini = int(index[i])
    if i + 1 < nbin:
        final = int(index[i + 1])
        bin_arr1[i, 0] = ini
        bin_arr1[i, 1] = final
        bin_arr[i].append(range(ini, final))

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

#    ar, critr, signir = stats.anderson(almr, dist='norm')
#    ai, criti, signii = stats.anderson(almi, dist='norm')

    skewr_actual.append(stats.skew(almr, axis=0, bias=True))

    skewi_actual.append(stats.skew(almi, axis=0, bias=True))

    kurtr_actual.append(stats.kurtosis(almr, axis=0, fisher=True, bias=True))
    kurti_actual.append(stats.kurtosis(almi, axis=0, fisher=True, bias=True))


# ==============================================================================

hist_almr = np.zeros(nbin-1, dtype=np.float32)
hist_almi = np.zeros(nbin-1, dtype=np.float32)

knuth_almr = np.zeros(nbin-1, dtype=np.float32)
knuth_almi = np.zeros(nbin-1, dtype=np.float32)


skewr = np.zeros((100, nbin-1), dtype=np.float32)
skewi = np.zeros((100, nbin-1), dtype=np.float32)
kurtr = np.zeros((100, nbin-1), dtype=np.float32)
kurti = np.zeros((100, nbin-1), dtype=np.float32)


for fn in xrange(0, 100):
    s1 = '/dataspace/sandeep/Bispectrum_data'
    s2 = '/Gaussian_30K_test/Gaussian_30K_Maps/haslam_30KgaussMap_%d.fits' % fn
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

        kurtr[fn, i] = stats.kurtosis(almr, axis=0, fisher=True, bias=True)
        kurti[fn, i] = stats.kurtosis(almi, axis=0, fisher=True, bias=True)

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

gs = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs[0, 0])
ax1.violinplot(skewr[:, 9], showmedians=True)
ax1.plot(1.0, skewr_actual[9], 'bo', ms=10)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)

ax2 = plt.subplot(gs[0, 1])
ax2.violinplot(kurtr[:, 9], showmedians=True)
ax2.plot(1.0, kurtr_actual[9], 'bo', ms=10)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)

plt.tight_layout()


"""
fig = plt.figure(1, figsize=(10, 6))

gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[0, 0])
ax1.errorbar(I, skewr_mean, yerr=skewr_std_dev, fmt='o', color='blue', ms=8, label='Gaussian')
ax1.plot(I, skewr_actual, '-', color='r', linewidth=2, label='Actual')

ax1.set_xlabel(r"$I$", fontsize=18)
ax1.set_ylabel(r"$S_{3}$", fontsize=18)
ax1.set_title("30K Real alm")
ax1.set_xlim(0, 10.1)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)

ax2 = plt.subplot(gs[0, 1])
ax2.errorbar(I, skewi_mean, yerr=skewi_std_dev, fmt='o', color='blue', ms=8, label='Gaussian')
ax2.plot(I, skewi_actual, '-', color='r', linewidth=2, label='Actual')
ax2.set_xlabel(r"$I$", fontsize=18)
ax2.set_ylabel(r"$S_{3}$", fontsize=18)
ax2.set_title("30K Imag alm")

ax2.set_xlim(0, 10.1)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)

plt.tight_layout()


ax1 = plt.subplot(gs[1, 0])
ax1.errorbar(I, kurtr_mean, yerr=kurtr_std_dev, fmt='o', color='blue', ms=8, label='Gaussian')
ax1.plot(I, kurtr_actual, '-', color='r', linewidth=2, label='Actual')

ax1.set_title("30K Real alm")
ax1.set_xlabel(r"$I$", fontsize=18)
ax1.set_ylabel(r"$S_{4}$", fontsize=18)
ax1.set_xlim(0, 10.1)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)

ax2 = plt.subplot(gs[1, 1])
ax2.errorbar(I, kurti_mean, yerr=kurti_std_dev, fmt='o', color='blue', ms=8, label='Gaussian')
ax2.plot(I, kurti_actual, '-', color='r', linewidth=2, label='Actual')

ax2.set_title("30K Imag alm")
ax2.set_xlabel(r"$I$", fontsize=18)
ax2.set_ylabel(r"$S_{4}$", fontsize=18)
ax2.set_xlim(0, 10.1)
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)

plt.tight_layout()

fig.savefig("/dataspace/sandeep/Bispectrum_data/Gaussian_30K_test/plots/bin_skewness_Kurtosis.eps", dpi=100)
"""

plt.show()


