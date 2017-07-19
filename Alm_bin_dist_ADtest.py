import numpy as np
import healpy as hp
from astroML.plotting import hist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'
Haslam_512 = hp.fitsfunc.read_map(name)

nside_f_est = 512

f_name = "/dataspace/sandeep/Bispectrum_data/Input_Maps/ApodizeBinaryMask_%s_%0.1fdeg_apodi.fits" % ('18K', 2.0)
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
NG_actual_real =np.zeros(nbin-1, dtype=np.float32)
NG_actual_imag = np.zeros(nbin-1, dtype=np.float32)

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

    ar, critr, signir = stats.anderson(almr, dist='norm')
    ai, criti, signii = stats.anderson(almi, dist='norm')

    if ar > critr[2]:
        NG_actual_real[i] = ar
    if ai > criti[2]:
        NG_actual_imag[i] = ai

print NG_actual_imag
print NG_actual_real
NGr = np.zeros((1000, nbin-1), dtype=np.float32)
NGi = np.zeros((1000, nbin-1), dtype=np.float32)


for fn in xrange(0, 1000):
    s1 = '/dataspace/sandeep/Bispectrum_data'
    s2 = '/Gaussian_18K_test/Gaussian_18K_Maps/haslam_18KgaussMap_%d.fits' % fn
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
        ar, critr, signir = stats.anderson(almr, dist='norm')
        ai, criti, signii = stats.anderson(almi, dist='norm')
        if ar > critr[2]:
            NGr[fn, i] += 1

        if ai > criti[2]:
            NGi[fn, i] += 1

for i in xrange(0, nbin-1):
    print np.sum(NGr[:, i])
    print np.sum(NGi[:, i])
