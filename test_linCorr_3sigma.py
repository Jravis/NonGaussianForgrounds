import numpy as np
import healpy as hp
from multiprocessing import Process
import time
import pywigxjpf as wig
from numba import njit
import _countTriplet

#wig.wig_table_init(1000)
#wig.wig_temp_init(1000)

lmax = 256
nbin = 11
NSIDE = 128
npix = hp.nside2npix(NSIDE)

index = np.logspace(np.log10(10), np.log10(256), nbin, endpoint=True, dtype=np.int32)
ind = (index != 11)
index = index[ind]
print index


def summation(arr1, arr2, f_sky):

    bi_sum = np.sum(np.multiply(arr1, arr2))
    bi_sum /= (4.0*np.pi*f_sky)
    return bi_sum


def summation1(arr1, arr2, arr3, f_sky):

    tmp = np.multiply(arr1, arr2)
    bi_sum = np.sum(np.multiply(tmp, arr3))
    bi_sum /= (4.0*np.pi*f_sky)
    return bi_sum


def filter_arr(ini, final):

    delta_l = 2
    window_func = np.zeros(lmax, dtype=np.float)
    for l in xrange(ini, final):
        if ini + delta_l <= l <= final - delta_l:
            window_func[l] = 1.0

        elif ini <= l < ini + delta_l:

            window_func[l] = np.cos(np.pi * 0.5 * ((ini + delta_l) - l) / delta_l) ** 2

        elif final - delta_l < l < final:

            window_func[l] = 1.0 * np.cos(np.pi * 0.5 * (l - (final - delta_l)) / delta_l) ** 2

    return window_func


bin_arr = np.zeros((nbin - 1, 2), dtype=np.int32)
window_func_filter = np.zeros((nbin-1, lmax), dtype=np.float)

for i in xrange(0, nbin):
    ini = index[i]

    if i + 1 < nbin:
        final = index[i + 1]
        bin_arr[i, 0] = ini
        bin_arr[i, 1] = final - 1

        # Summing over all l in a given bin we are using top-hat filter
        # will use smoothing technique to stop ripple
        window_func_filter[i, :] = filter_arr(ini, final)


def gauss_almobs(loop1, sky_mask):

    alm_obs_Gauss = np.zeros((1000, 33153), dtype=np.complex128)

    for fn in xrange(0, 1000):

        s1 = '/dataspace/sandeep/Bispectrum_data'
        s2 = '/Gaussian_%s_test/Gaussian_%s_Maps/haslam_%sgaussMap_%d.fits' % (loop1, loop1, loop1, fn)

        filename = s1+s2
        Gauss_map = hp.fitsfunc.read_map(filename, verbose=False)*sky_mask
        alm_obs_Gauss[fn, :] = hp.sphtfunc.map2alm(Gauss_map, lmax=lmax, iter=3)

    return alm_obs_Gauss


def test_avg_arr(Apo_mask, gauss_alm):

    Gauss_esti_map = np.zeros((nbin-1, 1000, npix), dtype=np.double)

    for j in xrange(0, nbin-1):
        for i in xrange(0, 1000):

            alm_obs = hp.sphtfunc.almxfl(gauss_alm[i, :], window_func_filter[j, :], mmax=None, inplace=False)
            test_map = hp.sphtfunc.alm2map(alm_obs, NSIDE, verbose=False)
            Gauss_esti_map[j, i, :] = test_map*Apo_mask

    return Gauss_esti_map


def Gauss_Avg(I, J, K, inp_arr, map_indx):

    Gauss_esti_map_I = np.zeros((1000, npix), dtype=np.double)
    Gauss_esti_map_J = np.zeros((1000, npix), dtype=np.double)
    Gauss_esti_map_K = np.zeros((1000, npix), dtype=np.double)

    for i in xrange(0, 1000):
        if i != map_indx:
            a = inp_arr[I, i, :]
            b = inp_arr[J, i, :]
            c = inp_arr[K, i, :]

            Gauss_esti_map_I[i, :] = np.multiply(a, b)
            Gauss_esti_map_J[i, :] = np.multiply(b, c)
            Gauss_esti_map_K[i, :] = np.multiply(a, c)

    meanI = np.mean(Gauss_esti_map_I, axis=0)
    meanJ = np.mean(Gauss_esti_map_J, axis=0)
    meanK = np.mean(Gauss_esti_map_K, axis=0)

    return meanI,  meanJ, meanK


def bispec_estimator(loop, G_alm, apod_mask, mask_128_80K, nmin, nmax):

    for fn in xrange(nmin, nmax):

        s1 = '/dataspace/sandeep/Bispectrum_data'
        s2 = '/Gaussian_%s_test/Gaussian_%s_Maps/haslam_%sgaussMap_%d.fits' % (loop, loop, loop, fn)

        filename = s1+s2

        Haslam_128 = hp.fitsfunc.read_map(filename, verbose=False)

        haslam = Haslam_128 * mask_128_80K  # Galactic mask to reduce leakage for low temprature map
        # using Logrithmic bins

        esti_map = np.zeros((nbin, npix), dtype=np.double)

        alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)

        for i in xrange(0, nbin-1):
            # Summing over all l in a given bin we are using top-hat filter
            # will use smoothing technique to stop ripple
            alm_test = hp.sphtfunc.almxfl(alm_obs, window_func_filter[i, :], mmax=None, inplace=False)
            test_map = hp.sphtfunc.alm2map(alm_test, 128, verbose=False)
            esti_map[i, :] = test_map * apod_mask

        frac_sky = np.sum(apod_mask)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_%s_test/Gaussian_Bin_Bispectrum/' % loop
        s2 = 'BinnedBispectrum_Bin_GaussianMaps_linCorr_%d_%s_%d.txt' % (NSIDE, loop, fn)
        file_name = s1 + s2
        print file_name
        with open(file_name, 'w') as f:

            f.write("Bis\tI1\tI2\tI3\n")
            for I1 in xrange(0, nbin - 1):
                for I2 in xrange(I1,  nbin-1):
                    for I3 in xrange(I2, nbin-1):

                        Avg_arrI1, Avg_arrI2, Avg_arrI3 = Gauss_Avg(I1, I2, I3, G_alm, fn)
                        bis1 = summation(esti_map[I1, :], Avg_arrI2, frac_sky)
                        bis2 = summation(esti_map[I2, :], Avg_arrI3, frac_sky)
                        bis3 = summation(esti_map[I3, :], Avg_arrI1, frac_sky)
                        bis = bis1+bis2+bis3
                        Bis = summation1(esti_map[I1, :], esti_map[I2, :], esti_map[I3, :], frac_sky)
                        f.write("%0.6e\t%d\t%d\t%d\n" % ((Bis-bis), I1, I2, I3))


if __name__ == "__main__":

    nmin = 0
    nmax = 0
    count = 0
    min_core = 1
    max_core = 20
    increment = 50
    str = []
    TEMP = ['30K']#,'30K', '40K', '50K', '60K']

    for i in xrange(1, max_core + 1):
        s = 'Cell_Count%d' % i
        str.append(s)

    f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits" % TEMP[0]
    ap_mask_128 = hp.fitsfunc.read_map(f_name1, verbose=False)

    name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_80K_apod_300arcm_ns_128.fits'
    mask_80K = hp.fitsfunc.read_map(name, verbose=False)

    start = time.time()
    Gauss_alm = gauss_almobs(TEMP[0], mask_80K)
    Inp_Arr = test_avg_arr(ap_mask_128, Gauss_alm)
    stop = time.time()
    print "time taken for Gauss_alm"
    print stop-start
    print '---------------------------'

    for i in xrange(len(str)):
        nmin = count
        nmax = count + increment
        if nmax == 1000:
            nmax = 1001
        str[i] = Process(target=bispec_estimator, args=(TEMP[0], Inp_Arr, ap_mask_128, mask_80K, nmin, nmax))
        str[i].start()
        count = nmax

    for i in xrange(len(str)):
        str[i].join()
