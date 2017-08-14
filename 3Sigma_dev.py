"""
This is a Python code for Bispectrum on any scalar(Temprature only) map
we use without binned for all l since my l is less than 250 bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2,

To use 1000 gaussian simulation maps
"""

import numpy as np
import healpy as hp
from multiprocessing import Process
import pywigxjpf as wig
from numba import njit
import _countTriplet
#wig.wig_table_init(1000)
#wig.wig_temp_init(1000)


@njit()
def count_triplet(bin_1, bin_2, bin_3):
    count = 0
    for l3 in xrange(bin_1[0], bin_1[1]+1):
        for l2 in xrange(bin_2[0], bin_2[1]+1):
            for l1 in xrange(bin_3[0], bin_3[1]+1):
                if abs(l2-l1) <= l3 <= l2+l1 and (l3+l2+l1) % 2 == 0:
                    count += 1
    return count


@njit()
def g(l1, l2, l3):
    """
    :param l1:
    :param l2:
    :param l3:
    :return:
    """
    if l1 == l2 and l2 == l3:
        return 6.0
    elif l1 == l2 or l2 == l3 or l3 == l1:
        return 2.0
    else:
        return 1.0

#@njit()


def summation(arr1, arr2, arr3, frac_sky):

    foo = np.multiply(arr1, arr2)
    bi_sum = np.sum(np.multiply(foo, arr3))
    bi_sum /= (4.0*np.pi*frac_sky)
    return bi_sum


def bispec_estimator(loop, apod_mask, nmin, nmax):

    NSIDE = 128
    npix = hp.nside2npix(NSIDE)

    name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_80K_apod_300arcm_ns_128.fits'
    mask_128_80K = hp.fitsfunc.read_map(name)

    for fn in xrange(nmin, nmax):

        s1 = '/dataspace/sandeep/Bispectrum_data'
        s2 = '/Gaussian_%s_test/Gaussian_%s_Maps/haslam_%sgaussMap_%d.fits' % (loop, loop, loop, fn)

        filename = s1+s2

        Haslam_128 = hp.fitsfunc.read_map(filename, verbose=False)

        haslam = Haslam_128 * mask_128_80K  # Galactic mask to reduce leakage for low temprature map
        # using Logrithmic bins
        lmax = 256
        nbin = 11

        index = np.logspace(np.log10(10), np.log10(256), nbin, endpoint=True, dtype=np.int32)
        ind = (index != 11)
        index = index[ind]
        print index
        delta_l = 2.
        esti_map = np.zeros((nbin, npix), dtype=np.double)
        bin_arr = np.zeros((nbin - 1, 2), dtype=np.int32)
        alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)

        for i in xrange(0, nbin):
            window_func = np.zeros(lmax, float)
            ini = index[i]

            if i + 1 < nbin:
                final = index[i + 1]
                bin_arr[i, 0] = ini
                bin_arr[i, 1] = final - 1

                # Summing over all l in a given bin we are using top-hat filter
                # will use smoothing technique to stop ripple

                for l in xrange(ini, final):  # Summing over all l in a given bin

                    if ini + delta_l <= l <= final - delta_l:
                        window_func[l] = 1.0

                    elif ini <= l < ini + delta_l:

                        window_func[l] = np.cos(np.pi * 0.5 * ((ini + delta_l) - l) / delta_l) ** 2

                    elif final - delta_l < l < final:

                        window_func[l] = 1.0 * np.cos(np.pi * 0.5 * (l - (final - delta_l)) / delta_l) ** 2

                alm_test = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=False)
                test_map = hp.sphtfunc.alm2map(alm_test, 128, verbose=False)
                esti_map[i, :] = test_map * apod_mask

        f_sky = np.sum(apod_mask)

        s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_%s_test/Gaussian_Bin_Bispectrum/' % loop
        s2 = 'BinnedBispectrum_Bin_GaussianMaps_%d_%s_%d.txt' % (NSIDE, loop, fn)
        file_name = s1+s2

        with open(file_name, 'w') as f:
            f.write("Bis\tI3\tI2\tI1\tcount\n")
            # we will have l3>=l2>=l1 scheme
            # If considering all mode

            for I1 in xrange(0, nbin - 1):
                for I2 in xrange(I1, nbin - 1):
                    for I3 in xrange(I2, nbin - 1):

                        bis = summation(esti_map[I1, :], esti_map[I2, :], esti_map[I3, :], f_sky)
                        f.write("%0.6e\t%d\t%d\t%d\n" % (bis, I1, I2, I3))
                        #trip_count = _countTriplet.countTriplet(bin_arr[i, :], bin_arr[j, :], bin_arr[k, :])
                        #f.write("%0.6e\t%d\t%d\t%d\t%d\n" % (bis, i, j, k, trip_count))


if __name__ == "__main__":

    nmin = 0
    nmax = 0
    count = 0
    min_core = 1
    max_core = 20
    increment = 50
    str = []
    TEMP = ['60K']#,'30K', '40K', '50K', '60K']

    for i in xrange(1, max_core + 1):
        s = 'Cell_Count%d' % i
        str.append(s)

    for i in xrange(len(str)):

        f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits" % TEMP[0]
        ap_mask_128 = hp.fitsfunc.read_map(f_name1, verbose=False)
        nmin = count
        nmax = count + increment
        if nmax == 1000:
            nmax = 1001
        str[i] = Process(target=bispec_estimator, args=(TEMP[0], ap_mask_128, nmin, nmax))
        str[i].start()
        count = nmax

    for i in xrange(len(str)):
        str[i].join()

