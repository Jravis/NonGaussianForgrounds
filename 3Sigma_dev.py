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
import math as m


#wig.wig_table_init(1000)
#wig.wig_temp_init(1000)


@njit()
def count_triplet(bin_1, bin_2, bin_3):
    """
    This routine count number of valid l-triplet in a i-trplet bin
    which we use to evaluate average
    :param bin_min:
    :param bin_max:
    :return:
    """
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

@njit()
def summation(arr1, arr2, arr3, arr4, num_pix):
    """
    :param arr1:
    :param arr2:
    :param arr3:
    :param arr4:
    :param num_pix:
    :return:
    """
    bi_sum = 0.0
    for ipix in xrange(0, num_pix):
        product = arr1[ipix]*arr2[ipix]*arr3[ipix]*arr4[ipix]
        bi_sum += product
    bi_sum /= (4.0*np.pi*np.sum(arr4))
    return bi_sum


def bispec_estimator(nside_f_est, loop, ap_map, nmin, nmax):
    """
    :param nside_f_est:
    :param loop:
    :param limit:
    :return:
    """
    npix = hp.nside2npix(nside_f_est)
    print npix
    for fn in xrange(nmin, nmax):

        s1 = '/dataspace/sandeep/Bispectrum_data'
        s2 = '/Gaussian_104K_test/Gaussian_104K_Maps/haslam_104KgaussMap_%d.fits' % fn

        filename = s1+s2
        haslam = hp.fitsfunc.read_map(filename)*ap_map
        lmax = 251
        nbin = 12

        # using Logrithmic bins

        index = 10**np.linspace(np.log10(2), np.log10(251), nbin)
        #index = 10**np.linspace(np.log10(11), np.log10(251), nbin)

        # logrithmic bins

        for i in xrange(len(index)):
            index[i] = int(index[i])

        bin_arr = np.zeros((nbin-1, 2), dtype=int)
        esti_map = np.zeros((nbin, npix), dtype=np.double)

        #fwhm = 56./60.  # For Haslam FWHM is 56 arc min
        #beam_l = hp.sphtfunc.gauss_beam(m.radians(fwhm), lmax=lmax, pol=False)

        for i in xrange(0, nbin):

            alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)

            window_func = np.zeros(lmax, float)

            ini = int(index[i])

            if i+1 < nbin:

                final = int(index[i+1])

                bin_arr[i, 0] = ini

                bin_arr[i, 1] = final - 1

                for j in xrange(ini, final):  # Summing over all l in a given bin

                    window_func[j] = 1.0

                alm_obs = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=True)
                #alm_obs = hp.sphtfunc.almxfl(alm_obs, 1./beam_l, mmax=None, inplace=True)

                alm_true = alm_obs

                esti_map[i, :] = hp.sphtfunc.alm2map(alm_true, nside_f_est, verbose=False)

        s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_104K_test/Gaussian_Bin_Bispectrum/'
        s2 = 'BinnedBispectrum_GaussianMaps_%d_%dk_%d.txt' % (nside_f_est, loop, fn)
        file_name = s1+s2

        with open(file_name, 'w') as f:
            f.write("Bis\ti\tj\tk\tcount\n")
            for i in xrange(0, nbin-1):
                for j in xrange(i, nbin-1):
                    for k in xrange(j, nbin-1):
                        bis = summation(esti_map[i, :], esti_map[j, :], esti_map[k, :], ap_map, npix)
                        trip_count = count_triplet(bin_arr[i, :], bin_arr[j, :], bin_arr[k, :])
                        f.write("%0.6e\t%d\t%d\t%d\t%d\n" % (bis, i, j, k, trip_count))


if __name__ == "__main__":

    NSIDE = 512
    nmin = 0
    nmax = 0
    count = 0
    min_core = 1
    max_core = 20
    increment = 50
    str = []
    TEMP = 104

    f_name = "/dataspace/sandeep/Bispectrum_data/Input_Maps/ApodizeBinaryMask_%s_%0.1fdeg_apodi.fits" % ('104K', 5.0)
    apd_map = hp.fitsfunc.read_map(f_name)

    for i in xrange(1, max_core + 1):
        s = 'Cell_Count%d' % i
        str.append(s)
    print len(str)

    for i in xrange(len(str)):
        nmin = count
        nmax = count + increment
        if nmax == 1000:
            nmax = 1001
        print nmin, nmax, i
        str[i] = Process(target=bispec_estimator, args=(NSIDE, TEMP, apd_map, nmin, nmax))
        str[i].start()
        count = nmax

    for i in xrange(len(str)):
        str[i].join()

