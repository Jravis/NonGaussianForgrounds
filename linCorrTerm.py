"""
This code is to compute Linear correction term for Bispectrum to 
take care of beam effect.
"""

import numpy as np 
import healpy as hp
from multiprocessing import Process
import pywigxjpf as wig
from numba import njit
import _countTriplet

name = '/dataspace/sandeep/Bispectrum_data/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)
npix = hp.nside2npix(128)
lmax = 256

#@njit()


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
                if (l3+l2+l1) % 2 == 0:
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


def summation(arr1, arr2, f_sky):

    bi_sum = np.sum(np.multiply(arr1, arr2))
    bi_sum /= (4.0*np.pi*f_sky)
    return bi_sum


def filter_arr(ini, final):

    delta_l = 2
    window_func = np.zeros(lmax, float)
    for l in xrange(ini, final):
        if ini + delta_l <= l <= final - delta_l:
            window_func[l] = 1.0

        elif ini <= l < ini + delta_l:

            window_func[l] = np.cos(np.pi * 0.5 * ((ini + delta_l) - l) / delta_l) ** 2

        elif final - delta_l < l < final:

            window_func[l] = 1.0 * np.cos(np.pi * 0.5 * (l - (final - delta_l)) / delta_l) ** 2

    return window_func


def gauss_almobs(loop1, sky_mask):

    alm_obs_Gauss =[]# np.zeros((1000, 33153), dtype=np.float64)
    for fn in xrange(0, 1000):

        s1 = '/dataspace/sandeep/Bispectrum_data'
        s2 = '/Gaussian_%s_test/Gaussian_%s_Maps/haslam_%sgaussMap_%d.fits' % (loop1, loop1, loop1, fn)

        filename = s1+s2
        Gauss_map = hp.fitsfunc.read_map(filename, verbose=False)*sky_mask
        alm_obs_Gauss.append(hp.sphtfunc.map2alm(Gauss_map, lmax=lmax, iter=3))

    return alm_obs_Gauss


def Gauss_Avg(BinI,  BinJ, BinK, Apo_mask, gauss_alm, ns):

    Gauss_esti_map_I = np.zeros((1000, npix), dtype=np.double)
    Gauss_esti_map_J = np.zeros((1000, npix), dtype=np.double)
    Gauss_esti_map_K = np.zeros((1000, npix), dtype=np.double)

    window_func_I = filter_arr(BinI[0], BinI[1])
    window_func_J = filter_arr(BinJ[0], BinJ[1])
    window_func_K = filter_arr(BinK[0], BinK[1])

    for i in xrange(0, 1000):

        alm_obs_I = hp.sphtfunc.almxfl(gauss_alm[i], window_func_I, mmax=None, inplace=True)
        test_map_I = hp.sphtfunc.alm2map(alm_obs_I, ns, verbose=False)

        alm_obs_J = hp.sphtfunc.almxfl(gauss_alm[i], window_func_J, mmax=None, inplace=True)
        test_map_J = hp.sphtfunc.alm2map(alm_obs_J, ns, verbose=False)

        alm_obs_K = hp.sphtfunc.almxfl(gauss_alm[i], window_func_K, mmax=None, inplace=True)
        test_map_K = hp.sphtfunc.alm2map(alm_obs_K, ns, verbose=False)

        a = test_map_I*Apo_mask
        b = test_map_J*Apo_mask
        c = test_map_K*Apo_mask

        Gauss_esti_map_I[i, :] = np.multiply(a, b)
        Gauss_esti_map_J[i, :] = np.multiply(b, c)
        Gauss_esti_map_K[i, :] = np.multiply(a, c)

    meanI = np.mean(Gauss_esti_map_I, axis=0)
    meanJ = np.mean(Gauss_esti_map_J, axis=0)
    meanK = np.mean(Gauss_esti_map_K, axis=0)

    return meanI,  meanJ, meanK


def bispec_estimator(nside_f_est, loop, apod_mask):

    # Masking and apodization

    Haslam_128 = hp.pixelfunc.ud_grade(Haslam_512, nside_out=128)

    name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_80K_apod_300arcm_ns_128.fits'
    mask_128_80K = hp.fitsfunc.read_map(name)

    haslam = Haslam_128 * mask_128_80K  # Galactic mask to reduce leakage for low temprature map
    nbin = 11
    index = np.logspace(np.log10(10), np.log10(256), nbin, endpoint=True, dtype=np.int32)
    ind = (index != 11)
    index = index[ind]

    # using Logrithmic bins

    delta_l = 2.
    esti_map = np.zeros((nbin, npix), dtype=np.double)
    bin_arr = np.zeros((nbin - 1, 2), dtype=np.int32)
    alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)

    for i in xrange(0, nbin):
        window_func = np.zeros(lmax, float)
        ini = index[i]

        if i+1 < nbin:
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
            esti_map[i, :] = test_map*apod_mask
    frac_sky = np.sum(apod_mask)

    Gauss_alm = gauss_almobs(loop, mask_128_80K)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_%s_test/' % loop
    s2 = 'All_mode/Analysis_Bin_Bispectrum_%d_%s.txt' % (nside_f_est, loop)

    file_name = s1+s2
    print file_name

    with open(file_name, 'w') as f:

        f.write("Bis\tI1\tI2\tI3\tcount\n")
        for I1 in xrange(0, nbin - 1):
            for I2 in xrange(I1, nbin-1):
                for I3 in xrange(I2, nbin-1):
                   
                    foo = [bin_arr[I1, 0], bin_arr[I1, 1]+1]
                    bar = [bin_arr[I2, 0], bin_arr[I2, 1]+1]
                    yo = [bin_arr[I3, 0], bin_arr[I3, 1]+1]
                    Avg_arrI1, Avg_arrI2, Avg_arrI3 = Gauss_Avg(foo, bar, yo, apod_mask, Gauss_alm, nside_f_est)

                    bis1 = summation(esti_map[I1, :], Avg_arrI2, frac_sky)
                    bis2 = summation(esti_map[I2, :], Avg_arrI3, frac_sky)
                    bis3 = summation(esti_map[I3, :], Avg_arrI1, frac_sky)
                    bis = bis1+bis2+bis3

                    f.write("%0.6e\t%d\t%d\t%d\n" % (bis, I1, I2, I3))


if __name__ == "__main__":

    NSIDE = 128
    TEMP = ['30K', '40K', '50K', '60K']
    min_core = 1
    max_core = 4
    strn = []
    for i in xrange(0, max_core):
        s = 'Cell_Count%d' % (i + 1)
        strn.append(s)
    print len(TEMP), len(strn)

    for i in xrange(len(strn)):
        f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits" % TEMP[i]
        print f_name1
        ap_mask_128 = hp.fitsfunc.read_map(f_name1)

        strn[i] = Process(target=bispec_estimator, args=(NSIDE, TEMP[i], ap_mask_128))
        strn[i].start()

    for i in xrange(len(strn)):
        strn[i].join()
