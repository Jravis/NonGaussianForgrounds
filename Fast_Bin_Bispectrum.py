"""
This is a Python code for Bispectrum on any scalar(Temprature only) map
we use Binned bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2,
"""

import numpy as np
import healpy as hp
from multiprocessing import Process
import pywigxjpf as wig
from numba import njit
import _countTriplet

name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)
nside_f_est =128
npix = hp.nside2npix(nside_f_est)
lmax = 256

#nbin =11
#index = np.logspace(np.log10(10), np.log10(256), nbin, endpoint=True, dtype=np.int32)
#ind = (index != 11)
#index = index[ind]
#print index

# *****************************************************************
# for 15 bin scheme

index = [10, 19, 27, 39, 46, 55, 65, 77, 91, 109, 129, 153, 181, 215, 256]
index = np.asarray(index, dtype=np.int32)
nbin = len(index)

# *****************************************************************
# for 20 bin scheme
#index = [10, 15, 21, 27, 34, 42, 53, 59, 66, 74, 83, 93, 104, 117, 130, 146, 163, 183, 204, 256]
#index = np.asarray(index, dtype=np.int32)
#nbin = len(index)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def summation(arr1, arr2, f_sky):

    bi_sum = np.sum(np.multiply(arr1, arr2), dtype=np.double)
    bi_sum /= (4.0*np.pi*f_sky)
    return bi_sum


def summation1(arr1, arr2, arr3, f_sky):

    tmp = np.multiply(arr1, arr2)
    bi_sum = np.sum(np.multiply(tmp, arr3), dtype=np.double)
    bi_sum /= (4.0*np.pi*f_sky)
    return bi_sum


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bin_arr = np.zeros((nbin - 1, 2), dtype=np.int32)
window_func_filter = np.zeros((nbin-1, lmax), dtype=np.float64)

for i in xrange(0, nbin):
    ini = index[i]

    if i + 1 < nbin:
        final = index[i + 1]
        bin_arr[i, 0] = ini
        bin_arr[i, 1] = final - 1

        # Summing over all l in a given bin we are using top-hat filter
        # will use smoothing technique to stop ripple
        window_func_filter[i, :] = filter_arr(ini, final)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def gauss_almobs(loop1, sky_mask):

    alm_obs_Gauss = np.zeros((1000, 33153), dtype=np.complex128)

    for fn in xrange(0, 1000):

        s1 = '/dataspace/sandeep/Bispectrum_data'
        s2 = '/Gaussian_%s_test/Gaussian_%s_Maps/haslam_%sgaussMap_%d.fits' % (loop1, loop1, loop1, fn)

        filename = s1+s2
        Gauss_map = hp.fitsfunc.read_map(filename, verbose=False)*sky_mask
        alm_obs_Gauss[fn, :] = hp.sphtfunc.map2alm(Gauss_map, lmax=lmax, iter=3)

    return alm_obs_Gauss


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def test_avg_arr(Apo_mask, gauss_alm):

    Gauss_esti_map = np.zeros((nbin-1, 1000, npix), dtype=np.double)

    for j in xrange(0, nbin-1):
        for i in xrange(0, 1000):
            alm_obs = hp.sphtfunc.almxfl(gauss_alm[i, :], window_func_filter[j, :], mmax=None, inplace=False)
            test_map = hp.sphtfunc.alm2map(alm_obs, NSIDE, verbose=False)
            Gauss_esti_map[j, i, :] = test_map*Apo_mask

    return Gauss_esti_map


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Gauss_Avg(I, J, K, inp_arr):

    Gauss_esti_map_I = np.zeros((1000, npix), dtype=np.double)
    Gauss_esti_map_J = np.zeros((1000, npix), dtype=np.double)
    Gauss_esti_map_K = np.zeros((1000, npix), dtype=np.double)

    for i in xrange(0, 1000):
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def bispec_estimator(loop, apod_mask, mask_128_80K):
    """
    :param nside_f_est:
    :param loop:
    :param limit:
    :return:
    """

    # Masking and apodization
    Gauss_alm = gauss_almobs(loop, mask_128_80K)
    G_alm = test_avg_arr(ap_mask_128, Gauss_alm)

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    Haslam_128 = hp.pixelfunc.ud_grade(Haslam_512, nside_out=128)
    haslam = Haslam_128 * mask_128_80K  # Galactic mask to reduce leakage for low temprature map

    esti_map = np.zeros((nbin, npix), dtype=np.double)
    alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)

    for i in xrange(0, nbin-1):
        alm_test = hp.sphtfunc.almxfl(alm_obs, window_func_filter[i, :], mmax=None, inplace=False)
        test_map = hp.sphtfunc.alm2map(alm_test, 128, verbose=False)
        esti_map[i, :] = test_map*apod_mask

    frac_sky = np.sum(apod_mask)

    s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_%s_test/' % loop
    s2 = 'All_mode/Analysis_Bin_Bispectrum_%d_%s.txt' % (nside_f_est, loop)

    file_name = s1+s2
    print file_name
    with open(file_name, 'w') as f:
        f.write("Bis\tI1\tI2\tI3\n")
        for I1 in xrange(0, nbin - 1):
            for I2 in xrange(I1, nbin - 1):
                for I3 in xrange(I2, nbin - 1):
                    Avg_arrI1, Avg_arrI2, Avg_arrI3 = Gauss_Avg(I1, I2, I3, G_alm)
                    bis1 = summation(esti_map[I1, :], Avg_arrI2, frac_sky)
                    bis2 = summation(esti_map[I2, :], Avg_arrI3, frac_sky)
                    bis3 = summation(esti_map[I3, :], Avg_arrI1, frac_sky)
                    bis = bis1 + bis2 + bis3
                    Bis = summation1(esti_map[I1, :], esti_map[I2, :], esti_map[I3, :], frac_sky)
                    f.write("%0.6e\t%d\t%d\t%d\n" % ((Bis - bis), I1, I2, I3))


if __name__ == "__main__":

    NSIDE = 128
    TEMP = ['30K', '40K', '50K', '60K']

    min_core = 1
    max_core = 4
    strn = []
    for i in xrange(0, max_core):
        s = 'Cell_Count%d' % (i+1)
        strn.append(s)
    print len(TEMP), len(strn)

    name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_80K_apod_300arcm_ns_128.fits'
    mask_80K = hp.fitsfunc.read_map(name, verbose=False)

    for i in xrange(len(strn)):

        f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits" % TEMP[i]
        print f_name1
        ap_mask_128 = hp.fitsfunc.read_map(f_name1)
        strn[i] = Process(target=bispec_estimator, args=(TEMP[i],  ap_mask_128, mask_80K))
        strn[i].start()

    for i in xrange(len(strn)):
        strn[i].join()


