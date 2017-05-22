"""
This is a Python code for Bispectrum on any scalar(Temprature only) map
we use without binned for all l since my l is less than 250 bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2,
"""

import numpy as np
import healpy as hp
from multiprocessing import Process
import pywigxjpf as wig
from numba import njit
import math as m

name = '/home/sandeep/final_Bispectrum/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)

wig.wig_table_init(1000)
wig.wig_temp_init(1000)


def masking_map(map1, nside, npix, limit):
    """
    This routine to apply mask that we decided using count in cell
    scheme.
    """
    mask = np.zeros(hp.nside2npix(nside), dtype=np.double)
    area = hp.pixelfunc.nside2pixarea(nside, degrees=False)
    for ipix in xrange(0, npix):
        temp = map1[ipix]*area
        if temp < limit:
            mask[ipix] = 1.0
    return mask


def apodiz(mask):
    width = m.radians(2.0)
    apodiz_mask = hp.sphtfunc.smoothing(mask, fwhm=width)
    index = (apodiz_mask < 0)
    apodiz_mask[index] = 0.000
    return apodiz_mask


@njit()
def count_triplet(bin_min, bin_max):
    """
    This routine count number of valid l-triplet in a i-trplet bin
    which we use to evaluate average
    :param bin_min:
    :param bin_max:
    :return:
    """
    count = 0
    for l3 in xrange(bin_min, bin_max):
        for l2 in xrange(bin_min, l3+1):
            for l1 in xrange(bin_min, l2+1):
                if abs(l2-l1) <= l3 <= l2+l1 and (l3+l2+l1) % 2 == 0:  # we applied selection condition tirangle inequality and#parity condition
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


def bispec_estimator(nside_f_est, loop, limit, nmin, nmax):
    """
    :param nside_f_est:
    :param loop:
    :param limit:
    :return:
    """
    npix = hp.nside2npix(nside_f_est)
    print npix

    binary_mask = masking_map(Haslam_512, nside_f_est, npix, limit)
    ap_map = apodiz(binary_mask)

    for fn in xrange(nmin, nmax):
        
        s1 = '/home/sandeep/final_Bispectrum/NonGuassian_Maps_Elsner2009'
        s2 = '/Gaussian_50K_test/Gaussian_Haslam_Maps/haslam_gaussMap_%d.fits' % fn
        filename = s1+s2
        haslam = hp.fitsfunc.read_map(filename)
        lmax = 251
        nbin = 12
        # using Logrithmic bins
        index = 10**np.linspace(np.log10(2), np.log10(251), nbin)
        #logrithmic bins
        for i in xrange(len(index)):
            index[i] = int(index[i])

        bin_arr = [[] for i in range(12)]

        esti_map = np.zeros((nbin, npix), dtype=np.double)
        fwhm = 56./3600.  # For Haslam FWHM is 56 arc min
        beam_l = hp.sphtfunc.gauss_beam(m.radians(fwhm), lmax=lmax, pol=False)
        filtered_map = np.zeros(hp.nside2npix(nside_f_est), dtype=np.float32)

        for i in xrange(0, nbin):
            alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)
            window_func = np.zeros(lmax, float)
            ini = int(index[i])
            if i+1 < nbin:

                final = int(index[i+1])
                bin_arr[i].append(range(ini, final))

                for j in xrange(ini, final):       # Summing over all l in a given bin
                    window_func[j] = 1.0
                    alm_obs = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=True)
                    beam = 1./beam_l
                    alm_obs = hp.sphtfunc.almxfl(alm_obs, beam, mmax=None, inplace=True)
                    alm_true = alm_obs
                    filtered_map += hp.sphtfunc.alm2map(alm_true, nside_f_est, verbose=False)

            esti_map[i, :] = filtered_map

        cl = hp.sphtfunc.anafast(haslam, lmax=lmax, iter=3)
        bin_cl = np.zeros(nbin, dtype=np.float32)
        for i in xrange(0, nbin):
            cl_sum = 0.0
            ini = int(index[i])
            if i+1 < nbin:
                final = int(index[i+1])
                for j in xrange(ini, final):
                    cl_sum += cl[j]
            bin_cl[i] = np.asarray(cl_sum)

        s1 = '/home/sandeep/final_Bispectrum/NonGuassian_Maps_Elsner2009'
        s2 = '/Gaussian_50K_test/Gaussian_Bispectrum/BinnedBispectrum_GaussianMaps_%d_%dk_%d.txt' % (nside_f_est, loop, fn)
        file_name = s1+s2
        print file_name
        with open(file_name, 'w') as f:
            f.write("Bis\tavg_Bis\tCl1\tCl2\tCl3\ti\tj\tk\n")
            for i in xrange(0, nbin-1):
                for j in xrange(0, i+1):
                    for k in xrange(0, j+1):
                        if np.min(bin_arr[k]) - np.max(bin_arr[j]) <= np.max(bin_arr[i]) <= np.max(bin_arr[k]) + np.max(bin_arr[j]):
                            bis = summation(esti_map[i, :], esti_map[j, :], esti_map[k, :], ap_map, npix)
                            trip_count = count_triplet(np.min(bin_arr[k]), np.max(bin_arr[i]))
                            if trip_count != 0.:
                                avgbis = bis / (1.0 * trip_count)
                                f.write("%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e\t%d\t%d\t%d\n" % (
                                        bis, avgbis, bin_cl[k], bin_cl[j], bin_cl[i], i, j, k))


if __name__ == "__main__":

    NSIDE = 512
    #Cell_Count2 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162))
    #Cell_Count1 = Process(target=bispec_estimator, args=(NSIDE, 200, 0.0002553, 0, 101))

    Cell_Count1 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 0, 101))
    Cell_Count1.start()
    Cell_Count2 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 101, 201))
    Cell_Count2.start()
    Cell_Count3 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 201, 301))
    Cell_Count3.start()
    Cell_Count4 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 301, 401))
    Cell_Count4.start()
    Cell_Count5 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 401, 501))
    Cell_Count5.start()
    Cell_Count6 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 501, 601))
    Cell_Count6.start()
    Cell_Count7 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 601, 701))
    Cell_Count7.start()
    Cell_Count8 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 701, 801))
    Cell_Count8.start()
    Cell_Count9 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 801, 901))
    Cell_Count9.start()
    Cell_Count10 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162, 901, 1001))
    Cell_Count10.start()

    Cell_Count1.join()
    Cell_Count2.join()
    Cell_Count3.join()
    Cell_Count4.join()
    Cell_Count5.join()
    Cell_Count6.join()
    Cell_Count7.join()
    Cell_Count8.join()
    Cell_Count9.join()
    Cell_Count10.join()
