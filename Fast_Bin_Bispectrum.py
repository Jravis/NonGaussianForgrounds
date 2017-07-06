"""
This is a Python code for Bispectrum on any scalar(Temprature only) map 
we use Binned bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2, 
"""

import numpy as np 
import healpy as hp
from multiprocessing import Process
import pywigxjpf as wig
import matplotlib.pyplot as plt
from numba import njit
import math as m
name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)

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
    # print "numpix %d -" % num_pix
    #print "sum mask %d -" % np.sum(arr4)
    bi_sum = 0.0
    for ipix in xrange(0, num_pix):
        product = arr1[ipix]*arr2[ipix]*arr3[ipix]*arr4[ipix]
        bi_sum += product
    bi_sum /= (4.0*np.pi*np.sum(arr4))
    return bi_sum


def bispec_estimator(nside_f_est, loop, ap_map):
    """
    :param nside_f_est:
    :param loop:
    :param limit:
    :return:
    """

# Masking and apodization

    npix = hp.nside2npix(nside_f_est)
    haslam = Haslam_512 * ap_map
    lmax = 250
    nbin = 12

# using Logrithmic bins

#    index = 10**np.linspace(np.log10(2), np.log10(251), nbin)  #logrithmic bins
    index = 10 ** np.linspace(np.log10(11), np.log10(251), nbin)
    for i in xrange(len(index)):
        index[i] = int(index[i])

    print index
    # creating filtered map using equation 6 casaponsa et al. and eq (2.6) in Bucher et.al 2015
    esti_map = np.zeros((nbin, npix), dtype=np.double)

  #  fwhm = 56./60.  # For Haslam FWHM is 56 arc min
  # beam_l = hp.sphtfunc.gauss_beam(m.radians(fwhm), lmax=lmax, pol=False)

    bin_arr = np.zeros((nbin - 1, 2), dtype=int)

    for i in xrange(0, nbin):
        alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)
        window_func = np.zeros(lmax, float)
        ini = int(index[i])
        if i+1 < nbin:
            final = int(index[i + 1])
            bin_arr[i, 0] = ini
            bin_arr[i, 1] = final - 1

            for j in xrange(ini, final):  # Summing over all l in a given bin
                window_func[j] = 1.0
            alm_obs = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=True)
#            alm_obs = hp.sphtfunc.almxfl(alm_obs, 1./beam_l, mmax=None, inplace=True)
            alm_true = alm_obs
            esti_map[i, :] = hp.sphtfunc.alm2map(alm_true, nside_f_est, verbose=False)

    s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_18K_test/'
    s2 = 'Analysis_18KBin_Bispectrum_%d_%d.txt' % (nside_f_est, loop)
    file_name = s1+s2
    print file_name
    with open(file_name, 'w') as f:
        f.write("Bis\ti\tj\tk\tcount\n")
        for i in xrange(0, nbin - 1):
            for j in xrange(i, nbin-1):
                for k in xrange(j, nbin-1):
                    bis = summation(esti_map[i, :], esti_map[j, :], esti_map[k, :], ap_map, npix)
                    trip_count = count_triplet(bin_arr[i, :], bin_arr[j, :], bin_arr[k, :])
                    f.write("%0.6e\t%d\t%d\t%d\t%d\n" % (bis, i, j, k, trip_count))

if __name__ == "__main__":

    NSIDE = 512
    TEMP = 18
    f_name = "/dataspace/sandeep/Bispectrum_data/Input_Maps/ApodizeBinaryMask_%s_%0.1fdeg_apodi.txt" % ('200K', 2.0)
    print f_name
    apd_map = hp.fitsfunc.read_map(f_name)
    Cell_Count1 = Process(target=bispec_estimator, args=(NSIDE, TEMP, apd_map))
    Cell_Count1.start()
    Cell_Count1.join()



    #Cell_Count1 = Process(target=bispec_estimator, args=(NSIDE, 18, 0.000073))
    #Cell_Count1.start()
    #Cell_Count2 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162))
    #Cell_Count2.start()
    #Cell_Count3 = Process(target=bispec_estimator, args=(NSIDE, 200, 0.0002553))
    #Cell_Count3.start()
    #Cell_Count4 = Process(target=bispec_estimator, args=(NSIDE, 30, 0.000122))
    #Cell_Count4.start()

    #Cell_Count2.join()
    #Cell_Count3.join()
    #Cell_Count4.join()




