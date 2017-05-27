"""
This is a Python code for Bispectrum on any scalar(Temprature only) map
we use Binned bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2,
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pywigxjpf as wig
from numba import njit
import math as m
from multiprocessing import Process

name = '/home/sandeep/final_Bispectrum/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)
lmax = 251
NSIDE = 512

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


def gaussian_maps(nmin, nmax):
    """
    :param number:
    :return:
    """
    np.random.seed(49390927)  # fixing random Seed
    limit = 0.0002553 # 200
    #limit = 0.000162
    npix = hp.nside2npix(NSIDE)
    print npix
    binary_mask = masking_map(Haslam_512, NSIDE, npix, limit)
    ap_map = apodiz(binary_mask)
    haslam = Haslam_512 * ap_map

    cl = hp.sphtfunc.anafast(haslam, lmax=250, iter=3)
    s1 = "/dataspace/sandeep/Bispectrum_data"
    s2 = "/Gaussian_200K_test/Gaussian_Haslam_cl/haslam_50K_cl.txt"
    name = s1+s2
    np.savetxt(name, cl, fmt="%0.6f")
    # creating filtered map

    for i in xrange(nmin, nmax):
        Map = hp.sphtfunc.synfast(cl, NSIDE, lmax=250, pol=True, pixwin=False, fwhm=0.0, sigma=None, verbose=False)
        Map = Map*ap_map
        Map_cl = hp.sphtfunc.anafast(Map, lmax=250, iter=3)
        s1 = "/home/sandeep/final_Bispectrum/NonGuassian_Maps_Elsner2009"
        s2 = "/Gaussian_50K_test/Gaussian_Haslam_cl/haslam_gaussMap_cl_%d.txt" % i
        filename = s1+s2
        np.savetxt(filename, Map_cl, fmt='%0.6f')
        s1 = "/home/sandeep/final_Bispectrum/NonGuassian_Maps_Elsner2009"
        s2 = "/Gaussian_50K_test/Gaussian_Haslam_Maps/haslam_gaussMap_%d.fits" % i
        filename = s1+s2
        hp.fitsfunc.write_map(filename, Map)


if __name__ == "__main__":

    Cell_Count1 = Process(target=gaussian_maps, args=(0, 101))
    Cell_Count1.start()
    Cell_Count2 = Process(target=gaussian_maps, args=(101, 201))
    Cell_Count2.start()
    Cell_Count3 = Process(target=gaussian_maps, args=(201, 301))
    Cell_Count3.start()
    Cell_Count4 = Process(target=gaussian_maps, args=(301, 401))
    Cell_Count4.start()
    Cell_Count5 = Process(target=gaussian_maps, args=(401, 501))
    Cell_Count5.start()
    Cell_Count6 = Process(target=gaussian_maps, args=(501, 601))
    Cell_Count6.start()
    Cell_Count7 = Process(target=gaussian_maps, args=(601, 701))
    Cell_Count7.start()
    Cell_Count8 = Process(target=gaussian_maps, args=(701, 801))
    Cell_Count8.start()
    Cell_Count9 = Process(target=gaussian_maps, args=(801, 901))
    Cell_Count9.start()
    Cell_Count10 = Process(target=gaussian_maps, args=(901, 1001))
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

    esti_cl = np.zeros((1000, lmax), dtype=np.float32)
    s1 = "/home/sandeep/final_Bispectrum/NonGuassian_Maps_Elsner2009"
    s2 = "/Gaussian_50K_test/Gaussian_Haslam_cl/haslam_50K_cl.txt"
    name = s1+s2

    cl = np.genfromtxt(name)
    for i in xrange(0, 1000):
        s1 = "/home/sandeep/final_Bispectrum/NonGuassian_Maps_Elsner2009"
        s2 = "/Gaussian_50K_test/Gaussian_Haslam_cl/haslam_gaussMap_cl_%d.txt" % i
        name = s1+s2
        Map_cl = np.genfromtxt(name)
        esti_cl[i, :] = Map_cl

    mean = np.mean(esti_cl, 0)
    std_dev = np.std(esti_cl, 0)

    l = np.arange(lmax)
    plt.figure(1, figsize=(7, 7))

    plt.fill_between(l, l*(l+1)*(mean-std_dev), l*(l+1)*(mean+std_dev), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
    plt.plot(l, l*(l+1)*mean, '-', color='crimson', linewidth=2, label='mean Cl')
    plt.plot(l, l*(l+1)*cl, '-', color='orange', linewidth=2, label='original Cl')
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(which='both')
    plt.legend()
    plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
    plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
    plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)
    plt.savefig("/home/sandeep/final_Bispectrum/NonGuassian_Maps_Elsner2009/1000Gaussian_Cl_50K.eps",
                dpi=100)
    plt.show()
