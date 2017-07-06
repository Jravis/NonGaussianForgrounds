"""
This is a Python code for Bispectrum on any scalar(Temprature only) map
we use Binned bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2,
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import math as m
from multiprocessing import Process

#name = '/dataspace/sandeep/Bispectrum_data/haslam408_dsds_Remazeilles2014.fits'
#Haslam_512 = hp.fitsfunc.read_map(name)
#lmax = 251
#NSIDE = 512

"""
def masking_map(map1, nside, npix, limit):

    mask = np.zeros(hp.nside2npix(nside), dtype=np.double)
    area = hp.pixelfunc.nside2pixarea(nside, degrees=False)
    for ipix in xrange(0, npix):
        temp = map1[ipix]*area
        if temp < limit:
            mask[ipix] = 1.0
    for ipix in xrange(0, npix):
        theta1, phi = hp.pixelfunc.pix2ang(nside, ipix)
        if 70. <= np.degrees(theta1) <= 110:
            mask[ipix] = 0.0
    return mask


def apodiz(mask):
    width = m.radians(2.0)
    apodiz_mask = hp.sphtfunc.smoothing(mask, fwhm=width)
    index = (apodiz_mask < 0)
    apodiz_mask[index] = 0.000
    return apodiz_mask
"""

def gaussian_maps(nmin, nmax):
    """
    :param nmin:
    :param nmax:
    :return:
    """

    np.random.seed(49390927)  # fixing random Seed
#    limit = 0.0002553  # 200
#    #limit = 0.000162   # 50k at 2 degree apodization
#    binary_mask = masking_map(Haslam_512, NSIDE, npix, limit)
#    ap_map = apodiz(binary_mask)
#    haslam = Haslam_512 * ap_map

    f_name = "/dataspace/sandeep/Bispectrum_data/Input_Maps/MaskedMap_%s_%0.1fdeg_apodi.fits" % ('18K', 2.0)
    print f_name
    f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/ApodizeBinaryMask_%s_%0.1fdeg_apodi.fits" % ('18K', 2.0)
    print f_name1

    inp_map = hp.fitsfunc.read_map(f_name)
    ap_map = hp.fitsfunc.read_map(f_name1)
    NSIDE = 512

    npix = hp.nside2npix(NSIDE)

    cl = hp.sphtfunc.anafast(inp_map, lmax=250, iter=3)
    s1 = "/dataspace/sandeep/Bispectrum_data"
    s2 = "/Gaussian_18K_test/haslam_18K_cl.txt"
    name = s1+s2
    np.savetxt(name, cl, fmt='%0.6f')

    for i in xrange(nmin, nmax):
        Map = hp.sphtfunc.synfast(cl, NSIDE, lmax=250, pol=True, pixwin=False, fwhm=0.0, sigma=None, verbose=False)
        Map1 = Map*ap_map
        Map_cl = hp.sphtfunc.anafast(Map1, lmax=250, iter=3)
        s1 = "/dataspace/sandeep/Bispectrum_data"
        s2 = "/Gaussian_18K_test/Gaussian_18K_cl/haslam_18KgaussMap_cl_%d.txt" % i
        filename = s1+s2
        np.savetxt(filename, Map_cl, fmt='%0.6f')
        s1 = "/dataspace/sandeep/Bispectrum_data"
        s2 = "/Gaussian_18K_test/Gaussian_18K_Maps/haslam_18KgaussMap_%d.fits" % i
        filename = s1+s2
        hp.fitsfunc.write_map(filename, Map)

if __name__ == "__main__":

    nmin = 0
    nmax = 0
    count = 0
    min_core = 1
    max_core = 20
    increment = 50
    str = []

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
        str[i] = Process(target=gaussian_maps, args=(nmin, nmax))
        str[i].start()
        count = nmax

    for i in xrange(len(str)):
        str[i].join()

    lmax = 251

    esti_cl = np.zeros((100, lmax), dtype=np.float32)
    s1 = "/dataspace/sandeep/Bispectrum_data"
    s2 = "/Gaussian_18K_test/haslam_18K_cl.txt"
    name = s1+s2
    cl = np.genfromtxt(name)

    for i in xrange(0, 100):
        s1 = "/dataspace/sandeep/Bispectrum_data"
        s2 = "/Gaussian_18K_test/Gaussian_18K_cl/haslam_18KgaussMap_cl_%d.txt" % i
        filename = s1+s2
        Map_cl = np.genfromtxt(filename)
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
    plt.savefig("/dataspace/sandeep/Bispectrum_data/Gaussian_18K_test/plots/1000Gaussian_Cl_18K.eps", dpi=100)
plt.show()

