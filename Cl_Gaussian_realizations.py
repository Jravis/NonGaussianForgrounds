"""
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import math as m
from multiprocessing import Process


def gaussian_maps(nmin, nmax):
    """
    :param nmin:
    :param nmax:
    :return:
    """

    np.random.seed(49390927)  # fixing random Seed

    key = ['60K']#, 30'40K', '50K', '60K']

    f_name = "/dataspace/sandeep/Bispectrum_data/haslam408_dsds_Remazeilles2014.fits"
    print f_name
    Haslam_512 = hp.fitsfunc.read_map(f_name)
    Haslam_128 = hp.pixelfunc.ud_grade(Haslam_512, nside_out=128)


    f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits" % key[0]
    print f_name1
    ap_map_128 = hp.fitsfunc.read_map(f_name1)

    inp_map_128 = Haslam_128*ap_map_128

    NSIDE_128 = 128

    cl = hp.sphtfunc.anafast(inp_map_128, iter=3, lmax=3*NSIDE_128-1)

    s1 = "/dataspace/sandeep/Bispectrum_data"
    s2 = "/Gaussian_%s_test/haslam_%s_ns128_cl.txt"%(key[0], key[0])

    #s2 = "/Niser_testing/haslam_allSky_cl.txt"
    name = s1+s2
    np.savetxt(name, cl, fmt='%0.6f')

    for i in xrange(nmin, nmax):

        f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/all_sky_polSpice/Haslam_spectra_fullsky_%s.fits" % key[0]
        PolSpi_cl = hp.fitsfunc.read_cl(f_name1)

        #Map = hp.sphtfunc.synfast(cl, NSIDE, lmax=250, pol=True, pixwin=False, fwhm=0.0, sigma=None, verbose=False)
        
        Map_128 = hp.sphtfunc.synfast(PolSpi_cl, NSIDE_128,
                                      lmax=3*NSIDE_128-1,pixwin=True,
                                      fwhm= np.radians(33.0/60.), verbose=False)

        Map1 = Map_128*ap_map_128 # no masking for now
        Map_cl = hp.sphtfunc.anafast(Map1, lmax=3*NSIDE_128-1, iter=3)

        s1 = "/dataspace/sandeep/Bispectrum_data"
        s2 = "/Gaussian_%s_test/Gaussian_%s_cl/haslam_%sgaussMap_cl_%d.txt" % (key[0], key[0], key[0], i)
        #s2 = "/Niser_testing/Gaussian_cl/haslam_gaussMap_cl_%d.txt" % i
        filename = s1+s2

        np.savetxt(filename, Map_cl, fmt='%0.6f')
        
        s1 = "/dataspace/sandeep/Bispectrum_data"
        s2 = "/Gaussian_%s_test/Gaussian_%s_Maps/haslam_%sgaussMap_%d.fits" % (key[0], key[0], key[0], i)
        #s2 = "/Niser_testing/Gaussian_Maps/haslam_gaussMap_%d.fits" % i

        filename = s1+s2
        hp.fitsfunc.write_map(filename, Map1)

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

    
"""
    lmax = 251
    s1 = "/dataspace/sandeep/Bispectrum_data"
    #s2 = "/Gaussian_18K_test/haslam_18K_cl.txt"
    s2 = "/Niser_testing/haslam_allSky_cl.txt"
    name = s1+s2
    cl = np.genfromtxt(name)

    esti_cl = np.zeros((1000, len(cl)), dtype=np.float32)
    for i in xrange(0, 1000):
        s1 = "/dataspace/sandeep/Bispectrum_data"
        #s2 = "/Gaussian_18K_test/Gaussian_18K_cl/haslam_18KgaussMap_cl_%d.txt" % i
        s2 = "/Niser_testing/Gaussian_cl/haslam_gaussMap_cl_%d.txt" % i
        filename = s1+s2
        Map_cl = np.genfromtxt(filename)
        esti_cl[i, :] = Map_cl

    mean = np.mean(esti_cl, 0)
    std_dev = np.std(esti_cl, 0)

    #l = np.arange(lmax)
    l = np.arange(len(cl))
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
    plt.savefig("/dataspace/sandeep/Bispectrum_data/Gaussian_Cl_allSky.eps", dpi=100)
plt.show()
"""
