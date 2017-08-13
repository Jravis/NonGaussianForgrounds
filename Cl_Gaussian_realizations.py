"""
This is for creating 1000 Gaussian realization using masked map Cl APS
"""
import numpy as np
import healpy as hp
from multiprocessing import Process

np.random.seed(49390927)  # fixing random Seed


def gaussian_maps(nmin, nmax):

    """
    :param nmin: Min number index of map
    :param nmax: Max number index of map
    :return:
    """

    key = ['50K']#, 30'40K', '50K', '60K']

    f_name = "/dataspace/sandeep/Bispectrum_data/haslam408_dsds_Remazeilles2014.fits"

    haslam_512 = hp.fitsfunc.read_map(f_name)

    # Degrading Haslam from Nside 512 to 128

    haslam_128 = hp.pixelfunc.ud_grade(haslam_512, nside_out=128)

    # Reading 5 degree Apodized mask

    f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits" % key[0]
    print f_name1
    ap_map_128 = hp.fitsfunc.read_map(f_name1)

    # Masking All sky map with apodized mask

    inp_map_128 = haslam_128*ap_map_128

    NSIDE_128 = 128

    cl = hp.sphtfunc.anafast(inp_map_128, iter=3, lmax=3*NSIDE_128-1)

    s1 = "/dataspace/sandeep/Bispectrum_data"
    s2 = "/Gaussian_%s_test/haslam_%s_ns128_cl.txt" % (key[0], key[0])

    name = s1+s2
    np.savetxt(name, cl, fmt='%0.6f')

    for i in xrange(nmin, nmax):

        # Reading masked sky cl which have been computed using PolSpice

        f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/all_sky_polSpice/Haslam_spectra_fullsky_%s.fits" \
                  % key[0]

        PolSpi_cl = hp.fitsfunc.read_cl(f_name1)

        # Creating Gaussian maps of Nside 128 using PolSpice Cl smoothed with beam of 30 arcmin

        Map_128 = hp.sphtfunc.synfast(PolSpi_cl, NSIDE_128,
                                      lmax=3*NSIDE_128-1,pixwin=True,
                                      fwhm=np.radians(30.0/60.), verbose=False)

        Map1 = Map_128*ap_map_128  # no masking gaussian map
        Map_cl = hp.sphtfunc.anafast(Map1, lmax=3*NSIDE_128-1, iter=3)  # computing cl from gaussian map

        s1 = "/dataspace/sandeep/Bispectrum_data"
        s2 = "/Gaussian_%s_test/Gaussian_%s_cl/haslam_%sgaussMap_cl_%d.txt" % (key[0], key[0], key[0], i)
        filename = s1+s2
        np.savetxt(filename, Map_cl, fmt='%0.6f')
        
        s1 = "/dataspace/sandeep/Bispectrum_data"
        s2 = "/Gaussian_%s_test/Gaussian_%s_Maps/haslam_%sgaussMap_%d.fits" % (key[0], key[0], key[0], i)
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

