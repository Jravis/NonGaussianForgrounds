"""
This code for Apodizing the mask to account for
Masking effect in power spectrum estimation
"""
import os
import numpy as np
import healpy as hp
import sys
sys.path.insert(0, '/home/sandeep/software/PolSpice_v03-03-02/src/')
import ispice

import ephem
import matplotlib.pyplot as plt
import getopt
#import matplotlib
#import matplotlib.pyplot as plt

use_planck_cmap = True
cmap1 = None


#matplotlib.use("agg")
"""
if use_planck_cmap:
        ############### CMB colormap
    from matplotlib.colors import ListedColormap
    colombi1_cmap = ListedColormap(np.loadtxt("/dataspace/sandeep/Bispectrum_data/Planck_Parchment_RGB.txt")/255.)
    colombi1_cmap.set_bad("gray") # color of missing pixels
    colombi1_cmap.set_under("white") # color of background, necessary if you want to use
    # this colormap directly with
    # hp.mollview(m, cmap=colombi1_cmap)
    cmap1 = colombi1_cmap
"""


def loadMap(filename):
    """
    Given a filename, load and return the HEALPix map.
    """

    if os.path.splitext(filename)[1] == '.healnpy':
        map = np.load(filename)
    elif os.path.splitext(filename)[1] == '.fits':
        map = hp.fitsfunc.read_map(filename)

    elif os.path.splitext(filename)[1] == '.txt':
        map = np.genfromtxt(filename, dtype=np.float64)

    return map


def getMapValue(map, ra, dec, theta):
    """
    Given a HEALPix map and a right ascension/declianation pair, return
    the map value at that point.  Use the 'fhwm' keyword to provide the
    beam full width at have max in degrees
    """

    nSide = hp.pixelfunc.npix2nside(map.size)
    # Extract the region around the source
    vec = hp.pixelfunc.ang2vec(np.pi / 2 - np.deg2rad(dec) , np.deg2rad(ra))
    vec = np.array(vec)
    #innerPixels = hp.query_disc(nSide, vec, radius=np.radians(1.5*56./60.))
    innerPixels = hp.query_disc(nSide, vec, radius=np.radians(theta))
    return innerPixels


def masking_map(map1, nside, npix, limit, Galcut):

    mask = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    index = (map1< limit) 
    mask[index] = 1.0

    if Galcut == 'Y':
        for ipix in xrange(0, npix):
            theta1, phi = hp.pixelfunc.pix2ang(nside, ipix)
            if 75. < np.degrees(theta1)< 105.0:
                mask[ipix] = 0.0

    inner_pix = getMapValue(map1,329.6, 17.5, 54.0)
    outer_pix = getMapValue(map1,329.6, 17.5, 62.0)
    index = np.setdiff1d(outer_pix, inner_pix)
    index1 = []
    for ipix1 in  index:
        theta, phi = hp.pixelfunc.pix2ang(nside, ipix1)
        if np.degrees(theta) < 90.0:
            if 0.0 < np.degrees(phi)< 60.0:
                index1.append(ipix1)
            if 320.0 < np.degrees(phi)< 360.0:
                index1.append(ipix1)

    index1=np.asarray(index1)
    mask[index1]=0.0
        
#    map1[index] = 0.00
#    map1[index1] = 0.00
    return mask


def apodiz(mask, theta):
    apodiz_mask = hp.sphtfunc.smoothing(mask, fwhm=theta,
                                        verbose=False)
    ind = (apodiz_mask < 0.0)
    apodiz_mask[ind] = 0.0

    return apodiz_mask



def main(fname, NSIDE):

    input_map = loadMap(fname)
    Haslam_128 = hp.pixelfunc.ud_grade(input_map, nside_out=128)
    NPIX = hp.pixelfunc.nside2npix(NSIDE)
    key = ['50K']#, 30'40K', '50K', '60K']
    for fn in key:
        #fname1 = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_binary_512/Mask_%s_binary_ns_512.fits' % fn
        fname2 = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_512/Mask_%s_apod_300arcm_ns_512.fits' % fn

        #BMask_512 = loadMap(fname1)
        ApodMask_512 = loadMap(fname2)

        #hp.mollview(BMask_512, xsize=2000, coord=['G'], unit=r'$T_{B}(K)$', nest=False)
        hp.mollview(input_map*ApodMask_512, xsize=2000, coord=['G'],
                    unit=r'$T_{B}(K)$', nest=False, title='408 MHz,%s' %
                    '46K')

        f_name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/maps_512/Map_%s_apod_300arcm_ns_512.fits' % fn
        hp.fitsfunc.write_map(f_name, input_map*ApodMask_512)
        f_name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/maps_512/Map_%s_apod_300arcm_ns_512.pdf' % fn
        plt.savefig(f_name, dpi=300)


        #fname1 = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_binary_128/Mask_%s_binary_ns_128.fits' % fn
        fname2 = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits' % fn

        #BMask_128 = loadMap(fname1)
        ApodMask_128 = loadMap(fname2)

        #hp.mollview(BMask_128, xsize=2000, coord=['G'], unit=r'$T_{B}(K)$', nest=False)
        hp.mollview(Haslam_128*ApodMask_128, xsize=2000, coord=['G'],
                    unit=r'$T_{B}(K)$', nest=False, title='408 MHz,%s' %
                    '46K')


        f_name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/maps_128/Map_%s_apod_300arcm_ns_128.fits' % fn
        hp.fitsfunc.write_map(f_name, Haslam_128*ApodMask_128)

        f_name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/maps_128/Map_%s_apod_300arcm_ns_128.pdf' % fn
        plt.savefig(f_name, dpi=300)

if __name__ == "__main__":

    filename = '/dataspace/sandeep/Bispectrum_data/haslam408_dsds_Remazeilles2014.fits'
    main(filename, 512)
"""
    dpi= 300
    figsize_inch = 10, 8
    fig = plt.figure(figsize=figsize_inch, dpi=dpi)
    hp.mollview(haslam, xsize=2000, cbar=True, cmap=cmap1,  unit=r'$T_{B}(K)$', nest=False, title='408 MHz,%s' % TEMP[0])
    fname = '/dataspace/sandeep/Bispectrum_data/Input_Maps/Map_ns128_300arcmnApod_%s.png'%TEMP[0]

    fig.savefig(fname)
    """


plt.show()


