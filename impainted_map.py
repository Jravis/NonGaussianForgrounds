"""
This routine to create smooth maps with mask 
reffer Casponsa et al 2013 
"""


"""
This routine to create smooth maps with mask
reffer Casponsa et al 2013
"""

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
from multiprocessing import Pool, Process
from PyAstronomy import pyasl
from astropy.table import Table, Column
import pywigxjpf as wig


def masking_map(map1, nside, npix, limit):

    """
    This routine to apply mask that we decided using count in cell
    scheme.
    """
    area = hp.pixelfunc.nside2pixarea(nside, degrees=False)
    for ipix in xrange(0, npix):
        temp = map1[ipix]*area
        if temp > limit:
            map1[ipix] = hp.UNSEEN
        else:
            Binary_mask[ipix] = 1.0
    return map1, Binary_mask


def impainting(nside, masked_map, npix):
    for i in xrange(0, 1000):
        for ipix in xrange(0, npix):
            theta, phi = hp.pixelfunc.pix2ang(NSIDE, ipix, nest=False)
            if masked_map[ipix] == 0.0:
                hpxidx = hp.pixelfunc.get_all_neighbours(nside, theta=theta, phi=phi, nest=False)
                masked_map[ipix] = np.sum(masked_map[hpxidx])/(1.0*len(hpxidx))
    return masked_map

if __name__ == "__main__":
    name = "/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits"
    print name

    NSIDE = 128
    Binary_mask = np.zeros(hp.nside2npix(NSIDE), dtype=np.double)
    NPIX = hp.pixelfunc.nside2npix(NSIDE)
    l = np.arange(0, 3*NSIDE)
    Haslam_512 = hp.fitsfunc.read_map(name)
    LIMIT = 0.0128
    Haslam_128 = hp.pixelfunc.ud_grade(Haslam_512, nside_out=128, order_in='RING', order_out='RING')
    Haslam_128, Binary_mask = masking_map(Haslam_128, NSIDE, NPIX, LIMIT)
    Masked_haslam = Haslam_128*Binary_mask
    NPIX= hp.nside2npix(NSIDE)

    imp_map = impainting(NSIDE, Masked_haslam, NPIX)
    np.savetxt("/home/sandeep/final_Bispectrum/imp_haslam_128_200.txt", imp_map, fmt='%f')

    LMAX = (3.*NSIDE-1)
    Cl = hp.sphtfunc.anafast(imp_map, map2=None, nspec=None, lmax=None,
                             mmax=None, iter=3, alm=False, pol=False, use_weights=False,
                             datapath=None)
    hp.mollview(imp_map, xsize=2000, unit=r'$T_{B}(K)$', nest=False)
    plt.savefig("/home/sandeep/final_Bispectrum/128_200_impainted.eps",
                dpi=100)

    plt.figure(2, figsize=(8, 6))
    plt.plot(l, l * (l + 1) * Cl, '-', color='g', linewidth=2, label='')
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    plt.savefig("/home/sandeep/final_Bispectrum/128_200_imp_haslam.eps", dpi=100)

    hp.fitsfunc.write_map("/home/sandeep/final_Bispectrum/imp_haslam_128_200.fits", imp_map)
    plt.show()







