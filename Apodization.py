"""
This code for Apodizing the mask to account for
Masking effect in power spectrum estimation
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from numba import jit
from matplotlib import rc,rcParams

@jit()
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

    for ipix in xrange(0, npix):
        theta, phi = hp.pixelfunc.pix2ang(nside, ipix)
        if 70. <= np.degrees(theta) <= 110:
                mask[ipix] = 0.0

    return mask


@jit()
def apodiz(mask, theta):
    apodiz_mask = hp.sphtfunc.smoothing(mask, fwhm=np.radians(theta))
    index = (apodiz_mask < 0)
    apodiz_mask[index] = 0.000
    return apodiz_mask

if __name__ == "__main__":
    name = '/dataspace/sandeep/Bispectrum_data/haslam408_dsds_Remazeilles2014.fits'
    print name

    NSIDE = 512
    NPIX = hp.pixelfunc.nside2npix(NSIDE)
    Haslam_512 = hp.fitsfunc.read_map(name)
    theta = 2.0
    LIMIT = 0.000162 # for gaussian 2 degree apodization 50 K
    Binary_mask = masking_map(Haslam_512, NSIDE, NPIX, LIMIT)
    imp_map_50 = apodiz(Binary_mask, theta)

    LIMIT = 0.000122  # for gaussian 2 degree apodization 30K
    Binary_mask = masking_map(Haslam_512, NSIDE, NPIX, LIMIT)
    imp_map_30 = apodiz(Binary_mask, theta)

    LIMIT = 0.000073
    Binary_mask = masking_map(Haslam_512, NSIDE, NPIX, LIMIT)
    imp_map_18 = apodiz(Binary_mask, theta)

    #LIMIT = 0.0002553  # 2 degree apodization
    LIMIT = 0.05553  # 2 degree apodization
    Binary_mask = masking_map(Haslam_512, NSIDE, NPIX, LIMIT)
    imp_map_200 = apodiz(Binary_mask, theta)

    Haslam_18 = Haslam_512*imp_map_18
    Haslam_200 = Haslam_512*imp_map_200
    Haslam_50 = Haslam_512*imp_map_50
    Haslam_30 = Haslam_512*imp_map_30

    LMAX = 300
    l = np.arange(0, 301)
    cl_18 = hp.sphtfunc.anafast(Haslam_18, lmax=LMAX)
    cl_200 = hp.sphtfunc.anafast(Haslam_200, lmax=LMAX)
    cl_50 = hp.sphtfunc.anafast(Haslam_50, lmax=LMAX)
    cl_30 = hp.sphtfunc.anafast(Haslam_30, lmax=LMAX)

#    hp.mollview(imp_map_18, xsize=2000, unit=r'$T_{B}(K)$', nest=False)
#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_mask_512_18K.eps",
#                dpi=100)
    #hp.fitsfunc.write_map("/home/sandeep/final_Bispectrum/imp_haslam_512_18K.fits", imp_map_18)

#    hp.mollview(imp_map_50, xsize=2000, unit=r'$T_{B}(K)$', nest=False)
#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_mask_512_50K.eps",
#                dpi=100)
    #hp.fitsfunc.write_map("/home/sandeep/final_Bispectrum/imp_haslam_512_50K.fits", imp_map_50)

#    hp.mollview(imp_map_200, xsize=2000, unit=r'$T_{B}(K)$', nest=False)
#    hp.mollview(imp_map_30, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_mask_512_200K.eps",
#                dpi=100)
    #hp.fitsfunc.write_map("/home/sandeep/final_Bispectrum/imp_haslam_512_200K.fits", imp_map_200)

    hp.mollview(Haslam_18, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_map_18.eps",
#                dpi=100)

    hp.mollview(Haslam_50, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_map_50.eps",
#                dpi=100)

    hp.mollview(Haslam_200, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

    hp.mollview(Haslam_30, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_map_200.eps",
#                dpi=100)

#    hp.mollview(Haslam_512, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

    plt.figure(8, figsize=(7, 7))
    plt.plot(l, l * (l + 1) * cl_18, '-', color='crimson', linewidth=2, label='18K')
    plt.plot(l, l * (l + 1) * cl_50, '-', color='orangered', linewidth=2, label='50K')
    plt.plot(l, l * (l + 1) * cl_200, '-', color='indigo', linewidth=2, label='200K')
    plt.plot(l, l * (l + 1) * cl_30, '-', color='green', linewidth=2, label='30K')
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(which='both')
    plt.legend()
    plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
    plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
    plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)
    #plt.savefig("/home/sandeep/final_Bispectrum/AllCl.eps",
    #            dpi=100)

    plt.show()



