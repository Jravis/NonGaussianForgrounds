"""
This code for Apodizing the mask to account for
Masking effect in power spectrum estimation
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib import rc,rcParams


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
    apodiz_mask = hp.sphtfunc.smoothing(mask, fwhm=np.radians(2.0))
    index = (apodiz_mask < 0)
    apodiz_mask[index] = 0.000
    return apodiz_mask

if __name__ == "__main__":
    name = "/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits"
    print name

    NSIDE = 512
    NPIX = hp.pixelfunc.nside2npix(NSIDE)

    Haslam_512 = hp.fitsfunc.read_map(name)

#    LIMIT = 0.0000793 # for gaussian 5 degree apodization 30K
#    LIMIT = 0.0000966 # for gaussian 5 degree apodization
#    LIMIT = 0.000162 # for gaussian 2 degree apodization
    LIMIT = 0.000122 # for gaussian 2 degree apodization 30K
#    LIMIT = 0.0002

    Binary_mask = masking_map(Haslam_512, NSIDE, NPIX, LIMIT)
    imp_map_50 = apodiz(Binary_mask)

#    LIMIT = 0.000071 # 5 degree apodization
    LIMIT = 0.000073
#    LIMIT = 0.000073

    Binary_mask = masking_map(Haslam_512, NSIDE, NPIX, LIMIT)
    imp_map_18 = apodiz(Binary_mask)

#    LIMIT = 0.0001809 # 5 degree apodization
    LIMIT = 0.0002553 # 2 degree apodization
#    LIMIT = 0.0008

    Binary_mask = masking_map(Haslam_512, NSIDE, NPIX, LIMIT)
    imp_map_200 = apodiz(Binary_mask)

    Haslam_18 = Haslam_512*imp_map_18
    Haslam_200 = Haslam_512*imp_map_200
    Haslam_50 = Haslam_512*imp_map_50

    LMAX = 250
    l = np.arange(0, 251)
    cl_18 = hp.sphtfunc.anafast(Haslam_18, lmax=LMAX)
    cl_200 = hp.sphtfunc.anafast(Haslam_200, lmax=LMAX)
    cl_50 = hp.sphtfunc.anafast(Haslam_50, lmax=LMAX)

#    hp.mollview(imp_map_18, xsize=2000, unit=r'$T_{B}(K)$', nest=False)
#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_mask_512_18K.eps",
#                dpi=100)
    #hp.fitsfunc.write_map("/home/sandeep/final_Bispectrum/imp_haslam_512_18K.fits", imp_map_18)

    hp.mollview(imp_map_50, xsize=2000, unit=r'$T_{B}(K)$', nest=False)
#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_mask_512_50K.eps",
#                dpi=100)
    #hp.fitsfunc.write_map("/home/sandeep/final_Bispectrum/imp_haslam_512_50K.fits", imp_map_50)

#    hp.mollview(imp_map_200, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_mask_512_200K.eps",
#                dpi=100)
    #hp.fitsfunc.write_map("/home/sandeep/final_Bispectrum/imp_haslam_512_200K.fits", imp_map_200)

#    hp.mollview(Haslam_18, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_map_18.eps",
#                dpi=100)

    hp.mollview(Haslam_50, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_map_50.eps",
#                dpi=100)

#    hp.mollview(Haslam_200, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

#    plt.savefig("/home/sandeep/final_Bispectrum/Haslam_map_200.eps",
#                dpi=100)

#    hp.mollview(Haslam_512, xsize=2000, unit=r'$T_{B}(K)$', nest=False)

    plt.figure(8, figsize=(7, 7))
    plt.plot(l, l * (l + 1) * cl_18, '-', color='crimson', linewidth=2, label='18K')
    plt.plot(l, l * (l + 1) * cl_50, '-', color='orangered', linewidth=2, label='50K')
    plt.plot(l, l * (l + 1) * cl_200, '-', color='indigo', linewidth=2, label='200K')
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



