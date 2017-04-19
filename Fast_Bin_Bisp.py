"""
This is a Python code for Bispectrum on any scalar(Temprature only) map
we use Binned bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2,
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
import math as m
import os
import healpy as hp
from multiprocessing import Pool, Process
from PyAstronomy import pyasl
import time
from astropy.table import Table, Column
# from Wigner3j import Wigner3j
import pywigxjpf as wig

name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)

wig.wig_table_init(1000)
wig.wig_temp_init(1000)


def masking_Map(map1, nside, npixs, limit):
    """
    This routine to apply mask that we decided using count in cell
    scheme.
    """
    theta = 0.0
    phi = 0.0
    area = hp.pixelfunc.nside2pixarea(nside, degrees=False)
    # print area
    B_mask = np.zeros(hp.nside2npix(nside), dtype=np.double)

    for ipix in xrange(0, npixs):
        theta, phi = hp.pixelfunc.pix2ang(nside, ipix)
        temp = map1[ipix] * area
        if temp > limit:  # for 47k
            map1[ipix] = hp.UNSEEN
        else:
            B_mask[ipix] = 1.0

    Average = np.sum(map1 * B_mask) / np.sum(B_mask)

    return map1, B_mask, Average


def Count_Triplet(bin_min, bin_max):
    """
    This routine count number of valid l-triplet in a i-trplet bin
    which we use to evaluate average
    """
    l3 = 0
    l2 = 0
    l1 = 0
    count = 0
    gaunt = 0.0

    alpha = 0.0
    b = [6]
    for l3 in xrange(bin_min, bin_max):
        for l2 in xrange(bin_min, l3 + 1):
            for l1 in xrange(bin_min, l2 + 1):
                if abs(l2 - l1) <= l3 <= l2 + l1 and (
                        l3 + l2 + l1) % 2 == 0:  # we applied selection condition tirangle inequality and#parity condition
                    count += 1
    return count


def g(l1, l2, l3):
    if ((l1 == l2) and (l2 == l3)):
        return 6.0
    elif ((l1 == l2) or (l2 == l3) or (l3 == l1)):
        return 2.0
    else:
        return 1.0


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Bispec_Estimator(NSIDE_f_est, loop, limit):
    npix = hp.pixelfunc.nside2npix(NSIDE_f_est)

    Haslam_nside = hp.pixelfunc.ud_grade(Haslam_512, nside_out=
    NSIDE_f_est, order_in='RING', order_out='RING')

    Haslam_nside = hp.sphtfunc.smoothing(Haslam_nside, fwhm=0.016307192977800353)

    Haslam, Binary_Mask, avg = masking_Map(Haslam_nside, NSIDE_f_est, npix,
                                           limit)

    for ipix in xrange(0, npix):
        if Haslam[ipix] != hp.UNSEEN:
            Haslam[ipix] = (Haslam[ipix] - avg) / avg

    LMAX = 250
    Nbin = 124  # 40
    print LMAX, Nbin

    Cl = hp.sphtfunc.anafast(Haslam, map2=None, nspec=None, lmax=None,
                             mmax=None, iter=3, alm=False, pol=False, use_weights=False,
                             datapath=None)

    Lbin_max = LMAX
    Lbin_min = 2  # 50

    Delta_i = (Lbin_max - Lbin_min) / Nbin

    print Delta_i

    l = Lbin_min

    Esti_Map = np.zeros((Nbin, npix), dtype=np.double)

    for i in xrange(0, Nbin):
        alm = hp.sphtfunc.map2alm(Haslam, lmax=LMAX, mmax=None, iter=3, pol=False, use_weights=False, datapath=None)
        window_func = np.zeros(LMAX, float)
        window_func[l] = 1.0

        hp.sphtfunc.almxfl(alm, window_func, mmax=None, inplace=True)

        cls = hp.sphtfunc.alm2cl(alm, alms2=None, lmax=None, mmax=None,
                                 lmax_out=None, nspec=None)

        Map = hp.sphtfunc.synfast(cls, NSIDE_f_est, lmax=None, mmax=None, alm=False,
                                  pol=True, pixwin=False, fwhm=0.016307192977800353, sigma=None, new=False,
                                  verbose=True)

        Esti_Map[i, :] = Map

        print l
        l += Delta_i

    temp = 2
    index = np.zeros(Nbin, int)
    for i in xrange(0, Nbin):
        index[i] = temp
        temp += 2  # 5

    i3 = 0
    i2 = 0
    i1 = 0
    name = '/home/sandeep/final_Bispectrum/DimensionlessQ_Bispec/Temp_Fluctuation/Rework_16April2017/DimensionLess_Bin_Bispectrum_%d_%d.txt' % (
    NSIDE_f_est, loop)

    with open(name, 'w')  as f:
        f.write("Bis\tangAvg_Bis\tnorm_bis\tVarB\tCl1\tCl2\tCl3\ti1\ti2\ti3\tTripCount\n")

        for i in xrange(0, Nbin):
            for j in xrange(0, i + 1):
                for k in xrange(0, j + 1):

                    i3 = index[i]
                    i2 = index[j]
                    i1 = index[k]
                    Bis = 0.0
                    reduce_bis = 0.0
                    tripCount = 0
                    wignerCoff = 0.0
                    VarB = 0.0

                    if abs(i2 - i1) <= i3 <= i2 + i1 and (i3 + i2 + i1) % 2 == 0:

                        b = [2 * i1, 2 * i2, 2 * i3, 2 * 0, 2 * 0, 2 * 0]
                        wigner = wig.wig3jj(b)
                        alpha = np.sqrt(((2 * i1 + 1) * (2 * i2 + 1) * (2 * i3 + 1)) / (4. * np.pi)) * wigner

                        for ipix in xrange(0, npix):
                            Bis += Esti_Map[i, ipix] * Esti_Map[j, ipix] * Esti_Map[k, ipix] * Binary_Mask[ipix]

                        Bis /= np.sum(Binary_Mask)
                        Bis *= 4.0 * np.pi

                        tripCount = Count_Triplet(i1, i3)
                        Bis /= (1.0 * tripCount)
                        angAvg_bis = Bis / alpha

                        norm_bis = abs(angAvg_bis) / (Cl[i1] * Cl[i2] * Cl[i1]) ** 0.5

                        VarB = g(i1, i2, i3) * (
                        ((2 * i1 + 1) * (2 * i2 + 1) * (2 * i3 + 1)) / (4. * np.pi)) * wigner ** 2 * Cl[i1] * Cl[i2] * \
                               Cl[i3]
                        VarB /= (1.0 * tripCount ** 2.0)

                        f.write("%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e\t%d\t%d\t%d\t%d\n" % (Bis, angAvg_bis,
                                                                                                       norm_bis, VarB,
                                                                                                       Cl[i1], Cl[i2],
                                                                                                       Cl[i3], i1,
                                                                                                       i2, i3,
                                                                                                       tripCount))


# ****************************************************************************************

if __name__ == "__main__":
    print "Enter the Nside to which you want to upgrade or degrade the given map"
    NSIDE = int(raw_input(""))
    print "NSIDE = %d" % NSIDE

    Cell_Count1 = Process(target=Bispec_Estimator, args=(NSIDE, 18, 0.0012))
    Cell_Count1.start()

    Cell_Count2 = Process(target=Bispec_Estimator, args=(NSIDE, 50, 0.0032))
    Cell_Count2.start()
    Cell_Count3 = Process(target=Bispec_Estimator, args=(NSIDE, 200, 0.0128))
    Cell_Count3.start()

    Cell_Count4 = Process(target=Bispec_Estimator, args=(NSIDE, 30, 0.00193))
    Cell_Count4.start()

    Cell_Count1.join()
    Cell_Count2.join()
    Cell_Count3.join()



