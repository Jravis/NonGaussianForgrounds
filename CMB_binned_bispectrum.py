"""
This is a Python code for Bispectrum on any scalar(Temprature only) map
we use Binned bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2,
"""

import numpy as np
import healpy as hp
import pywigxjpf as wig
from numba import njit
import math as m

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
        for l2 in xrange(bin_min, l3 + 1):
            for l1 in xrange(bin_min, l2 + 1):
                if abs(l2 - l1) <= l3 <= l2 + l1 and (
                        l3 + l2 + l1) % 2 == 0:  # we applied selection condition tirangle inequality and#parity condition
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
def summation( arr1, arr2, arr3, arr4, num_pix):
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
        product = arr1[ipix] * arr2[ipix] * arr3[ipix] * arr4[ipix]
        bi_sum += product
    bi_sum /= (4.0 * np.pi * np.sum(arr4))
    return bi_sum

class map_making:
    """
    """
    def __init__(self, nside_f_est, masking=False, apodization=False, mask_limit=0.0
                 , ap_angle=2.0):
        """
        :param fnl:
        :param nside_f_est:
        :param alm_g:
        :param alm_ng:
        :param masking:
        :param apodization:
        :param mask_limit:
        :param ap_angle:
        """

        self.nside_f_est = nside_f_est
        self.masking = masking
        self.apodization = apodization
        self.mask_limt = mask_limit
        self.ap_angle = ap_angle
        self.npix = hp.nside2npix(self.nside_f_est)

    def masking_map(self, map1, nside, npix, limit):
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

    def apodiz(self, mask, angle):
        """
        :param mask: Input Mask
        :param angle: angle to do apodization e.g 2 degree or 5 dgree
        :return:
        """
        width = m.radians(angle)
        apodiz_mask = hp.sphtfunc.smoothing(mask, fwhm=width)
        index = (apodiz_mask < 0)
        apodiz_mask[index] = 0.000
        return apodiz_mask

    def simulated_map(self, type, fnl, alm_g, alm_ng):
        """
        :param type:
        :return:
        """

        alm = alm_g + fnl*alm_ng
        if type == 'gaussian':
            in_map = hp.sphtfunc.alm2map(self.alm_g, self.nside_f_est, verbose=False)
        if type == 'Non gaussian':
            in_map = hp.sphtfunc.alm2map(alm, self.nside_f_est, verbose=False)
        ap_map = np.ones(self.npix, dtype=np.float)
        if self.masking:
            binary_mask = self.masking_map(in_map, self.nside_f_est, self.npix, self.mask_limit)
        if self.apodization:
            ap_map = self.apodiz(binary_mask, self.ap_angle)
            in_map = in_map * ap_map

        return in_map, ap_map

    def binned_map(self, nbin, lmax, bin_l,  inp_fnl, inp_alm_l, inp_alm_nl, beam, map_type, beam_corr):
        """
        :param nbin:
        :param lmax:
        :param bin_l:
        :param beam:
        :param map_type:
        :param beam_corr:
        :return:
        """
        index = bin_l
        bin_arr = [[] for i in range(nbin-1)]

        esti_map = np.zeros((nbin, self.npix), dtype=np.double)

        if beam_corr:
            fwhm = beam
            beam_l = hp.sphtfunc.gauss_beam(m.radians(fwhm), lmax=lmax, pol=False)
        filtered_map = np.zeros(hp.nside2npix(self.nside_f_est), dtype=np.float64)
        in_map, ap_map = self.simulated_map(map_type, inp_fnl,inp_alm_l, inp_alm_nl)

        for i in xrange(0, nbin):
            alm_obs = hp.sphtfunc.map2alm(in_map, lmax=lmax, iter=3)
            window_func = np.zeros(lmax, float)
            ini = int(index[i])
            if i+1 < nbin:
                final = int(index[i+1])
                bin_arr[i].append(range(ini, final))
                for j in xrange(ini, final):# Summing over all l in a given bin
                    window_func[j] = 1.0
                    alm_obs = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=True)
                    if beam_corr:
                        beam = 1./beam_l
                        alm_obs = hp.sphtfunc.almxfl(alm_obs, beam, mmax=None, inplace=True)
                    alm_true = alm_obs
                    filtered_map += hp.sphtfunc.alm2map(alm_true, self.nside_f_est, verbose=False)
            esti_map[i, :] = filtered_map
        #print bin_arr
        return esti_map, ap_map, bin_arr


class binned_bispectrum:

    map_making.nside_f_est =512
    map_making.masking ='No'
    map_making.apodization = 'No'
    wig.wig_table_init(1000)
    wig.wig_temp_init(1000)
    # binned map  equation(6) casaponsa et. al.

    def __init__(self, Inp_alm_l, Inp_alm_nl, inp_bin, inp_fNL, NSIDE=512, LMAX=1024):
        self.Inp_alm_l = Inp_alm_l
        self.Inp_alm_nl = Inp_alm_nl
        self.inp_bin = inp_bin
        self.Nbins = len(inp_bin)
        self.LMAX = LMAX
        self.FNL = inp_fNL
        self.NSIDE = NSIDE
        test1 = map_making(self.NSIDE)
        self.esti_map, self.ap_ma, self.binL = test1.binned_map(self.Nbins, self.LMAX, self.inp_bin, self.FNL, self.Inp_alm_l,
                                                                self.Inp_alm_nl, beam=None, map_type='Non gaussian'
                                                                , beam_corr=None)

        self.nbin = len(inp_bin)
        self.npix = hp.nside2npix(self.NSIDE)
        self.bis = list()
        self.avg_bis = list()
        self.I = list()
        self.J = list()
        self.K = list()

    def bispectrum(self):
        """
        :rtype: object
        :return:
        """
        for i in xrange(0, self.nbin - 1):
            for j in xrange(i, self.nbin-1):
                for k in xrange(j, self.nbin-1):

                    if np.min(self.binL[k]) - np.max(self.binL[j]) <= np.max(self.binL[i]) <= np.max(self.binL[k]) + np.max(self.binL[j]):
                        temp = summation(self.esti_map[i, :], self.esti_map[j, :], self.esti_map[k, :], self.ap_ma,
                                              self.npix)
                        trip_count = count_triplet(np.min(self.binL[k]), np.max(self.binL[i]))
                        if trip_count != 0.:
                            self.avg_bis.append(temp / (1.0 * trip_count))
                            self.bis.append(temp)
                            self.I.append(i)
                            self.J.append(j)
                            self.K.append(k)

        self.bis = np.asarray(self.bis)
        self.avg_bis = np.asarray(self.avg_bis)
        self.I = np.asarray(self.I)
        self.J = np.asarray(self.J)
        self.K = np.asarray(self.K)

        return self.bis, self.avg_bis, self.I, self.J, self.K

    wig.wig_temp_free()
    wig.wig_table_free()



