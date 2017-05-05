"""
This is a Python code for Bispectrum on any scalar(Temprature only) map
we use Binned bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2,
"""

import numpy as np
import healpy as hp
from multiprocessing import Process
import pywigxjpf as wig
from numba import njit
import math as m
from numba import jitclass



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
            in_map = hp.sphtfunc.alm2map(self.alm_g, self.nside_f_est)
        if type == 'Non gaussian':
            in_map = hp.sphtfunc.alm2map(alm, self.nside_f_est)
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
        esti_map = np.zeros((nbin, self.npix), dtype=np.double)
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
                for j in xrange(ini, final):# Summing over all l in a given bin
                    window_func[j] = 1.0
                    alm_obs = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=True)
                    if beam_corr:
                        beam = 1./beam_l
                        alm_obs = hp.sphtfunc.almxfl(alm_obs, beam, mmax=None, inplace=True)
                    alm_true = alm_obs
                    filtered_map += hp.sphtfunc.alm2map(alm_true, self.nside_f_est)

            esti_map[i, :] = filtered_map
        return esti_map, ap_map



class binned_bispectrum:

    map_making.nside_f_est =512
    map_making.masking ='No'
    map_making.apodization = 'No'
    wig.wig_table_init(1000)
    wig.wig_temp_init(1000)
    # binned map  equation(6) casaponsa et. al.


    def __init__(self, Inp_alm_l, Inp_alm_nl, inp_bin,inp_FNL, NSIDE,LMAX=1024):
        self.Inp_alm_l = Inp_alm_l
        self.Inp_alm_nl = Inp_alm_nl
        self.inp_bin = inp_bin
        self.LMAX = LMAX
        self.FNL = inp_FNL
        self.esti_map_ng, self.ap_map_ng = map_making.binned_map(self.inp_bin, self.LMAXin_l,self.FNL, self.inp_alm_l, self.inp_alm_nl, beam=None, map_type='Non gaussian', beam_corr=None)
        self.esti_map_g, self.ap_map_g = map_making.binned_map(self.inp_bin, self.LMAXin_l,self.FNL, self.inp_alm_l, self.inp_alm_nl, beam=None, map_type='gaussian', beam_corr=None)
        self.cl = hp.sphtfunc.anafast(map_making, lmax=self.LMAX, iter=3)
        self.bin_cl = list()
        self.nbin = len(inp_bin)
        self.NSIDE = NSIDE
        self.npix = hp.nside2npix(self.NSIDE)
        self.bis = list()
        self.ang_avgbis = list()
        self.var_bis = list()



    @njit()
    def count_triplet(self, bin_min, bin_max):
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
    def g(self,l1, l2, l3):
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
    def summation(self, arr1, arr2, arr3, arr4, num_pix):
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

    @njit()
    def binned_cl(self):
        """

        :return:
        """
        index = self.inp_bin
        for i in xrange(0, self.nbin):
            cl_sum = 0.0
            ini = int(index[i])
            if i+1 < self.nbin:
                final = int(index[i+1])
                for j in xrange(ini, final):
                    cl_sum += self.cl[j]
                self.bin_cl.append(cl_sum)
        self.bin_cl = np.asarray(self.bin_cl)


    def bispectrum(self):
        """

        :return:
        """
        for i in xrange(0, self.nbin):
            for j in xrange(0, i+1):
                for k in xrange(0, j+1):
                    i3 = index[i]
                    i2 = index[j]
                    i1 = index[k]

                    if abs(i2-i1) <= i3 <= i2+i1 and (i3+i2+i1) % 2 == 0:

                        b = [2*i1, 2*i2, 2*i3, 2*0, 2*0, 2*0]
                        wigner = wig.wig3jj(b)
                        alpha = np.sqrt(((2*i1+1) * (2*i2+1) * (2*i3+1)) / (4.*np.pi)) * wigner

                        temp = self.summation(self.esti_map_ng[i, :], self.esti_map_ng[j, :], self.esti_map_ng[k, :], self.ap_map_ng, self.npix)
                        trip_count = self.count_triplet(i1, i3)
                        if trip_count != 0.:
                            temp /= (1.0*trip_count)
                            self.bis.append(temp)
                            self.ang_avg_bis.append(temp/alpha)
                            self.var_bis.append((self.g(i1, i2, i3) / trip_count ** 2) * alpha ** 2 * self.bin_cl[i] * self.bin_cl[j] * self.bin_cl[k])

        self.bis = np.asarray(self.bis)
        self.ang_avgbis = np.asarray(self.ang_avgbis)
        self.var_bis = np.asarray(self.var_bis)

        return self.bis, self.ang_avgbis, self.var_bis


    def linear_correction(self):

    def fnl_etimator(self):







if __name__ == "__main__":

    filename = '/home/sandeep/final_Bispectrum/NonGuassian_Maps_Elsner2009/alm_l_0001_v3.fits'
    els_alm_l = hp.fitsfunc.read_alm(filename, hdu=1, return_mmax=False)
    filename = '/home/sandeep/final_Bispectrum/NonGuassian_Maps_Elsner2009/alm_nl_0001_v3.fits'
    els_alm_nl = hp.fitsfunc.read_alm(filename, hdu=1, return_mmax=False)



    lmax = 250
    nbin = 12
    # using Logrithmic bins
    index = 10 ** np.linspace(np.log10(2), np.log10(251), nbin)  # logrithmic bins
    for i in xrange(len(index)):
        index[i] = int(index[i])
    print index


