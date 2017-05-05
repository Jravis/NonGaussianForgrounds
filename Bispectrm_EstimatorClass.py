import numpy as np
import healpy as hp
from multiprocessing import Process
import pywigxjpf as wig
from numba import njit
import math as m
from numba import jitclass
from CMB_binned_bispectrum import map_making

class binned_bispectrum:

    map_making.nside_f_est =512
    map_making.masking ='No'
    map_making.apodization = 'No'


    wig.wig_table_init(1000)
    wig.wig_temp_init(1000)

    # binned map  equation(6) casaponsa et. al.
    lmax = 250
    nbin = 12
    # using Logrithmic bins
    index = 10 ** np.linspace(np.log10(2), np.log10(251), nbin)  # logrithmic bins
    for i in xrange(len(index)):
        index[i] = int(index[i])
    print index
    def __init__(self):


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
            for l2 in xrange(bin_min, l3+1):
                for l1 in xrange(bin_min, l2+1):
                    if abs(l2-l1) <= l3 <= l2+l1 and (l3+l2+l1) % 2 == 0:  # we applied selection condition tirangle inequality and#parity condition
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
    def summation(arr1, arr2, arr3, arr4, num_pix):
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
            product = arr1[ipix]*arr2[ipix]*arr3[ipix]*arr4[ipix]
            bi_sum += product
        bi_sum /= (4.0*np.pi*np.sum(arr4))
        return bi_sum



    def bispec_estimator(nside_f_est, loop, limit):
        """
        :param nside_f_est:
        :param loop:
        :param limit:
        :return:
        """
        # creating filtered map using equation 6 casaponsa et al. and eq (2.6) in Bucher et.al 2015

        cl = hp.sphtfunc.anafast(map_making, lmax=lmax, iter=3)
        bin_cl = []

        # Var have cl1*cl2*cl3  binned in l equation (2.11) in Bucher et al 2015
        for i in xrange(0, nbin):
            cl_sum = 0.0
            ini = int(index[i])
            if i+1 < nbin:
                final = int(index[i+1])
                for j in xrange(ini, final):
                    cl_sum += cl[j]
                bin_cl.append(cl_sum)
            bin_cl = np.asarray(bin_cl)

        s1 = '/home/sandeep/final_Bispectrum/DimensionlessQ_Bispec/Temp_Fluctuation'
        s2 = '/Rework_16April2017/DimensionLess_Bin_Bispectrum_%d_%d.txt' % (nside_f_est, loop)
        file_name = s1+s2

        with open(file_name, 'w') as f:
            f.write("Bis\tangAvg_Bis\tVarB\tCl1\tCl2\tCl3\ti1\ti2\ti3\tTripCount\n")
            for i in xrange(0, nbin):
                for j in xrange(0, i+1):
                    for k in xrange(0, j+1):
                        i3 = index[i]
                        i2 = index[j]
                        i1 = index[k]
                        if abs(i2-i1) <= i3 <= i2+i1 and (i3+i2+i1) % 2 == 0:
                        b = [2*i1, 2*i2, 2*i3, 2*0, 2*0, 2*0]
                        wigner = wig.wig3jj(b)
                        alpha = np.sqrt(((2*i1+1) * (2*i2+1) * (2*i3+1)) / (4.*np.pi)) * wigner
                        bis = summation(esti_map[i, :], esti_map[j, :], esti_map[k, :], ap_map, npix)
                        ang_avg_bis = bis/alpha
                        trip_count = count_triplet(i1, i3)
                        if trip_count != 0.:
                            bis /= (1.0*trip_count)
                            var_bis = (g(i1, i2, i3)/trip_count**2)*alpha**2*bin_cl[i]*bin_cl[j]*bin_cl[k]
                            f.write("%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e\t%d\t%d\t%d\n" % (bis, ang_avg_bis,
                                    var_bis, bin_cl[i], bin_cl[j], bin_cl[k], i3, i2, i1))
