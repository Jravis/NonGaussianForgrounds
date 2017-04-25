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
name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)

wig.wig_table_init(1000)
wig.wig_temp_init(1000)


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
    width = m.radians(2.0)
    apodiz_mask = hp.sphtfunc.smoothing(mask, fwhm=width)
    index = (apodiz_mask < 0)
    apodiz_mask[index] = 0.000
    return apodiz_mask


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
# Masking and apodization
    npix = hp.nside2npix(nside_f_est)
    binary_mask = masking_map(Haslam_512, nside_f_est, npix, limit)
    ap_map = apodiz(binary_mask)
    haslam = Haslam_512 * ap_map


# binned map  equation(6) casaponsa et. al.
    lmax = 250
    nbin = 12
# using Logrithmic bins
    index = 10**np.linspace(np.log10(2), np.log10(251), nbin)  #logrithmic bins
    for i in xrange(len(index)):
        index[i] = int(index[i])
    print index
    # creating filtered map using equation 6 casaponsa et al. and eq (2.6) in Bucher et.al 2015

    esti_map = np.zeros((nbin, npix), dtype=np.double)
    fwhm = 56./3600.  # For Haslam FWHM is 56 arc min
    beam_l = hp.sphtfunc.gauss_beam(m.radians(fwhm), lmax=lmax, pol=False)
    filtered_map = np.zeros(hp.nside2npix(nside_f_est), dtype=np.float64)

    for i in xrange(0, nbin):
        alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)
        window_func = np.zeros(lmax, float)
        ini = int(index[i])
        if i+1 < nbin:
            final = int(index[i+1])
            for j in xrange(ini, final):# Summing over all l in a given bin
                window_func[j] = 1.0
                alm_obs = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=True)
                beam = 1./beam_l
                alm_obs = hp.sphtfunc.almxfl(alm_obs, beam, mmax=None, inplace=True)
                alm_true = alm_obs
                filtered_map += hp.sphtfunc.alm2map(alm_true, nside_f_est)

    esti_map[i, :] = filtered_map

    cl = hp.sphtfunc.anafast(haslam, lmax=lmax, iter=3)
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

if __name__ == "__main__":

    print "Enter the Nside to which you want to upgrade or degrade the given map"
    NSIDE = int(raw_input(""))
    print "NSIDE = %d" % NSIDE

    Cell_Count1 = Process(target=bispec_estimator, args=(NSIDE, 18, 0.000073))
    Cell_Count1.start()
    Cell_Count2 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.000162))
    Cell_Count2.start()
    Cell_Count3 = Process(target=bispec_estimator, args=(NSIDE, 200, 0.0002553))
    Cell_Count3.start()
    Cell_Count4 = Process(target=bispec_estimator, args=(NSIDE, 30, 0.000122))
    Cell_Count4.start()

    #Cell_Count1.join()
    #Cell_Count2.join()
    #Cell_Count3.join()
    Cell_Count4.join()




