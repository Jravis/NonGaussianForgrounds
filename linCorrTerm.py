"""
This code is to compute Linear correction term for Bispectrum to 
take care of beam effect.
"""

import numpy as np 
import healpy as hp
from multiprocessing import Process
import pywigxjpf as wig
from numba import njit
import _countTriplet

name = '/dataspace/sandeep/Bispectrum_data/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)
lmax = 256


#@njit()
def count_triplet(bin_1, bin_2, bin_3):
    """
    This routine count number of valid l-triplet in a i-trplet bin
    which we use to evaluate average
    :param bin_min:
    :param bin_max:
    :return:
    """
    count = 0
    for l3 in xrange(bin_1[0], bin_1[1]+1):
        for l2 in xrange(bin_2[0], bin_2[1]+1):
            for l1 in xrange(bin_3[0], bin_3[1]+1):
                if (l3+l2+l1) % 2 == 0:
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
def summation(arr1, arr2, f_sky):
    """
    :param arr1:
    :param arr2:
    :param arr3:
    :param arr4:
    :param num_pix:
    :return:
    """
    bi_sum = np.sum(np.multiply( arr1, arr2))

    bi_sum /= (4.0*np.pi*f_sky)
    return bi_sum

def Gauss_Avg(loop1, BinJ, BinK, Apo_map, gal_mask, ns):

    Gauss_esti_map = np.zeros((1000, npix), dtype=np.double)

    window_func_J = np.zeros(lmax, float)
    for j in xrange(BinJ[0], BinJ[1]):
        window_func[j] = 1.0
    
    window_func_K = np.zeros(lmax, float)
    for j in xrange(BinK[0], BinK[1]):
        window_func[j] = 1.0

    for fn in xrange(0, 1000):

        s1 = '/dataspace/sandeep/Bispectrum_data'
        s2 = '/Gaussian_%s_test/Gaussian_%s_Maps/haslam_%sgaussMap_%d.fits'% (loop1, loop1, loop1, fn)

        filename = s1+s2
        Gauss_map = hp.fitsfunc.read_map(filename)*gal_mask
        alm_obs = hp.sphtfunc.map2alm(Gauss_map, lmax=lmax, iter=3)
            
        alm_obs_J = hp.sphtfunc.almxfl(alm_obs, window_func_J, mmax=None, inplace=True)
        test_map_J = hp.sphtfunc.alm2map(alm_obs_J, ns, verbose=False)

        alm_obs_K = hp.sphtfunc.almxfl(alm_obs, window_func_K, mmax=None, inplace=True)
        test_map_K = hp.sphtfunc.alm2map(alm_obs_K, ns, verbose=False)

        a = test_map_J*Apo_mask
        b = test_map_K*Apo_mask
        Gauss_esti_map[fn, :] = np.multiply(a,b)

    mean = np.mean(Gauss_esti_map, axis=0)

    return mean



def bispec_estimator(nside_f_est, loop, apod_mask):
    """
    :param nside_f_est:
    :param loop:
    :param limit:
    :return:
    """

# Masking and apodization

    npix = hp.nside2npix(nside_f_est)
    Haslam_128 = hp.pixelfunc.ud_grade(Haslam_512, nside_out=128)

    haslam = Haslam_128 * gal_cut_mask


    nbin = 11
    index = np.logspace(np.log10(10), np.log10(256), nbin, endpoint=True, dtype=np.int32)
    ind = (index!=11)
    index=index[ind]

    # using Logrithmic bins
    esti_map = np.zeros((nbin, npix), dtype=np.double)
    bin_arr = np.zeros((nbin - 1, 2), dtype=np.int32)

    for i in xrange(0, nbin):
        alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)
        window_func = np.zeros(lmax, float)
        ini = index[i]
        if i+1 < nbin:
            final = index[i + 1]
            bin_arr[i, 0] = ini
            bin_arr[i, 1] = final - 1

            for j in xrange(ini, final):  # Summing over all l in a given bin
                window_func[j] = 1.0


            alm_obs = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=True)
            test_map = hp.sphtfunc.alm2map(alm_obs, nside_f_est, verbose=False)
            esti_map[i, :] = test_map*apod_mask


    frac_sky = np.sum(apod_mask)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_%s_test/'% (loop)
    s2 = 'All_mode/Analysis_Bin_Bispectrum_%d_%s.txt' % (nside_f_est, loop)

    file_name = s1+s2
    print file_name

    with open(file_name, 'w') as f:

        f.write("Bis\tI1\tI2\tI3\tcount\n")
        for I1 in xrange(0, nbin - 1):
            for I2 in xrange(I1, nbin-1):
                for I3 in xrange(I2, nbin-1):
                   
                    foo = [bin_arr[I2, 0], bin_arr[I2, 1]+1]
                    bar = [bin_arr[I3, 0], bin_arr[I3, 1]+1]

                    bis1 = summation(esti_map[I1, :],Gauss_Avg(loop, foo, bar,
                                    apod_mask,gal_cut_mask, nside_f_est), frac_sky)

                    f.write("%0.6e\t%d\t%d\t%d\n" % (bis, I1, I2, I3))


if __name__ == "__main__":

    NSIDE = 128
    TEMP = ['40K']#, 30'40K', '50K', '60K']

    f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits" % TEMP[0]
    print f_name1
    ap_mask_128 = hp.fitsfunc.read_map(f_name1)

    Cell_Count1 = Process(target=bispec_estimator, args=(NSIDE, TEMP[0],
                          ap_mask_128))
    Cell_Count1.start()
    Cell_Count1.join()


