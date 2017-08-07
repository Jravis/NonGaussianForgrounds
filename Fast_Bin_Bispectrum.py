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
import _countTriplet

name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)

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
def summation(arr1, arr2, arr3, frac_sky):
    """
    :param arr1:
    :param arr2:
    :param arr3:
    :param arr4:
    :param num_pix:
    :return:
    """
    bi_sum = 0.0
    foo = np.multiply(arr1, arr2)

    bi_sum = np.sum(np.multiply(foo, arr3))
    bi_sum /= (4.0*np.pi*frac_sky)

    return bi_sum


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

    haslam = Haslam_128 * ap_map # Galactic mask to reduce leakage for low temprature map 


    # using Logrithmic bins
    lmax = 256
    nbin = 11

    index = np.logspace(np.log10(10), np.log10(256), nbin, endpoint=True, dtype=np.int32)
    ind = (index!=11)
    index=index[ind]
    print index
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

            for j in xrange(ini, final):  # Summing over all l in a given bin we are using top-hat filter
                window_func[j] = 1.0      # will use smoothing technique to stop ripple

            alm_obs = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=True)
            test_map = hp.sphtfunc.alm2map(alm_obs, nside_f_est, verbose=False)
            
            esti_map[i, :] = test_map*apod_mask


    npix = np.sum(apod_mask)
    s1 = '/dataspace/sandeep/Bispectrum_data/Gaussian_%s_test/'% (loop)
    s2 = 'All_mode/Analysis_Bin_Bispectrum_%d_%s.txt' % (nside_f_est, loop)

    file_name = s1+s2
    print file_name

    with open(file_name, 'w') as f:

        f.write("Bis\tI1\tI2\tI3\tcount\n")

        #we will have l3>=l2>=l1 scheme
        
        for I1 in xrange(0, nbin - 1):
            for I2 in xrange(I1, nbin-1):
                for I3 in xrange(I2, nbin-1):

                    bis = summation(esti_map[I1, :], esti_map[I2, :], esti_map[I3, :], , npix)
                   
#                   trip_count = count_triplet(bin_arr[I1, :], bin_arr[I2, :], bin_arr[I3, :])
#                    trip_count = _countTriplet.countTriplet(bin_arr[i, :], bin_arr[j, :], bin_arr[k, :])
                    #f.write("%0.6e\t%d\t%d\t%d\t%d\n" % (bis, i, j, k, trip_count))

# If considering all mode

                    f.write("%0.6e\t%d\t%d\t%d\n" % (bis, I1, I2, I3))


if __name__ == "__main__":

    NSIDE = 128
    TEMP = ['40K']#, 30'40K', '50K', '60K']

    f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits" % TEMP[0]
    print f_name1
    ap_mask_128 = hp.fitsfunc.read_map(f_name1)

    Cell_Count1 = Process(target=bispec_estimator, args=(NSIDE, TEMP[0], ap_mask_128))
    Cell_Count1.start()
    Cell_Count1.join()


