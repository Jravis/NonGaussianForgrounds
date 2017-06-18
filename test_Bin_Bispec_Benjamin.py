"""
This is a Python code for Bispectrum on any scalar(Temprature only) map
we use Binned bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2,
"""

import numpy as np
import healpy as hp
from numba import njit


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
            if abs(l2-l1) <= l3 <= l2+l1 and (l3+l2+l1) % 2 == 0:
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
def summation(arr1, arr2, arr3,  num_pix):
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
        product = arr1[ipix]*arr2[ipix]*arr3[ipix]
        bi_sum += product
    bi_sum /= (4.0*np.pi*num_pix)
    return bi_sum


filename = '/dataspace/sandeep/Bispectrum_data/fnl_test/Elsner_alm/alm_l_0001_v3.fits'
filename1 = '/dataspace/sandeep/Bispectrum_data/fnl_test/Elsner_alm/alm_nl_0001_v3.fits'

alm_1 = hp.read_alm(filename)
alm_nl_1 = hp.read_alm(filename1)
map_20 = hp.alm2map((alm_1 + 20 * alm_nl_1), 1024)

lmax = 2500

"""
cl = hp.anafast(map_20, lmax=lmax)
ell = np.arange(len(cl))
# map_save = hp.mollview(map_20*1e6*2.7255, min=-500, max=500, unit="$\mu K$") # gives me the attached figure.
plt.figure(1)
plt.plot(ell, ell * (ell+1) * cl, color='crimson')
plt.xlabel('ell'); plt.ylabel('ell(ell+1)cl'); plt.grid()
plt.savefig("/home/sandeep/Benjamin_Test_cl.eps", dpi=100)
plt.show()
"""

nside_f_est = 1024

npix = hp.nside2npix(nside_f_est)


index = [2, 4, 10, 18, 30, 40, 53, 71, 99, 126, 154, 211, 243, 281, 309, 343, 378, 420, 445, 476, 518, 549, 591, 619
         , 659, 700, 742, 771, 800, 849, 899, 931, 966, 1001, 1035, 1092, 1150, 1184, 1230, 1257, 1291, 1346, 1400, 1460
         , 1501, 1520, 1540, 1575, 1610, 1665, 1725, 1795, 1846, 1897, 2001, 2091, 2240, 2500]


nbin = len(index)
#print "Total number of bins %d :-" % nbin
index = np.asarray(index, dtype=np.int32)
print ""
print index
# creating filtered map using equation 6 casaponsa et al. and eq (2.6) in Bucher et.al 2015

bin_arr = np.zeros((nbin-1, 2), dtype=int)
esti_map = np.zeros((nbin, npix), dtype=np.double)
filtered_map = np.zeros(hp.nside2npix(nside_f_est), dtype=np.float64)

for i in xrange(0, nbin):
    alm_obs = hp.sphtfunc.map2alm(map_20, lmax=lmax, iter=3)
    window_func = np.zeros(lmax, dtype=np.float32)
    ini = index[i]
    if i+1 < nbin:
        final = index[i+1]
        bin_arr[i, 0] = ini
        bin_arr[i, 1] = final
        for j in xrange(ini, final):  # Summing over all l in a given bin
            window_func[j] = 1.0
        alm_true = hp.sphtfunc.almxfl(alm_obs, window_func, mmax=None, inplace=True)
        esti_map[i, :] = hp.sphtfunc.alm2map(alm_true, nside_f_est, verbose=False)*2.7522

print bin_arr

s1 = '/home/sandeep/Benjamin_test/'
s2 = 'Ben_1_Analysis_Bin_Bispectrum_%d.txt' % nside_f_est
file_name = s1+s2
with open(file_name, 'w') as f:
    f.write("Bis\ti\tj\tk\tcount\n")
    for i in xrange(0, nbin - 1):
        for j in xrange(i, nbin-1):
            for k in xrange(j, nbin-1):
                #if np.min(bin_arr[k]) - np.max(bin_arr[j]) <= np.max(bin_arr[i]) <= np.max(bin_arr[k]) + np.max(bin_arr[j]):
                bis = summation(esti_map[i, :], esti_map[j, :], esti_map[k, :], npix)
                trip_count = count_triplet(bin_arr[i,:], bin_arr[j,:], bin_arr[k,:])
                f.write("%0.6e\t%d\t%d\t%d\t%d\n" % (bis, i, j, k, trip_count))



