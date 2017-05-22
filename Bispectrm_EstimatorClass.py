import numpy as np
import healpy as hp
from multiprocessing import Process
import CMB_binned_bispectrum as CB
bins = 10 ** np.linspace(np.log10(2), np.log10(1024), 25)

for i in xrange(len(bins)):
    bins[i] = int(bins[i])
bins = np.delete(bins, 0)


def code_test(nside, nmin, nmax):
    for fn in xrange(nmin, nmax):
        if fn < 10:
            filename = '/home/sandeep/final_Bispectrum/fnl_test/Elsner_alm/alm_l_000%d_v3.fits' % fn
            filename1 = '/home/sandeep/final_Bispectrum/fnl_test/Elsner_alm/alm_nl_000%d_v3.fits' % fn
        if 10 <= fn < 100:
            filename = '/home/sandeep/final_Bispectrum/fnl_test/Elsner_alm/alm_l_00%d_v3.fits' % fn
            filename1 = '/home/sandeep/final_Bispectrum/fnl_test/Elsner_alm/alm_nl_00%d_v3.fits' % fn
        if 100 <= fn < 1000:
            filename = '/home/sandeep/final_Bispectrum/fnl_test/Elsner_alm/alm_l_0%d_v3.fits' % fn
            filename1 = '/home/sandeep/final_Bispectrum/fnl_test/Elsner_alm/alm_nl_0%d_v3.fits' % fn

        els_alm_l = hp.fitsfunc.read_alm(filename)
        els_alm_nl = hp.fitsfunc.read_alm(filename1)
        test = CB.binned_bispectrum(els_alm_l, els_alm_nl, bins, 1.0, nside)
        bis, avg_bis, i, j, k = test.bispectrum()
        filename = '/home/sandeep/final_Bispectrum/fnl_test/900_fnl_1_Bispectrum/fnl_1_Bispectrum_%d.txt' % fn
        np.savetxt(filename, zip(bis, avg_bis, i, j, k), fmt='%0.6e,%0.6e,%d,%d,%d', delimiter=',',
                   header='bis,avg_bis,i,j,k')

if __name__ == "__main__":

    NSIDE = 512

    Cell_Count1 = Process(target=code_test, args=(NSIDE, 1, 31))
    Cell_Count1.start()

    Cell_Count2 = Process(target=code_test, args=(NSIDE, 31, 61))
    Cell_Count2.start()
    Cell_Count3 = Process(target=code_test, args=(NSIDE, 61, 91))
    Cell_Count3.start()
    Cell_Count4 = Process(target=code_test, args=(NSIDE, 91, 121))
    Cell_Count4.start()
    Cell_Count5 = Process(target=code_test, args=(NSIDE,  121, 151))
    Cell_Count5.start()
    Cell_Count6 = Process(target=code_test, args=(NSIDE, 151, 181))
    Cell_Count6.start()
    Cell_Count7 = Process(target=code_test, args=(NSIDE, 181, 211))
    Cell_Count7.start()
    Cell_Count8 = Process(target=code_test, args=(NSIDE, 211, 241))
    Cell_Count8.start()
    Cell_Count9 = Process(target=code_test, args=(NSIDE, 241, 271))
    Cell_Count9.start()
    Cell_Count10 = Process(target=code_test, args=(NSIDE, 271, 301))
    Cell_Count10.start()
    Cell_Count11 = Process(target=code_test, args=(NSIDE, 301, 331))
    Cell_Count11.start()
    Cell_Count12 = Process(target=code_test, args=(NSIDE, 331, 361))
    Cell_Count12.start()
    Cell_Count13 = Process(target=code_test, args=(NSIDE, 361, 391))
    Cell_Count13.start()
    Cell_Count14 = Process(target=code_test, args=(NSIDE, 391, 421))
    Cell_Count14.start()
    Cell_Count15 = Process(target=code_test, args=(NSIDE, 421, 451))
    Cell_Count15.start()
    Cell_Count16 = Process(target=code_test, args=(NSIDE, 451, 481))
    Cell_Count16.start()
    Cell_Count17 = Process(target=code_test, args=(NSIDE, 511, 541))
    Cell_Count17.start()
    Cell_Count18 = Process(target=code_test, args=(NSIDE, 541, 571))
    Cell_Count18.start()
    Cell_Count19 = Process(target=code_test, args=(NSIDE, 571, 601))
    Cell_Count19.start()
    Cell_Count20 = Process(target=code_test, args=(NSIDE, 601, 631))
    Cell_Count20.start()
    Cell_Count21 = Process(target=code_test, args=(NSIDE, 631, 661))
    Cell_Count21.start()
    Cell_Count22 = Process(target=code_test, args=(NSIDE, 661, 691))
    Cell_Count22.start()
    Cell_Count23 = Process(target=code_test, args=(NSIDE, 691, 721))
    Cell_Count23.start()
    Cell_Count24 = Process(target=code_test, args=(NSIDE, 721, 751))
    Cell_Count24.start()
    Cell_Count25 = Process(target=code_test, args=(NSIDE, 751, 781))
    Cell_Count25.start()
    Cell_Count26 = Process(target=code_test, args=(NSIDE, 781, 831))
    Cell_Count26.start()
    Cell_Count27 = Process(target=code_test, args=(NSIDE, 831, 861))
    Cell_Count27.start()
    Cell_Count28 = Process(target=code_test, args=(NSIDE, 861, 900))
    Cell_Count28.start()

    Cell_Count1.join()
    Cell_Count2.join()
    Cell_Count3.join()
    Cell_Count4.join()
    Cell_Count5.join()
    Cell_Count6.join()
    Cell_Count7.join()
    Cell_Count8.join()
    Cell_Count9.join()
    Cell_Count10.join()
    Cell_Count11.join()
    Cell_Count12.join()
    Cell_Count13.join()
    Cell_Count14.join()
    Cell_Count15.join()
    Cell_Count16.join()
    Cell_Count17.join()
    Cell_Count18.join()
    Cell_Count19.join()
    Cell_Count20.join()
    Cell_Count21.join()
    Cell_Count22.join()
    Cell_Count23.join()
    Cell_Count24.join()
    Cell_Count25.join()
    Cell_Count26.join()
    Cell_Count27.join()
    Cell_Count28.join()

