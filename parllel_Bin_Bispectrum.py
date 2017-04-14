"""
This is a Python code for Bispectrum on any scalar(Temprature only) map 
we use Binned Bispectrum estimator Bucher et. al. 2010 and
arXiv:1509.08107v2 also see Casponsa et. al. 2014
Extensively use of Healpy and Healpix thanks to Gorski et.al. 2005
Eric Hivon and Andrea Zonca for Healpy
"""

import numpy as np 
import healpy as hp
from multiprocessing import Process
# pywigxjpf for to import python routine to evaluate
# wigner 3j coefficient that comes in Guant integral
import pywigxjpf as wig

name = "/home/sandeep/Haslam/lambda_haslam408_dsds.fits"
print name
Haslam_512 = hp.fitsfunc.read_map(name) # Here we Use map from point source subtracted from lambda


def masking_map(map1, nside, npixs, limit):

    """
    This routine to apply mask that we decided using count in cell 
    scheme, we returning masked map and Binary Mask.
    """
    area = hp.pixelfunc.nside2pixarea(nside, degrees=False)
    b_mask = np.zeros(hp.nside2npix(nside), dtype=np.double)# Binary Mask

    for ipix in xrange(0, npixs):

        # area of pixel for given nside
        temp = map1[ipix]*area
        # for given Temperature level
        if temp > limit:
            # UNSEEN Healpy Native
            map1[ipix] = hp.UNSEEN
        else:
            b_mask[ipix] = 1.0

    return map1, b_mask


def count_triplet(bin_min, bin_max):
    """
    :param bin_min:
    :param bin_max:
    :return:  This routine count number of valid l-triplet in a i-trplet bin
    which we use to evaluate average
    """

    count = 0
    for l3 in xrange(bin_min, bin_max):
        for l2 in xrange(bin_min, l3+1):
            for l1 in xrange(bin_min, l2+1):
                # we applied selection condition tirangle inequality and#parity condition
                if abs(l2-l1) <= l3 <= l2+l1 and (l3+l2+l1)%2 ==0:
                    count += 1
    return count


def g(l1, l2, l3):
    """
    :param l1:
    :param l2:
    :param l3:
    :return: This function returns value for the different possible
    configuration in a triangle
    """

    if (l1 == l2) and (l2 == l3):
        return 6.0
    elif (l1 == l2) or (l2 == l3) or (l3 == l1):
        return 2.0
    else:
        return 1.0


def bispec_estimator(NSIDE_f_est, loop, limit):
    """
    :param NSIDE_f_est: Nside of map for estimation
    :param loop: Map maximum Temp in kelvin
    :param limit: limit that is needed by masking routine
    :return: This is main routine which work in parallel as
    different process and compute various Bispectrum.
    """
    # Allocate memory in wigfix to compute wigner 3j symbol

    wig.wig_table_init(1000)
    wig.wig_temp_init(1000)

    npix = hp.pixelfunc.nside2npix(NSIDE_f_est)

    # Degrading Map of Nside 512 to Nside 128 since resolution of haslam is 56 arcmin

    Haslam_nside = hp.pixelfunc.ud_grade(Haslam_512, nside_out=
                    NSIDE_f_est, order_in='RING', order_out='RING')

    # Masking map and obtaining Binary mask
    Haslam, Binary_Mask = masking_map(Haslam_nside, NSIDE_f_est, npix,
                                      limit)
    # Max value of l that we are looking at and total l bin
    LMAX = 250
    Nbin = 40 # I am using equal width bin one can use varying width bins also

    print LMAX, Nbin
    # APS(angular power spectrum)
    # APS of given masked map using anfast

    Cl = hp.sphtfunc.anafast(Haslam, map2=None, nspec=None, lmax=None,
         mmax =None, iter=3, alm=False, pol=False, use_weights=False,
         datapath = None)

    lbin_max = LMAX
    lbin_min = 50
    delta_i = (lbin_max-lbin_min)/Nbin
    print delta_i

    # Here we are computing Maps created using synfast at different l of Nside for estimation

    Esti_Map = np.zeros((Nbin, npix), dtype=np.double)

    for i in xrange(0, Nbin):

        # Computing alm for max l iteratively
        alm = hp.sphtfunc.map2alm(Haslam, lmax=LMAX, mmax=None, iter=3, pol=False, use_weights=False, datapath=None)

        # Creating mask in l so that it will pick al corresponding to that l

        window_func = np.zeros(LMAX, dtype=np.double)
        window_func[lbin_min] = 1.0
        hp.sphtfunc.almxfl(alm, window_func, mmax=None, inplace=True)

        # obtaining APS cl using alm = alm*window_func

        cls = hp.sphtfunc.alm2cl(alm, alms2=None, lmax=None, mmax=None,
              lmax_out=None, nspec=None)

        # Given cl one can compute map this gives maximally filtered map

        Map = hp.sphtfunc.synfast(cls, NSIDE_f_est, lmax=None, mmax=None, alm=False,
              pol=True, pixwin=False, fwhm=0.0, sigma=None, new=False,
              verbose=True)

        # for given bin Map of Nside of estimation

        Esti_Map[i, :] = Map
    
        lbin_min += delta_i

    temp = 50
    index = np.zeros(Nbin, int)
    for i in xrange(0, Nbin):
        index[i] = temp
        print temp
        temp += 5
    print index

    # index of these maps are now aquired

    name1 = '/home/sandeep/final_Bispectrum/%d/Bin_Bispectrum_%d_%d.txt' % (NSIDE_f_est, NSIDE_f_est, loop)
    with open(name1, 'w') as f:
        f.write("Bis\tangAvg_bis\tnorm_bis\tVarB\tCl1\tCl2\tCl3\ti1\ti2\ti3\tTripCount\n")

        for i in xrange(0, Nbin):
            for j in xrange(0, i+1):
                for k in xrange(0, j+1):

                    i3 = index[i]
                    i2 = index[j]
                    i1 = index[k]
                    Bis = 0.0

                    # we applied selection condition tirangle inequality and#parity condition
                    if abs(i2-i1) <= i3 <= i2+i1 and (i3+i2+i1)%2 == 0:

                        # Intiation of array for wigner 3j coefficient evaluation

                        b = [2*i1, 2*i2, 2*i3, 2*0, 2*0, 2*0]
                        wigner = wig.wig3jj(b)

                        alpha = np.sqrt((2*i1+1) * (2*i2+1) * (2*i3+1)) * (wigner/np.sqrt(np.pi*4.0))

                        # Bispectrum Estimation in next phase I'll parllel this loop
                        for ipix in xrange(0, npix):
                            Bis += Esti_Map[i, ipix]*Esti_Map[j, ipix]*Esti_Map[k, ipix]*Binary_Mask[ipix]

                        # Partial sky correction which also been taken care in previous step by multiplying Mask
                        Bis /= (4.0*np.pi*np.sum(Binary_Mask))
                        # Counting triplets in given bin
                        tripCount = count_triplet(i1, i3)
                        Bis /= (1.0*tripCount)

                        # Angle avg Bispectrum Assumes statistical isotropy in angle like in CMB

                        angAvg_bis = Bis/alpha
                        # Normalized bispectrum Given in Komatsu et. al.1998 COBE DMR
                        norm_bis = abs(angAvg_bis)/(Cl[i1]*Cl[i2]*Cl[i3])**0.5

                        # Variance in Bispectrum still not sure about this calcualtion
                        varb = g(i1, i2, i3) * (((2*i1+1) * (2*i2+1) * (2*i3+1)) / (4.*np.pi)) * wigner**2 * Cl[i1] * Cl[i2] * Cl[i3]
                        varb /= (1.0*tripCount**2.0)

                        f.write("%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e\t%0.6e\t%d\t%d\t%d\t%d\n" % (Bis, angAvg_bis,
                                norm_bis, varb, Cl[i1], Cl[i2], Cl[i3], i1, i2, i3, tripCount))

    wig.wig_temp_free()
    wig.wig_table_free()

# ****************************************************************************************

if __name__ == "__main__":
    print "Enter the Nside to which you want to upgrade or degrade the given map"
    NSIDE = int(raw_input(""))
    print "NSIDE = %d" % NSIDE

    Cell_Count1 = Process(target=bispec_estimator, args=(NSIDE, 18, 0.0012))
    Cell_Count1.start()

    Cell_Count2 = Process(target=bispec_estimator, args=(NSIDE, 50, 0.0032))
    Cell_Count2.start()

    Cell_Count3 = Process(target=bispec_estimator, args=(NSIDE, 200, 0.0128))
    Cell_Count3.start()

    Cell_Count1.join()
    Cell_Count2.join()
    Cell_Count3.join()




