"""
This code is for to compute Count
Probablity distribution (Histogram of flux in different cell size)
"""

import numpy as np
import healpy as hp
from multiprocessing import Process


def hist(data, mbin):
    """
    :param data: Data to calculate histogram
    :param mbin: bin
    :return: Return  frequency normalized
    """
    cfrq = []
    frq = []
    Count = len(data)
    for j in xrange(len(mbin)):
        count_1 = 0
        count_2 = 0
        for ii in xrange(len(data)):
            if mbin[j] < data[ii]:
                count_1 += 1
        for ii in xrange(len(data)):
            if j != (len(mbin)-1):
                if mbin[j] <= data[ii] < mbin[j+1]:
                    count_2 += 1
        cfrq.append(count_1)
        frq.append(count_2)

    frq = np.asarray(frq, dtype=np.double)
    return frq, Count


def masking_map(map1, nside, npixs, limit):
    """
    :param map1: Input Map for masking
    :param nside: nside of Input map
    :param npixs: number of pixel in map
    :param limit: limit on temperature for masking
    :return: This routine to apply mask that we decided using count in cell
    scheme. Masking unwanted pixel either by angular cut or temperature cut in
    Haslam
    """
    count3 = 0
    if limit == -1.0:  # -1.0 for All Sky without masking
        return map1, npixs
    else:
        area = hp.pixelfunc.nside2pixarea(nside, degrees=False)
        """
    theta = 0.0
    phi = 0.0
    This for galactic cut
    for ipix in xrange(0, np):
        theta , phi = hp.pixelfunc.pix2ang(nside,ipix)
        if 60 < m.degrees(theta) < 120.0:
            map1[ipix]= hp.UNSEEN
    """
        for ipix in xrange(0, npixs):
            temp = map1[ipix] * area
            if temp > limit:
                map1[ipix] = hp.UNSEEN
                count3 += 1
        return map1, npixs-count3


# **********************
# Main part of code    *
# **********************


name = "/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits"  # Reading Haslam map temprature(K)

Haslam_512 = hp.fitsfunc.read_map(name)
nside = 128
npix = hp.pixelfunc.nside2npix(nside)

# Degrading our map 512 to 128
Haslam_nside = hp.pixelfunc.ud_grade(Haslam_512, nside_out=nside, order_in='RING', order_out='RING')


def pn_estim(nmin, nmax, loop, count, bmask):

    """
    :param nmin: nmin is minimum of pixel index
    :param nmax: nmax is maximum of pixel index
    :param loop: Earlier used as angular scale of cell for estimation
    Now just temperature scale
    :param count: count total number of pixel
    :return:Function That will run parllel as seprate process
    Earlier statement act as gloabl variable
    """

    # You can mask the pixel using masking_map routine
    Haslam, number = masking_map(Haslam_nside, nside, npix, bmask)
    # making map dimension less dividing T/T_mean in pixel
    index = (Haslam != hp.UNSEEN)
    avg = np.mean(Haslam[index])
    pn = []
    print avg

# radius = m.radians(loop * 1.0) This when you want to do different cell size
    # area of single pixel of given Nside
    area = hp.pixelfunc.nside2pixarea(nside, degrees=False)

    for ind in xrange(nmin, nmax):
# theta_cen, phi_cen = hp.pixelfunc.pix2ang(nside, ind)
        """
        Query disc routine in Healpix and healpy to give pixel within angular distance from pixel center
            if 50 > m.degrees(theta_cen) or  m.degrees(theta_cen) > 130.0:
                hpxidx = hp.query_disc(256, hp.ang2vec(theta_cen, phi_cen),
                     radius, nest=False, inclusive=False)

                Pn[ind] = np.sum(Heslam[hpxidx])*area #*np.pi*radius**2.0
        """

        if Haslam[ind] != hp.UNSEEN:
#            Haslam[ind] = (Haslam[ind]-avg)/avg
            pn.append(Haslam[ind] * area)  # *np.pi*radius**2.0

    if loop == 128:
        name1 = '/home/sandeep/Haslam_128_AllSky.txt'
        name2 = '/home/sandeep/Count_128_AllSky.txt'
        name3 = '/home/sandeep/Frq_Bin_128_AllSky.txt'
    else:
        name1 = '/home/sandeep/Parllel_Heslam/Model_fitting_data/Haslam_128_%dK.txt' % loop
        name2 = '/home/sandeep/Parllel_Heslam/Model_fitting_data/Count_128_%dK.txt' % loop
        name3 = '/home/sandeep/Parllel_Heslam/Model_fitting_data/Frq_Bin_128_%d.txt' % loop

    with open(name1, 'w') as f:
        for i in xrange(len(pn)):
            f.write("%f\n" % pn[i])

    with open(name2, 'w') as f:
        f.write("%d\n" % count)

    pn = np.asarray(pn)
    bin_width1 = (max(pn) - min(pn))/1000

    bins1 = np.arange(np.amin(pn),  np.amax(pn)+bin_width1, bin_width1)
    frq1, length = hist(pn, bins1)
    print'yeh hai number'
    print bin_width1

    with open(name3, 'w') as f:
        for i in xrange(len(frq1)):
            f.write("%0.6e\t%0.6e\t%0.6e\n" % (frq1[i]/np.sum(frq1), bins1[i], frq1[i]))

if __name__ == "__main__":

    npix = hp.pixelfunc.nside2npix(nside)
    print npix
    count1 = 0
    count2 = 0
    Cell_Count1 = Process(target=pn_estim, args=(0, npix, 128, count1, -1.0))
    #Cell_Count2 = Process(target=pn_estim, args=(0, npix, 18, count1, 0.0012))
    #Cell_Count3 = Process(target=pn_estim, args=(0, npix, 50, count2, 0.0032))
    #Cell_Count4 = Process(target=pn_estim, args=(0, npix, 200, count2, 0.0128))

    Cell_Count1.start()
    #Cell_Count2.start()
    #Cell_Count3.start()
    #Cell_Count4.start()
    Cell_Count1.join()
    #Cell_Count2.join()
    #Cell_Count3.join()
    #Cell_Count4.join()

