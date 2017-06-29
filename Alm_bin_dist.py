import numpy as np
import healpy as hp
import math as m
from astroML.plotting import hist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

#name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'

s1 = '/dataspace/sandeep/Bispectrum_data'
s2 = '/Gaussian_50K_test/Gaussian_50K_Maps/haslam_50KgaussMap_10.fits'

name = s1+s2
Haslam_512 = hp.fitsfunc.read_map(name)


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

    for ipix in xrange(0, npix):
        theta1, phi = hp.pixelfunc.pix2ang(nside, ipix)
        if 70. <= np.degrees(theta1) <= 110:
            mask[ipix] = 0.0

    return mask


def apodiz(mask):
    width = m.radians(2.0)
    apodiz_mask = hp.sphtfunc.smoothing(mask, fwhm=width)
    index = (apodiz_mask < 0)
    apodiz_mask[index] = 0.000
    return apodiz_mask

nside_f_est = 512
limit = 0.000162  # 50 K
#limit = 0.0002553  # 200K

npix = hp.nside2npix(nside_f_est)
binary_mask = masking_map(Haslam_512, nside_f_est, npix, limit)
ap_map = apodiz(binary_mask)
haslam = Haslam_512 * ap_map

lmax = 251
nbin = 12

index = 10 ** np.linspace(np.log10(2), np.log10(251), nbin)  # logrithmic bins

for i in xrange(len(index)):
    index[i] = int(index[i])

bin_arr = [[] for i in range(12)]
bin_arr1 = np.zeros((nbin-1, 2), dtype=int)
for i in xrange(0, nbin):
    ini = int(index[i])
    if i + 1 < nbin:
        final = int(index[i + 1])
        bin_arr1[i, 0] = ini
        bin_arr1[i, 1] = final
        bin_arr[i].append(range(ini, final))
fwhm = 56./60.  # For Haslam FWHM is 56 arc min
beam_l = hp.sphtfunc.gauss_beam(m.radians(fwhm), lmax=lmax, pol=False)
alm_obs = hp.sphtfunc.map2alm(haslam, lmax=lmax, iter=3)
alm_obs = hp.sphtfunc.almxfl(alm_obs, 1. / beam_l, mmax=None, inplace=True)


indX = 0
indY = 0
Count = 0


plt.figure(1, figsize=(17, 15))
gs = gridspec.GridSpec(4, 3)
for indX in xrange(0, 4):
    for indY in xrange(0, 3):
        if Count < 11:
            Count += 1
        else:
            break
        #print Count
        for i in xrange(Count-1, Count):
            alm = []
            temp = bin_arr[i][0]
                #print temp
            for l in temp:
                for m in xrange(-l, l+1):
                    if m < 0.0:
                        indx = hp.sphtfunc.Alm.getidx(lmax, l, -m)
                        alm.append(np.conjugate(alm_obs[indx]))
                    else:
                        indx = hp.sphtfunc.Alm.getidx(lmax, l, m)
                        alm.append(alm_obs[indx])

        print ""
        alm = np.asarray(alm)

        index = (alm.real != 0.0)
        index1 = (alm.imag != 0.0)

        almr = alm.real[index]
        almi = alm.imag[index1]

        """
        print "l Bin %d" % (i+1)
        print bin_arr1[i, :]

        #print "real part of alm in l bin"
        #print "imaginary part of alm in l bin"
        #print almr
        print ""

        ar, critr, signir = stats.anderson(almr, dist='norm')
        ai, criti, signii = stats.anderson(almi, dist='norm')

        print "Anderson-Darling test of normality"
        print "The test is a one-sided test and the hypothesis that the distribution is of a specific form " \
              "is rejected if the test statistic, A, is greater than the critical value. "

        print "The Anderson-Darling test statistic for real part of alm A = %f" % ar
        print "The critical values for this distribution for real parts of alm"
        print critr
        print "Significance level in % correponding to critical values for real parts of alm"
        print signir
        print ""

        print "The Anderson-Darling test statistic for real part of alm A = %f" % ai
        print "The critical values for this distribution for real parts of alm"
        print criti
        print "Significance level in % correponding to critical values for real parts of alm"
        print signii
        print ""

        skewr = stats.skew(almr, axis=0, bias=True)
        skewi= stats.skew(almi, axis=0, bias=True)
        print "Skewness for real part of alm in l bin data %f" % skewr
        print ""
        print "Skewness for imag part of alm in l bin data %f" % skewi

        print ""
        kurtr = stats.kurtosis(almr, axis=0, fisher=True, bias=True)
        kurti = stats.kurtosis(almi, axis=0, fisher=True, bias=True)
        print "Kurtosis for real part of alm in l bin data %f" % kurtr
        print ""
        print "Kurtosis for imag part of alm in l bin data %f" % kurti

        """

        real_alm, real_bin_edges = np.histogram(almr, bins='auto')
        imag_alm, imag_bin_edges = np.histogram(almi, bins='auto')

        ax1 = plt.subplot(gs[indX, indY])
        h1 = hist(almr, bins='knuth', histtype='stepfilled', ec='k', fc='#AAAAAA', label='Knuth')
        ax1.plot(real_bin_edges[:-1], real_alm, "bo", ms=5, label='auto')

        #h1 = hist(almi, bins='knuth', histtype='stepfilled', ec='k', fc='skyblue', label='Knuth')
        #ax1.plot(imag_bin_edges[:-1], imag_alm, "ro", ms=5, label='auto')

        ax1.set_ylabel(r'$N$', fontsize='large', fontstyle='italic', weight='extra bold')
        ax1.set_title(r'50K Haslam ($I_{%d}$)' % (i+1))
        #ax1.set_xlabel(r'$Imag(a_{lm})$', fontsize='large', fontstyle='italic', weight='extra bold')
        ax1.set_xlabel(r'$Real(a_{lm})$', fontsize='large', fontstyle='italic', weight='extra bold')
        ax1.set_yscale("log")
        plt.minorticks_on()
        plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
        plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)
        plt.tight_layout()
        plt.legend()

#plt.savefig("/dataspace/sandeep/alm_50K_stat/ActualMap_Bin_imag_alm_log.eps", dpi=100)
plt.savefig("/dataspace/sandeep/alm_50K_stat/GaussianMap_Bin_Real_alm_log.eps", dpi=100)

hp.mollview(Haslam_512)
plt.show()



#k2, pvalue = stats.mstats.normaltest(alm.real, axis=0)
#print ""
#print k2
#print pvalue

#W, p_value= stats.shapiro(alm.real)
#print ""
#print W
#print p_value
