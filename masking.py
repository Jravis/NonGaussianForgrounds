"""
This is code is for creating different maps
with different mask on.
"""

import matplotlib.pyplot as plt
import healpy as hp


def masking_Map(map1, nside, npixs, limit):
    area = hp.pixelfunc.nside2pixarea(nside, degrees=False)
    print area
    for ipix in xrange(0, npixs):
        temp = map1[ipix]*area
        if temp > limit:
            map1[ipix] = hp.UNSEEN
    return map1

name = "/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits"  # Reading Haslam map temprature(K)
print name
Haslam_512 = hp.fitsfunc.read_map(name)

NSIDE = 512
npix = hp.pixelfunc.nside2npix(NSIDE)
limit_512 = 0.0008
Haslam_512 = hp.sphtfunc.smoothing(Haslam_512, fwhm=0.016307192977800353)
Haslam = masking_Map(Haslam_512, NSIDE, npix, limit_512)

NSIDE = 128
npix = hp.pixelfunc.nside2npix(NSIDE)
Haslam_128 = hp.pixelfunc.ud_grade(Haslam_512, nside_out=128, order_in= 'RING', order_out='RING')
Haslam_128 = hp.sphtfunc.smoothing(Haslam_128, fwhm=0.016307192977800353)
limit_128 = 0.00194   #0.0032 #0.0012 #0.0128
Haslam1 = masking_Map(Haslam_128, NSIDE, npix, limit_128)

NSIDE = 256
npix = hp.pixelfunc.nside2npix(NSIDE)
Haslam_256 = hp.pixelfunc.ud_grade(Haslam_512, nside_out= 256, order_in= 'RING', order_out='RING')
limit_256 = 0.0032
Haslam_256 = hp.sphtfunc.smoothing(Haslam_256, fwhm=0.016307192977800353)
Haslam2 = masking_Map(Haslam_256, NSIDE, npix, limit_256)


hp.mollview(Haslam, coord=['G'], xsize=2000, unit=r'$T_{B}(K)$', nest=False)
plt.savefig("/home/sandeep/Parllel_Heslam/masking_plots/512_200k.eps",
           dpi=100)

hp.mollview(Haslam1, coord=['G'], xsize=2000, unit=r'$T_{B}(K)$', nest=False)
plt.savefig("/home/sandeep/Parllel_Heslam/masking_plots/128_200k.eps",
           dpi=100)

hp.mollview(Haslam2, coord=['G'], xsize=2000, unit=r'$T_{B}(K)$', nest=False)
plt.savefig("/home/sandeep/Parllel_Heslam/masking_plots/256_200k.eps",
           dpi=100)

plt.show()


