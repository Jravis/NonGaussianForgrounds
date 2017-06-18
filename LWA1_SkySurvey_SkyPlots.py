
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits
import  numpy as np


lambda_G, beta_G = 180.01, 29.80 # Galactic pole in elliptical corrd
lambda_B, beta_B = 266.84, -5.54 # Galactic center in elliptical corrd


def ellip_gal(ell_lambda, ell_beta):

    sb = np.sin(ell_beta)
    cb = np.cos(ell_beta)
    sbeta_G = np.sin(beta_G)
    cbeta_G = np.cos(beta_G)

    sbeta_B = np.sin(beta_B)
    cbeta_B = np.cos(beta_B)

    bk = np.arccos(sbeta_B*cbeta_G - cbeta_B * cbeta_G * np.cos(lambda_G-lambda_B))
    print np.degrees(bk)
#    ell_beta = np.arcsin(sbeta_G*sb + cbeta_G * cb * np.cos(bk - l))
#    ell_lambda = np.arctan2((cb*np.sin(bk-l)), (cbeta_G*sb - sbeta_G*cb*np.cos(bk-l)))

    b = np.arcsin(sbeta_G*sb + cbeta_G*cb*np.cos(ell_lambda-lambda_B))
    l = bk - np.arctan2(cb * np.sin(ell_lambda-lambda_B), cbeta_G*sb - sbeta_G * cb * np.cos(ell_lambda-lambda_B))
    return b, l

def masking_map(map1, nside, npix):
    """
    This routine to apply mask that we decided using count in cell
    scheme.
    """
    mask = np.ones(hp.nside2npix(nside), dtype=np.double)
    for ipix in xrange(0, npix):
        theta1, phi = hp.pixelfunc.pix2ang(nside, ipix)
        theta, phi1 = ellip_gal(phi, theta1)
        print np.degrees(theta), theta
        ipix1 = hp.pixelfunc.ang2pix(nside, theta, phi1)
        mask[ipix1] = map1[ipix]

        #if 70. <= np.degrees(theta) <= 110:
        #    #print np.degrees(theta)
        #    mask[ipix] = 0.0

    return mask


array = [35]#, 38, 40, 45, 50, 60, 70, 74, 80]
Max = [35, 30, 25, 15, 10, 8, 4, 4, 4]
color = ['r', 'b', 'g']
count = 0
LMAX = 250
for fn in array:

    s1 = '/dataspace/sandeep/'
    s2 = 'LWA1_SkySurvey/healpix-all-sky-rav-wsclean-map-%d.fits' % fn
    filename = s1+s2
    tit = "%d" % fn
    map = hp.fitsfunc.read_map(filename)
    Nside = hp.pixelfunc.get_nside(map)
    Npix = hp.nside2npix(Nside)
    mask= masking_map(map,Nside, Npix)
    m = hp.ma(map)
    m.mask = np.isnan(m)
    #cl_200 = hp.sphtfunc.anafast(masked_map)
    print Nside, Npix
    map_save = hp.mollview(map, xsize=2000, flip='astro', title=tit, cmap='jet')#,max=Max[count])  # gives me the attached figure.
    map_save1 = hp.mollview(mask, xsize=2000, flip='astro', title=tit, cmap='jet')#,max=Max[count])  # gives me the attached figure.
    #map_save = hp.mollview(map*1e-3*masking_map(map,Nside, Npix), coord=['E', 'G'], xsize=2000, flip='astro', title=tit, cmap='hot')  # gives me the attached figure.

plt.show()
"""
l = np.arange(len(cl_200))

plt.figure(8, figsize=(7, 7))
plt.plot(l, l * (l + 1) * cl_200, '-', color=color[count], linewidth=2, label='%d' % fn)
    #plt.yscale("log")
plt.xscale("log")
plt.grid(which='both')
plt.legend()
plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)
count += 1

plt.show()

"""