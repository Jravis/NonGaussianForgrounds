"""
This code for Apodizing the mask to account for
Masking effect in power spectrum estimation
"""
import os
import numpy as np
import healpy as hp
import sys
import ephem
import matplotlib.pyplot as plt
import getopt


def loadMap(filename):
    """
    Given a filename, load and return the HEALPix map.
    """

    if os.path.splitext(filename)[1] == '.healnpy':
        map = np.load(filename)
    elif os.path.splitext(filename)[1] == '.fits':
        map = hp.fitsfunc.read_map(filename)

    elif os.path.splitext(filename)[1] == '.txt':
        map = np.genfromtxt(filename, dtype=np.float64)

    return map

def changeSides(map, nSides):
    """
    Take a HEALPix map upsample or downsample the map to make it have
    'nSides' sides and conserve flux.
    """

    nSidesOld = hp.pixelfunc.npix2nside(map.size)

    if getattr(map, 'mask', None) is not None:
        map2 = hp.pixelfunc.ud_grade(map.data, nSides)

        mask2 = hp.pixelfunc.ud_grade(map.mask.astype(np.float64), nSides)

        map2 = hp.ma(map2)
        map2.mask = np.where(mask2 >= 0.005, 1, 0).astype(np.bool)
    else:
        map2 = hp.pixelfunc.ud_grade(map, nSides)

    return map2


def changeResolution(map, fwhmCurrent, fwhmNew):
    """
    Given a HEALPix map with full width at half max resolution 'fwhmCurrent'
    degrees, smooth it to a new FWHM of 'fwhmNew' degrees.
    """

    if fwhmNew <= fwhmCurrent:
        return map
    else:
        smth = np.sqrt(fwhmNew ** 2 - fwhmCurrent ** 2)
        # smth /= 2*np.sqrt(2*np.log(2))

        if getattr(map, 'mask', None) is not None:
            data = map.data * 1.0
            data[np.where(map.mask == 1)] = 0
            map2 = hp.smoothing(data, fwhm=smth, degree=True)
            mask2 = hp.smoothing(map.mask.astype(np.float64), fwhm=smth, degree=True)

            map2 = hp.ma(map2)
            map2.mask = np.where(mask2 >= 0.005, 1, 0).astype(np.bool)
        else:
            map2 = hp.smoothing(map, fwhm=smth, degree=True)

        return map2


def convertMapCorrd(map, OldCord, NewCorrd):
    """
    Given a map in equatorial coordinates, convert it to Galactic coordinates.
    """

    rot = hp.rotator.Rotator(coord=(OldCord, NewCorrd))
    hasMask = False
    if getattr(map, 'mask', None) is not None:
        hasMask = True

    map2 = map * 0.0
    nSides = hp.pixelfunc.npix2nside(map.size)

    for i in xrange(map.size):
        theta, phi = hp.pixelfunc.pix2ang(nSides, i)
        theta, phi = rot(theta, phi)
        j = hp.pixelfunc.ang2pix(nSides, theta, phi)
        map2[j] += map[i]

        if hasMask:
            map2.mask[j] = map.mask[i]

    return map2


def convertMapToJ2000(map):
    """
    Given a map in Equatorial coodinates in the B1950 epoch, convert it to
    the J2000 epoch.
    """

    hasMask = False
    if getattr(map, 'mask', None) is not None:
        hasMask = True

    map2 = map * 0.0
    nSides = hp.pixelfunc.npix2nside(map.size)
    for i in xrange(map.size):
        theta, phi = hp.pixelfunc.pix2ang(nSides, i)
        eq = ephem.Equatorial(phi, np.pi / 2 - theta, epoch=ephem.B1950)
        eq = ephem.Equatorial(eq, epoch=ephem.J2000)
        j = hp.pixelfunc.ang2pix(nSides, np.pi / 2 - eq.dec, eq.ra)
        map2[j] += map[i]

        if hasMask:
            map2.mask[j] = map.mask[i]

    return map2


def getMapValue(map, ra, dec):
    """
    Given a HEALPix map and a right ascension/declianation pair, return
    the map value at that point.  Use the 'fhwm' keyword to provide the
    beam full width at have max in degrees
    """

    nSide = hp.pixelfunc.npix2nside(map.size)
    # Extract the region around the source
    vec = hp.pixelfunc.ang2vec(np.pi / 2 - np.deg2rad(dec) , np.deg2rad(ra))
    vec = np.array(vec)
    innerPixels = hp.query_disc(nSide, vec, radius=np.radians(1.5*56./60.))
    return innerPixels


def masking_map(map1, nside, npix, limit, Galcut):
    """
    This routine to apply mask that we decided using count in cell
    scheme.
    """

    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    #mask = np.zeros(hp.nside2npix(nside), dtype=np.double)
    area = hp.pixelfunc.nside2pixarea(nside, degrees=False)

    for ipix in xrange(0, npix):
        temp = map1[ipix]*area
        if temp > limit:
            mask[ipix] = 0.0

    if Galcut == 'Y':
        for ipix in xrange(0, npix):
            theta1, phi = hp.pixelfunc.pix2ang(nside, ipix)
            if 70. <= np.degrees(theta1) <= 110.:
                mask[ipix] = 0.0

    print 'Done'

    Cyg_A = getMapValue(map1, 23.39055556, 58.80000000)
    mask[Cyg_A] = 0.0
    Cas_A = getMapValue(map1, 19.99122222, 40.73388889)
    mask[Cas_A] = 0.0
    Tau_A = getMapValue(map1, 5.57555556, 22.01444444)
    mask[Tau_A] = 0.0
    Vir_A = getMapValue(map1, 17.76111111, -29.0078055)
    mask[Vir_A] = 0.0
    Her_A = getMapValue(map1, 12.51361111, 12.39111111)
    mask[Her_A] = 0.0
    Hyd_A = getMapValue(map1, 16.85222222, 4.99250000)
    mask[Hyd_A] = 0.0
    Sgr_A = getMapValue(map1, 9.30155556, -12.09555556)
    mask[Sgr_A] = 0.0
    Cen_A = getMapValue(map1, 13.42433333, -43.0191111)
    mask[Cen_A] = 0.0
    Vela_3 = getMapValue(map1, 8.56666667, -45.83333333)
    mask[Vela_3] = 0.0
    sky_3C48 = getMapValue(map1, 1.62808333, 33.15888889)
    mask[sky_3C48] = 0.0
    sky_3C123 = getMapValue(map1, 4.61787972, 29.67046380)
    mask[sky_3C123] = 0.0
    sky_3C147 = getMapValue(map1, 5.71002778, 49.85194444)
    mask[sky_3C147] = 0.0
    sky_3C196 = getMapValue(map1, 8.22666667, 48.21750000)
    mask[sky_3C196] = 0.0
    sky_3C270 = getMapValue(map1, 12.32311678, 5.82521530)
    mask[sky_3C270] = 0.0
    sky_3C295 = getMapValue(map1, 14.18902778, 52.20277778)
    mask[sky_3C295] = 0.0
    sky_3C353 = getMapValue(map1, 17.34113889, -0.97972222)
    mask[sky_3C353] = 0.0
    sky_3C380 = getMapValue(map1, 18.49216667, 48.74611111)
    mask[sky_3C380] = 0.0


# Fermi Bubble
    dataN = np.array([[345.1, 17.4],
                         [342.0, 25.5],
                         [339.1, 35.3],
                         [342.5, 44.8],
                         [3.1, 47.7],
                         [14.9, 37.5],
                         [18.3, 30.0],
                         [16.8, 16.8]])

    dataS = np.array([[11.7, -17.1],
                         [13.4, -25.0],
                         [15.1, -35.0],
                         [5.6, -51.1],
                         [347.8, -50.3],
                         [337.1, -39.5],
                         [337.1, -30.9],
                         [340.3, -23.3]])

    for i in xrange(0, 8):
        #thta_N = np.pi / 2 - np.radians(dataN[i, 1])
        #ph_N = np.radians(dataN[i, 0])

        #thta_S = np.pi / 2 - np.radians(dataS[i, 1])
        #ph_S = np.radians(dataS[i, 0])
        indx_N = getMapValue(map1, dataN[i, 0], dataS[i, 1])
        mask[indx_N] = 0.0
        indx_S = getMapValue(map1, dataS[i, 0], dataS[i, 1])
        mask[indx_S] = 0.0

    return mask


def apodiz(mask, theta):
    apodiz_mask = hp.sphtfunc.smoothing(mask, fwhm=np.radians(theta))
    index = (apodiz_mask < 0)
    apodiz_mask[index] = 0.000
    return apodiz_mask


def main(fname, NSIDE):

    input_map = loadMap(fname)
    NPIX = hp.pixelfunc.nside2npix(NSIDE)

    key = ['200K', '100K', '50K', '30K', '25K']
    arr = [0.0002553, 0.00035, 0.000162, 0.000122, 0.000101]
    clr = ['g', 'orange', 'crimson', 'b', 'k']

    count = 0

    for LIMIT in arr:
        if key[count] == '200K':
            galCut ='N'
        else:
            galCut = 'Y'

        Binary_mask = masking_map(input_map, NSIDE, NPIX, LIMIT, galCut)
        theta_ap = 2.0

        imp_map = apodiz(Binary_mask, theta_ap)
        masked_map = input_map*imp_map

        print max(masked_map)
        print masked_map

        f_name = "/dataspace/sandeep/Bispectrum_data/Input_Maps/ApodizeBinaryMask_%s_%0.1fdeg_apodi.fits" % (key[count],
                                                                                                             theta_ap)
        hp.fitsfunc.write_map(f_name, imp_map)

        f_name = "/dataspace/sandeep/Bispectrum_data/Input_Maps/MaskedMap_%s_%0.1fdeg_apodi.fits" % (key[count],
                                                                                                     theta_ap)
        hp.fitsfunc.write_map(f_name, masked_map)

#        print 'Enter the Lmax value you want for cl(APS) computation'
#        LMAX = int(raw_input(''))

        LMAX = 250

        l = np.arange(0, LMAX+1)

        cl = hp.sphtfunc.anafast(masked_map, lmax=LMAX)
        print count
        hp.mollview(imp_map, xsize=2000, coord=['G'], unit=r'$T_{B}(K)$', nest=False, title='%s' % key[count])

        name = "/dataspace/sandeep/Bispectrum_data/Input_Maps/ApodizeBinaryMask_%s_%0.1fdeg_apodi.pdf" % (key[count], theta_ap)
        plt.savefig(name, dpi=1200)

        hp.mollview(masked_map, xsize=2000, coord=['G'], unit=r'$T_{B}(K)$', nest=False, title='408 MHz,%s' % key[count])
        name = "/dataspace/sandeep/Bispectrum_data/Input_Maps/MaskedMap_%s_%0.1fdeg_apodi.pdf" % (key[count], theta_ap)
        plt.savefig(name, dpi=1200)

        fig = plt.figure(8, figsize=(7, 7))
        plt.plot(l, l * (l + 1) * cl, '-', color=clr[count], linewidth=2, label='%s' % key[count])
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(which='both')
        plt.legend()
        plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
        plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
        plt.minorticks_on()
        plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
        plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)
        count += 1

    fig.savefig("/dataspace/sandeep/Bispectrum_data/Input_Maps/AllCl.pdf", dpi=1200)

if __name__ == "__main__":

    filename = '/dataspace/sandeep/Bispectrum_data/haslam408_dsds_Remazeilles2014.fits'
    main(filename, 512)


plt.show()


