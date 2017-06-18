"""
; NAME:
;     EULER_2000
; PURPOSE:
;     Transform between Galactic, celestial, and ecliptic coordinates.
; EXPLANATION:
;     Use the procedure ASTRO to use this routine interactively
;
; CALLING SEQUENCE:
;      EULER_2000, AI, BI, AO, BO, [ SELECT, /FK4 ]
;
; INPUTS:
;       AI - Input Longitude in DEGREES, scalar or vector.  If only two
;               parameters are supplied, then  AI and BI will be modified to
;               contain the output longitude and latitude.
;       BI - Input Latitude in DEGREES
;
; OPTIONAL INPUT:
;       SELECT - Integer (1-6) specifying type of coordinate transformation.
;
;      SELECT   From          To        |   SELECT      From            To
;       1     RA-Dec (2000)  Galactic   |     4       Ecliptic      RA-Dec
;       2     Galactic       RA-DEC     |     5       Ecliptic      Galactic
;       3     RA-Dec         Ecliptic   |     6       Galactic      Ecliptic
;
;      If omitted, program will prompt for the value of SELECT
;      Celestial coordinates (RA, Dec) should be given in equinox J2000
;      unless the /FK4 keyword is set.
; OUTPUTS:
;       AO - Output Longitude in DEGREES
;       BO - Output Latitude in DEGREES
;
; INPUT KEYWORD:
;       /FK4 - If this keyword is set and non-zero, then input and output
;             celestial and ecliptic coordinates should be given in equinox
;             B1950.
;
; NOTES:
;       EULER was changed in December 1998 to use J2000 coordinates as the
;       default, ** and may be incompatible with earlier versions***.
; REVISION HISTORY:
;       Written W. Landsman,  February 1987
;       Adapted from Fortran by Daryl Yentis NRL
;       Converted to IDL V5.0   W. Landsman   September 1997
;       Made J2000 the default, added /FK4 keyword  W. Landsman December 1998
;       Renamed to euler_2000.pro by D. Finkbeiner 15 Apr 1999
;       -------------------
;       Memory managment improved 16 April 1999 - D. Finkbeiner.
;        - now makes heavy use of "temporary" function to deallocate
;          arrays
;        - arguments of atan() are now floats, since atan is the
;          limiting factor on memory usage, and numerical precision is
;          not such a big deal for atan.
;        - These changes reduce memory usage by 58%, but cause
;          differences of up to .03 arcsec relative to the standard
;          version of euler.  If you are interested in higher
;          precision than this, DO NOT USE THIS ROUTINE!
;       -------------------
;-
 On_error,2
"""
import numpy as np
def corrd_trans(ai, bi, keyword_set):


#   J2000 coordinate conversions are based on the following constants
    eps = 23.4392911111         #   Obliquity of the ecliptic
    alphaG = 192.85948             #  Right Ascension of Galactic North Pole
    deltaG = 27.12825               # Declination of Galactic North Pole
    lomega = 32.93192                #Galactic longitude of celestial equator
    alphaE = 180.02322              #Ecliptic longitude of Galactic North Pole
    deltaE = 29.811438523            #Ecliptic latitude of Galactic North Pole
    Eomega  = 6.3839743             #Galactic longitude of ecliptic equator

    if keyword_set == 'FK4':

        equinox = '(B1950)'
        psi = [0.57595865315, 4.9261918136, 0.00000000000, 0.0000000000,
               0.11129056012, 4.7005372834]
        stheta = [0.88781538514, -0.88781538514, 0.39788119938, -0.39788119938,
                  0.86766174755, -0.86766174755]
        ctheta = [0.46019978478, 0.46019978478, 0.91743694670, 0.9174369467,
                  0.49715499774, 0.49715499774]
        phi = [4.9261918136,  0.57595865315, 0.0000000000, 0.00000000000,
	           4.7005372834, 0.11129056012]
    else:
        equinox = '(J2000)'
        psi = [0.57477043300, 4.9368292465, 0.00000000000, 0.0000000000,
               0.11142137093, 4.71279419371]
        stheta = [0.88998808748,-0.88998808748, 0.39777715593,-0.39777715593,
                  0.86766622025, -0.86766622025]
        ctheta = [0.45598377618, 0.45598377618, 0.91748206207, 0.91748206207,
                  0.49714719172, 0.49714719172]
        phi = [4.9368292465,  0.57477043300, 0.0000000000, 0.00000000000,
               4.71279419371, 0.11142137093]
    if npar < 5:
        print ' '
        print ' 1 RA-DEC ' + equinox + ' to Galactic'
        print ' 2 Galactic       to RA-DEC' + equinox
        print ' 3 RA-DEC ' + equinox + ' to Ecliptic'
        print ' 4 Ecliptic       to RA-DEC' + equinox
        print ' 5 Ecliptic       to Galactic'
        print ' 6 Galactic       to Ecliptic'

        select = 0
        print 'Enter selection: ',
        select = np.int(raw_input(""))

    i = select - 1

    a = np.radians(ai) - phi[i]
    b = np.radains(bi)
    sb = np.sin(b)
    cb = np.cos(b)
    cbsa = cb * np.sin(a)
    xx = cb * np.cos(a)

    b = -stheta[i] * cbsa + ctheta[i] * sb
    bo = np.degrees(np.asin(b < 1.0))

    yy = ctheta[i] * cbsa + stheta[i] * sb

    a = np.atan(yy, xx)
    ao = np.degrees(a + (psi[i] + 4.*np.pi)) % 2.*np.pi

    return ao, bo




