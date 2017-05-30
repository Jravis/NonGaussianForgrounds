
import healpy as hp
import matplotlib.pyplot as plt
from astropy.io import fits

array = [35, 38, 40, 45, 50, 60, 70, 74, 80]
Max = [35, 30, 25, 15, 10, 8, 4, 4, 4]

count = 0
for fn in array:

    s1 = '/dataspace/sandeep/'
    s2 = 'LWA1_SkySurvey/healpix-all-sky-rav-wsclean-map-%d.fits' % fn
    filename = s1+s2
    tit = "%d" % fn
    map = hp.fitsfunc.read_map(filename)
    map_save = hp.mollview(map*1e-3, coord=['E', 'G'], xsize=2000, flip='astro', title=tit, cmap='hot',max=Max[count])  # gives me the attached figure.
    count += 1
plt.show()