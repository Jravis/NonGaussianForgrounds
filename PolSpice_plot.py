import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

theta_ap =2.0
count=0
clr = ['g', 'orange', 'crimson', 'b', 'k']
key = ['200K', '90K', '50K', '30K', '25K']
 
for i in xrange(len(clr)):

    f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/MaskedMap_%s_%0.1fdeg_apodi.fits" % (key[count],theta_ap)

    f_name2 ="/dataspace/sandeep/Bispectrum_data/Input_Maps/PolSpice_data/cl_%s_%0.1fdeg_apodi.fits" % (key[count],theta_ap)

    NSIDE=512

    masked_map = hp.fitsfunc.read_map(f_name1)
    spice_cl=  hp.fitsfunc.read_cl(f_name2)
    LMAX = 3*NSIDE-1

    l = np.arange(0, LMAX+1)
    cl = hp.sphtfunc.anafast(masked_map, lmax=LMAX)

    fig = plt.figure(count+1, figsize=(10, 6))
    plt.plot(l, l * (l + 1) * cl, '-', color='r', linewidth=2, label='%s' % key[count])
    plt.plot(l, l * (l + 1) * spice_cl, '-', color='b',
             linewidth=2, label='PolSpice %s' % key[count])
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(1e-3,1e5)
    #plt.xlim(1,256)
    plt.grid(which='both')
    plt.legend()
    plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
    plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
    plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)
    nn="/dataspace/sandeep/Bispectrum_data/Input_Maps/Pol_Cl_2deg_%s.pdf"%key[count]
    fig.savefig(nn, dpi=300)
    count += 1
plt.show()
