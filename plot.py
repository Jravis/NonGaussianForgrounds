import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import healpy as hp
import matplotlib

name = '/home/sandeep/Parllel_Heslam/haslam408_dsds_Remazeilles2014.fits'
print name
Haslam_512 = hp.fitsfunc.read_map(name)
Haslam_128 = hp.pixelfunc.ud_grade(Haslam_512, nside_out=128)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_80K_apod_300arcm_ns_128.fits'
mask_80K = hp.fitsfunc.read_map(name, verbose=False)
haslam = Haslam_128 * mask_80K

hp.mollview(haslam, xsize=2000, unit=r'$T_{B}(K)$', nest=False, title='%s' % '80K')
name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/maps_128/Map_80K_apod_300arcm_ns_128.png'
plt.savefig(name, dpi=600)
"""
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_60K_apod_300arcm_ns_128.fits'
mask_60K = hp.fitsfunc.read_map(name, verbose=False)
haslam_60K = Haslam_128 * mask_60K

name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_50K_apod_300arcm_ns_128.fits'
mask_50K = hp.fitsfunc.read_map(name, verbose=False)
haslam_50K = Haslam_128 * mask_50K

name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_40K_apod_300arcm_ns_128.fits'
mask_40K = hp.fitsfunc.read_map(name, verbose=False)
haslam_40K = Haslam_128 * mask_40K

name = '/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_30K_apod_300arcm_ns_128.fits'
mask_30K = hp.fitsfunc.read_map(name, verbose=False)
haslam_30K = Haslam_128 * mask_30K


dpi = 600
fig_size_inch = 12, 12
fig = plt.figure(1, figsize=(12, 12))


hp.mollview(haslam_60K,fig=fig.number, xsize=fig_size_inch[0]*dpi, unit=r'$T_{B}(K)$', nest=False, sub=(2, 2, 1), title='%s' % '60K')
hp.mollview(haslam_50K,fig=fig.number, xsize=fig_size_inch[0]*dpi, unit=r'$T_{B}(K)$', nest=False, sub=(2, 2, 2), title='%s' % '50K')
hp.mollview(haslam_40K,fig=fig.number, xsize=fig_size_inch[0]*dpi, unit=r'$T_{B}(K)$', nest=False, sub=(2, 2, 3), title='%s' % '40K')
hp.mollview(haslam_30K,fig=fig.number, xsize=fig_size_inch[0]*dpi, unit=r'$T_{B}(K)$', nest=False, sub=(2, 2, 4), title='%s' % '30K')

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05)
name = '/dataspace/sandeep/Bispectrum_data/Map_all_apod_300arcm_ns_128.png'
plt.savefig(name, dpi=600, bbox_inches="tight")









# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

key = ['60K', '50K', '40K', '30K']
clr = ['b', 'r', 'k', 'y']
count = 0
sky_sum = []
for i in xrange(len(key)):
    f_name1 = "/dataspace/sandeep/Bispectrum_data/Input_Maps/mask_apod_128/Mask_%s_apod_300arcm_ns_128.fits" % key[i]
    ap_map_128 = hp.fitsfunc.read_map(f_name1)
    sky_sum.append(np.sum(ap_map_128))

nbin = 11
index = np.logspace(np.log10(10), np.log10(256), 11, endpoint=True, dtype=np.int32)

# *****************************************************************
# for 15 bin scheme

#index = [10, 19, 27, 39, 46, 55, 65, 77, 91, 109, 129, 153, 181, 215, 256]
#index = np.asarray(index, dtype=np.int32)
#nbin = len(index)

print nbin
bin_arr = np.zeros((nbin - 1, 2), dtype=np.int32)
npix = hp.nside2npix(128)

frac_sky = np.asarray(sky_sum, dtype=np.float64)/npix
print frac_sky

for i in xrange(0, nbin-1):
    ini = index[i]
    if i+1 < nbin:
        final = index[i+1]
        """
        if ini+5 > final:
            bin_arr[i, 0] = ini
            temp = abs(final-ini)
            bin_arr[i, 1] = final+temp
            index[i+1] = final+temp
        else:
        """
        bin_arr[i, 0] = ini
        bin_arr[i, 1] = final


Indx = 0
Indy = 0
Actual = np.zeros((5, nbin-1), dtype=np.float64)
Mean = np.zeros((5, nbin-1), dtype=np.float64)
Std_Dev = np.zeros((5, nbin-1), dtype=np.float64)

lmax = 3*128-1

l = np.arange(lmax+1)


for fn in key:

    s1 = "/dataspace/sandeep/Bispectrum_data"
    s2 = "/Gaussian_%s_test/haslam_%s_ns128_cl.txt" % (fn, fn)
    name = s1+s2
    cl = np.genfromtxt(name)
    for i in xrange(0, nbin-1):

        window_func = np.zeros(lmax+1, float)

        for j in xrange(bin_arr[i, 0], bin_arr[i, 1]):
            window_func[j] = 1.0
        length = np.sum(window_func)
        Sum = []
        for nn in xrange(0, 1000):
            s1 = "/dataspace/sandeep/Bispectrum_data"
            s2 = "/Gaussian_%s_test/Gaussian_%s_cl/haslam_%sgaussMap_cl_%d.txt" % (fn, fn, fn, nn)
            filename = s1+s2
            Map_cl = np.genfromtxt(filename)
            #Sum.append(np.sum(Map_cl*window_func*l*(l+1))/length)

            Sum.append(np.sum(np.multiply(Map_cl*l*(l+1), window_func))/length)

        Sum = np.asarray(Sum)
        #Actual[count, i] = np.sum(cl*window_func*l*(l+1))/length
        Actual[count, i] = np.sum(np.multiply(cl*l*(l+1), window_func))/length
        Mean[count, i] = np.mean(Sum)
        Std_Dev[count, i] = np.std(Sum)
    count += 1

#bl = hp.sphtfunc.gauss_beam(np.radians(33./60.), lmax=3*128-1)
#wl = hp.sphtfunc.pixwin(128, pol=False)


I = np.arange(nbin-1)
I = np.add(I, 1)
print ""
print I
fig = plt.figure(6, figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.02, wspace=0.3)

ax1 = plt.subplot(gs[0, 0])
#ax1.fill_between(l, l*(l+1)*(Mean[0,:]-Std_Dev[0, :]), l*(l+1)*(Mean[0, :]+Std_Dev[0, :]), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
#ax1.plot(l, l*(l+1)*Mean[0, :], '-', color='crimson', linewidth=2, label='mean Cl')

ax1.plot(I, Actual[0, :], '-', ms=5,  color='b', linewidth=2)
ax1.errorbar(I, Mean[0, :], yerr=Std_Dev[0, :], fmt='ko', ecolor='k', capthick=4)

#ax1.plot(l, 1.*bl**2*wl[:-129]**2, '-', color='g', linewidth=2,
#        label='beam_window')

ax1.set_yscale("log")
plt.ylabel(r'$C_{I}$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlim(0.0, 11)
ax1.text(4, 450., r'$\mathbf{T_{Cut}:}$ 60K, '
                 r'$\mathbf{f_{sky}:}$ '
                 ' %0.1f %%' % (frac_sky[0]*100),
                 bbox={'facecolor':'white', 'alpha': 0.5, 'pad': 4}, verticalalignment='bottom')


ax2 = plt.subplot(gs[1, 0], sharex=ax1)
ax2.plot(I, (Actual[0, :]-Mean[0, :])/Std_Dev[0, :], 'bo-', linewidth=2)
ax2.axhline(y=0.0, color='k', linewidth=2)
plt.xlabel(r'$I$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$\Delta$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlim(0.0, 11)
plt.ylim(-1.0, 1.0)


ax3 = plt.subplot(gs[0, 1])
ax3.plot(I, Actual[1, :], '-', ms=5,  color='b', linewidth=2)
ax3.errorbar(I, Mean[1, :], yerr=Std_Dev[1, :], fmt='ko', ecolor='k', capthick=4)
ax3.set_yscale("log")
plt.ylabel(r'$C_{I}$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlim(0.0, 11)
ax3.text(4.0, 450, r'$\mathbf{T_{Cut}:}$ 50K, '
                   r'$\mathbf{f_{sky}:}$ '
                   ' %0.1f %%' % (frac_sky[1]*100),
                   bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 4})  #, verticalalignment='bottom')

ax4 = plt.subplot(gs[1, 1], sharex=ax3)
ax4.plot(I, (Actual[1, :]-Mean[1, :])/Std_Dev[1, :], 'bo-', linewidth=2)
ax4.axhline(y=0.0, color='k', linewidth=2)
plt.xlabel(r'$I$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$\Delta$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlim(0.0, 11)
plt.ylim(-1.0, 1.0)

#fig.savefig("/dataspace/sandeep/Bispectrum_data/BinnedCl_1000sim_1_nbin-15.png", dpi=600)
fig.savefig("/dataspace/sandeep/Bispectrum_data/BinnedCl_1000sim_1.png", dpi=600)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

fig = plt.figure(7, figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.01, wspace=0.3)

ax5 = plt.subplot(gs[0, 0])
ax5.plot(I, Actual[2, :], '-', ms=5,  color='b', linewidth=2)
ax5.errorbar(I, Mean[2, :], yerr=Std_Dev[2, :], fmt='ko', ecolor='k', capthick=4)
ax5.set_yscale("log")
plt.ylabel(r'$C_{I}$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlim(0., 11)
ax5.text(4.0, 450, r'$\mathbf{T_{Cut}:}$ 40K, '
                 r'$\mathbf{f_{sky}:}$ '
                 ' %0.1f %%' % (frac_sky[2]*100),
                 bbox={'facecolor':'white', 'alpha': 0.5, 'pad': 5})#, verticalalignment='bottom')

ax6 = plt.subplot(gs[1, 0], sharex=ax5)
ax6.plot(I, (Actual[2, :]-Mean[2, :])/Std_Dev[2, :], 'bo-', linewidth=2)
ax6.axhline(y=0.0, color='k', linewidth=2)
plt.xlabel(r'$I$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$\Delta$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlim(0., 11)
plt.ylim(-1.0, 1.0)


ax7 = plt.subplot(gs[0, 1])
ax7.plot(I, Actual[3, :], '-', ms=5,  color='b', linewidth=2)
ax7.errorbar(I, Mean[3, :], yerr=Std_Dev[3, :], fmt='ko', ecolor='k', capthick=4)
ax7.set_yscale("log")
plt.ylabel(r'$C_{I}$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlim(0., 11)
ax7.text(4.0, 450.0, r'$\mathbf{T_{Cut}:}$ 30K, '
                 r'$\mathbf{f_{sky}:}$ '
                 ' %0.1f %%' % (frac_sky[3]*100),
                 bbox={'facecolor':'white', 'alpha': 0.5, 'pad': 5}, verticalalignment='bottom')


ax8 = plt.subplot(gs[1, 1], sharex=ax7)
ax8.plot(I, (Actual[3, :]-Mean[3, :])/Std_Dev[3, :], 'bo-', linewidth=2)

ax8.axhline(y=0.0, color='k', linewidth=2)
plt.xlabel(r'$I$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$\Delta$', fontsize='large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=10)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=10)
plt.xlim(0., 11)
plt.ylim(-1.0, 1.0)

#fig.savefig("/dataspace/sandeep/Bispectrum_data/BinnedCl_1000sim_2_nbin-15.png", dpi=600)
fig.savefig("/dataspace/sandeep/Bispectrum_data/BinnedCl_1000sim_2.png", dpi=600)

"""
ax1 = plt.subplot(gs[0, 1])
ax1.fill_between(l, l*(l+1)*(Mean[1, :]-Std_Dev[1, :]), l*(l+1)*(Mean[1, :]+Std_Dev[1, :]), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
ax1.plot(l, l*(l+1)*Mean[1, :], '-', color='crimson', linewidth=2, label='mean Cl')
ax1.plot(l, l*(l+1)*Actual[1, :], '-', color='orange', linewidth=2, label='original Cl')
ax1.plot(l, 1.*bl**2*wl[:-129]**2, '-', color='g', linewidth=2,
        label='beam_window')
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_title('50K')

plt.grid(which='both')
plt.legend(loc=3)
plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)

plt.tight_layout()
ax1 = plt.subplot(gs[1, 0])
ax1.fill_between(l, l*(l+1)*(Mean[2, :]-Std_Dev[2, :]), l*(l+1)*(Mean[2, :]+Std_Dev[2, :]), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
ax1.plot(l, l*(l+1)*Mean[2, :], '-', color='crimson', linewidth=2, label='mean Cl')
ax1.plot(l, l*(l+1)*Actual[2, :], '-', color='orange', linewidth=2, label='original Cl')
ax1.plot(l, 1.*bl**2*wl[:-129]**2, '-', color='g', linewidth=2,
        label='beam_window')
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_title('40K')


plt.grid(which='both')
plt.legend(loc=3)
plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)

plt.tight_layout()

ax1 = plt.subplot(gs[1, 1])
ax1.fill_between(l, l*(l+1)*(Mean[3,:]-Std_Dev[3, :]), l*(l+1)*(Mean[3, :]+Std_Dev[3, :]), alpha=0.5, edgecolor='c', facecolor='paleturquoise')
ax1.plot(l, l*(l+1)*Mean[3, :], '-', color='crimson', linewidth=2, label='mean Cl')
ax1.plot(l, l*(l+1)*Actual[3, :], '-', color='orange', linewidth=2, label='original Cl')
ax1.plot(l, 1.*bl**2*wl[:-129]**2, '-', color='g', linewidth=2,
        label='beam_window')
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_title('30K')
plt.grid(which='both')
plt.legend(loc=3)
plt.xlabel(r'$l$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.ylabel(r'$l(l+1)C_{l}$', fontsize='x-large', fontstyle='italic', weight='extra bold')
plt.minorticks_on()
plt.tick_params(axis='both', which='minor', length=5, width=2, labelsize=14)
plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=14)

plt.tight_layout()

plt.savefig("/dataspace/sandeep/Bispectrum_data/1000_Realization_sim.pdf", dpi=100)
"""

plt.show()
