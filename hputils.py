#-*- coding: utf-8 -*-

"""
Collection of utilities for working with HEALPix maps.

$Rev: 600 $
$LastChangedBy: jdowell $
$LastChangedDate: 2017-03-21 13:28:44 -0600 (Tue, 21 Mar 2017) $
"""

import os
import sys
import numpy
import healpy
from scipy.optimize import leastsq
import pylab
from matplotlib.ticker import LinearLocator, FixedLocator, LogLocator
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.colorbar import make_axes, ColorbarBase

try:
	from matplotlib.colors import SymLogNorm
	from matplotlib.ticker import SymmetricalLogLocator
except ImportError:
	# Old matplotlib, using the function from the github repository
	
	class SymLogNorm(Normalize):
		"""
		The symmetrical logarithmic scale is logarithmic in both the
		positive and negative directions from the origin.
		Since the values close to zero tend toward infinity, there is a
		need to have a range around zero that is linear.  The parameter
		*linthresh* allows the user to specify the size of this range
		(-*linthresh*, *linthresh*).
		"""
		
		def __init__(self,  linthresh, linscale=1.0, vmin=None, vmax=None, clip=False):
			"""
			*linthresh*:
			The range within which the plot is linear (to
			avoid having the plot go to infinity around zero).
			*linscale*:
			This allows the linear range (-*linthresh* to *linthresh*)
			to be stretched relative to the logarithmic range.  Its
			value is the number of decades to use for each half of the
			linear range.  For example, when *linscale* == 1.0 (the
			default), the space used for the positive and negative
			halves of the linear range will be equal to one decade in
			the logarithmic range. Defaults to 1.
			"""
			
			Normalize.__init__(self, vmin, vmax, clip)
			self.linthresh = float(linthresh)
			self._linscale_adj = (linscale / (1.0 - 10.0**-1))
			
		def __call__(self, value, clip=None):
			if clip is None:
				clip = self.clip
				
			result, is_scalar = self.process_value(value)
			self.autoscale_None(result)
			vmin, vmax = self.vmin, self.vmax
			
			if vmin > vmax:
				raise ValueError("minvalue must be less than or equal to maxvalue")
			elif vmin == vmax:
				result.fill(0)
			else:
				if clip:
					mask = numpy.ma.getmask(result)
					result = numpy.ma.array(numpy.clip(result.filled(vmax), vmin, vmax),
									mask=mask)
				# in-place equivalent of above can be much faster
				resdat = self._transform(result.data)
				resdat -= self._lower
				resdat /= (self._upper - self._lower)
				
			if is_scalar:
				result = result[0]
			return result
			
		def _transform(self, a):
			"""
			Inplace transformation.
			"""
			masked = numpy.abs(a) > self.linthresh
			sign = numpy.sign(a[masked])
			log = (self._linscale_adj + numpy.log10(numpy.abs(a[masked]) / self.linthresh))
			log *= sign * self.linthresh
			a[masked] = log
			a[~masked] *= self._linscale_adj
			return a
			
		def _inv_transform(self, a):
			"""
			Inverse inplace Transformation.
			"""
			masked = numpy.abs(a) > (self.linthresh * self._linscale_adj)
			sign = numpy.sign(a[masked])
			exp = 10.0**(sign * a[masked] / self.linthresh - self._linscale_adj)
			exp *= sign * self.linthresh
			a[masked] = exp
			a[~masked] /= self._linscale_adj
			return a
			
		def _transform_vmin_vmax(self):
			"""
			Calculates vmin and vmax in the transformed system.
			"""
			vmin, vmax = self.vmin, self.vmax
			arr = numpy.array([vmax, vmin]).astype(float)
			self._upper, self._lower = self._transform(arr)
			
		def inverse(self, value):
			if not self.scaled():
				raise ValueError("Not invertible until scaled")
			val = numpy.ma.asarray(value)
			val = val * (self._upper - self._lower) + self._lower
			return self._inv_transform(val)
			
		def autoscale(self, A):
			"""
			Set *vmin*, *vmax* to min, max of *A*.
			"""
			self.vmin = numpy.ma.min(A)
			self.vmax = numpy.ma.max(A)
			self._transform_vmin_vmax()
			
		def autoscale_None(self, A):
			""" autoscale only None-valued vmin or vmax """
			if self.vmin is not None and self.vmax is not None:
				pass
			if self.vmin is None:
				self.vmin = numpy.ma.min(A)
			if self.vmax is None:
				self.vmax = numpy.ma.max(A)
			self._transform_vmin_vmax()


class LogNorm(Normalize):
	"""
	Normalize a given value to the 0-1 range on a log scale
	"""
	
	def __call__(self, value, clip=None):
		value = numpy.ma.asarray(value)
		mask = numpy.ma.getmaskarray(value)
		value = value.filled(self.vmax+1)
		if clip:
			numpy.clip(value, self.vmin, self.vmax)
			
		output = (value - self.vmin) / (self.vmax - self.vmin)
		output *= 9
		output += 1
		output = numpy.log10(output)
		
		output = numpy.ma.array(output, mask=mask)
		if output.shape == () and not mask:
			output = int(output)  # assume python scalar
		return output


class HistEqNorm(Normalize):
	"""
	Normalize a given value to the 0-1 range using histrogram equalization
	"""
	
	def __call__(self, value, clip=None):
		value = numpy.ma.asarray(value)
		mask = numpy.ma.getmaskarray(value)
		value = value.filled(self.vmax+1)
		if clip:
			numpy.clip(value, self.vmin, self.vmax)
			
		hist, bins = numpy.histogram(value, bins=256)
		hist = numpy.insert(hist, 0, 0)
		hist = hist.cumsum() / float(hist.sum())
		histeq = interp1d(bins, hist, bounds_error=False, fill_value=0.0)
		output = histeq(value)
		
		output = numpy.ma.array(output, mask=mask)
		if output.shape == () and not mask:
			output = int(output)  # assume python scalar
		return output


def loadMap(filename):
	"""
	Given a filename, load and return the HEALPix map.
	"""
	
	if os.path.splitext(filename)[1] == '.healnpy':
		map = numpy.load(filename)
	elif os.path.splitext(filename)[1] == '.fits':
		map = healpy.fitsfunc.read_map(filename)
	else:
		map = numpy.loadtxt(filename)
	wgt = map*0.0 + 1.0
	
	filename2 = filename.replace('map', 'wgt').replace('res', 'wgt').replace('err', 'wgt')
	if filename2 != filename and os.path.exists(filename2):
		if os.path.splitext(filename2)[1] == '.healnpy':
			wgt = numpy.load(filename2)
		else:
			wgt = numpy.loadtxt(filename2)
		map /= wgt
		
	mask = numpy.where( (wgt == 0) | ~numpy.isfinite(map), 1, 0 )
	
	map = healpy.ma(map)
	wgt = healpy.ma(wgt)
	map.mask = mask
	wgt.mask = mask
	
	return map, wgt


def get2DMollweideImage(map, xsize=800, coord=None, rot=None, flip='astro'):
	"""
	Given a map, return a 2-D image of the image in a Mollweide projection.  
	This function essentially implements to core functunality of the
	healpy.visufunc.mollview() function but allows the users to so all of
	the plotting/scaling.
	
	There are a few options to this function that have the same names/uses
	as in healpy.visufunc.mollview().  These are:
	  * xsize - output image size in pixels for the x coordinate, y will be
	            half this
	  * coord - sequence of 'G' (Galactic), 'E' (Ecliptic), or 'C' (celesital/
	            equatorial that convey what the map is and how to project it
	  * rot - rotation to apply to the map before projection.  This is a two
	          or three element tuple of longitude, latitude, and (optionally)
	          angle.  All values are assumed to be in degrees
	  * flip - east-west convention ('astro' or 'geo' used in the projection
	"""
	
	nSides = healpy.pixelfunc.npix2nside(map.size)
	f = lambda x,y,z: healpy.pixelfunc.vec2pix(nSides,x,y,z)
	
	p = healpy.projector.MollweideProj(coord=coord, rot=rot, flipconv=flip)
	p.set_proj_plane_info(xsize=xsize)
	img = p.projmap(map, f, coord=coord, rot=rot)
	
	return img


def changeSides(map, nSides):
	"""
	Take a HEALPix map upsample or downsample the map to make it have 
	'nSides' sides and conserve flux.
	"""
	
	nSidesOld = healpy.pixelfunc.npix2nside(map.size)
	
	if getattr(map, 'mask', None) is not None:
		map2 = healpy.pixelfunc.ud_grade(map.data, nSides)
		
		mask2 = healpy.pixelfunc.ud_grade(map.mask.astype(numpy.float64), nSides)
		
		map2 = healpy.ma(map2)
		map2.mask = numpy.where( mask2 >= 0.005, 1, 0 ).astype(numpy.bool)
	else:
		map2 = healpy.pixelfunc.ud_grade(map, nSides)
		
	return map2


def changeResolution(map, fwhmCurrent, fwhmNew):
	"""
	Given a HEALPix map with full width at half max resolution 'fwhmCurrent'
	degrees, smooth it to a new FWHM of 'fwhmNew' degrees.
	"""
	
	if fwhmNew <= fwhmCurrent:
		return map
	else:
		smth = numpy.sqrt( fwhmNew**2 - fwhmCurrent**2 )
		#smth /= 2*numpy.sqrt(2*numpy.log(2))
		
		if getattr(map, 'mask', None) is not None:
			data = map.data*1.0
			data[numpy.where(map.mask==1)] = 0
			map2 = healpy.smoothing(data, fwhm=smth, degree=True)
			mask2 = healpy.smoothing(map.mask.astype(numpy.float64), fwhm=smth, degree=True)
			
			map2 = healpy.ma(map2)
			map2.mask = numpy.where( mask2 >= 0.005, 1, 0 ).astype(numpy.bool)
		else:
			map2 = healpy.smoothing(map, fwhm=smth, degree=True)
			
		return map2 


def convertMapToGalactic(map):
	"""
	Given a map in equatorial coordinates, convert it to Galactic coordinates.
	"""
	
	rotCG = healpy.rotator.Rotator(coord=('C', 'G'))
	
	hasMask = False
	if getattr(map, 'mask', None) is not None:
		hasMask = True
		
	map2 = map*0.0
	nSides = healpy.pixelfunc.npix2nside(map.size)
	for i in xrange(map.size):
		theta,phi = healpy.pixelfunc.pix2ang(nSides, i)
		theta,phi = rotCG(theta,phi)
		j = healpy.pixelfunc.ang2pix(nSides, theta, phi)
		map2[j] += map[i]
		
		if hasMask:
			map2.mask[j] = map.mask[i]	
		
	return map2


def convertMapToEquatorial(map):
	"""
	Given a map in Galactic coordinates, convert it to equatorial coordinates.
	"""
	
	rotGC = healpy.rotator.Rotator(coord=('G', 'C'))
	
	hasMask = False
	if getattr(map, 'mask', None) is not None:
		hasMask = True
		
	map2 = map*0.0
	nSides = healpy.pixelfunc.npix2nside(map.size)
	for i in xrange(map.size):
		theta,phi = healpy.pixelfunc.pix2ang(nSides, i)
		theta,phi = rotGC(theta,phi)
		j = healpy.pixelfunc.ang2pix(nSides, theta, phi)
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
		
	map2 = map*0.0
	nSides = healpy.pixelfunc.npix2nside(map.size)
	for i in xrange(map.size):
		theta,phi = healpy.pixelfunc.pix2ang(nSides, i)
		eq = ephem.Equatorial(phi, numpy.pi/2-theta, epoch=ephem.B1950)
		eq = ephem.Equatorial(eq, epoch=ephem.J2000)
		j = healpy.pixelfunc.ang2pix(nSides, numpy.pi/2-eq.dec, eq.ra)
		map2[j] += map[i]
		
		if hasMask:
			map2.mask[j] = map.mask[i]
			
	return map2


try:
	# Newer versions of HEALpy
	from healpy._query_disc import query_disc as queryDisk
except ImportError:
	# Older version of healpy
	from healpy.query_disc_func import query_disc
	
	def queryDisk(nSides, vec, radius, degrees=False):
		"""
		Return a list of HEALPix pixels that lie with the specified radius (in 
		radians) for the given vector and number of sides.
		
		This function is wraps the HEALpy query_disc function and only changes 
		the command defaults.
		"""
		
		return query_disc(nSides, vec, radius, nest=False, deg=degrees)


def queryPolygon(nSides, polygon):
	"""
	Return a list of HEALPix pixels that lie with the specified polygon.  The
	polygon is specified as a list of two-element tuples of latitiude and
	longitude in radians.  For maps in equatorial coordinates these would be
	declination and right ascension.
	
	Based on:  http://forum.worldwindcentral.com/showthread.php?20739-Determining-whether-a-given-point-is-inside-or-outside-a-given-spherical-polygon
	"""
	
	# List of all pixels and their coordinates
	pixels = range(healpy.pixelfunc.nside2npix(nSides))
	theta, phi = healpy.pixelfunc.pix2ang(nSides, pixels)
	dec = numpy.pi/2 - theta
	ra = phi
	
	# List of all pixels that are inside - this is False initially
	inside = numpy.zeros(len(pixels), dtype=numpy.bool)
	
	# Loop over the polygon edges and count the number of crossings
	for i in xrange(1, len(polygon)):
		## Previous point
		dec1, ra1 = polygon[i-1]
		## Next point
		dec2, ra2 = polygon[i]
		
		## Crossing logic
		flip1 = numpy.logical_and(dec2 <= dec, dec < dec1)
		flip2 = numpy.logical_and(dec1 <= dec, dec < dec2)
		flip3 = numpy.logical_or(flip1, flip2)
		flip = numpy.logical_and(flip3, ra < (ra1-ra2)*(dec-dec2)/(dec1-dec2) + ra2)
		
		## Invert the crossers
		toFlip = numpy.where( flip )
		inside[toFlip] = numpy.logical_not(inside[toFlip])
		
	# Return a list of pixels that are on the inside
	return numpy.where( inside )[0]


def medianFilter(map, radius, innerRadius=0.0):
	"""
	Given a HEALPix map and a radius in degrees, apply a median filter to 
	the map.
	
	This function is based on the median_filter.pro utility that is part
	of the HEALPix package by Gorski, Hivon, and Banday.
	"""
	
	nSides = healpy.pixelfunc.npix2nside(map.size)
	nPix = map.size
	
	pbar = ProgressBar(max=nPix)
	
	out = map*0.0
	for i in xrange(nPix):
		vec = healpy.pixelfunc.pix2vec(nSides, i)
		vec = numpy.array(vec)
		inDisk = queryDisk(nSides, vec, radius*numpy.pi/180.0)
		if innerRadius > 0.0:
			inHole = queryDisk(nSides, vec, innerRadius*numpy.pi/180.0)
			inDisk = numpy.setdiff1d(inDisk, inHole)
			
		out[i] = numpy.ma.median( map[inDisk] )
		pbar += 1
		if i % 100 == 0:
			sys.stdout.write(pbar.show()+'\r')
			sys.stdout.flush()
	sys.stdout.write(pbar.show()+'\n')
	sys.stdout.flush()
	
	return out


def unsharpMask(map, fwhm, smoothFactor=7.0, scale=0.9, threshold=0.1):
	"""
	Given a map and a estimate of the FWHM in degrees, create and return an 
	unsharp masked version of the map.  The mask is controlled by three 
	keywords:  smoothFactor, scale, and threshold.  smoothFactor controls
	how the highpass filter is applied while scale and threshold control
	the level of masking.  For example:
	
	  highpass = conolve(map, fwhm -> smoothFactor*fwhm)
	  unsharp = map - scale*(map - highpass)*(|(map-highpass)/map|>=threshold)
	"""
	
	smth = changeResolution(map, fwhm, smoothFactor*fwhm)
	diff = map - smth
	usmd = map + scale*diff*(numpy.abs(diff/map)>=threshold)
	
	return usmd


def getMapValue(map, ra, dec, fwhm=2.2, hours=False, fit=False):
	"""
	Given a HEALPix map and a right ascension/declianation pair, return 
	the map value at that point.  Use the 'fhwm' keyword to provide the
	beam full width at have max in degrees
	"""
	
	if hours:
		ra *= 15.0
		
	nSide = healpy.pixelfunc.npix2nside(map.size)
	
	# Figure out the average pixel size in degrees
	pixSize = numpy.sqrt( 4*numpy.pi / map.size )*180/numpy.pi
	fwhmPix = fwhm / pixSize
	
	# Extract the region around the source
	vec = healpy.pixelfunc.ang2vec(numpy.pi/2-dec*numpy.pi/180, ra*numpy.pi/180)
	vec = numpy.array(vec)
	
	# Get the region around the source and an annulus to use for the sky
	outerPixels = queryDisk(nSide, vec, 2.5*fwhm, degrees=True)
	innerPixels = queryDisk(nSide, vec, 1.5*fwhm, degrees=True)
	sringPixels = [i for i in outerPixels if i not in innerPixels]
	
	# Extract the sky
	sregion = map[sringPixels]
	scoords = [healpy.pixelfunc.pix2ang(nSide, i) for i in sringPixels]
	scoords = numpy.array(scoords)
	scoords[:,0] = numpy.pi/2 - scoords[:,0]
	sky = numpy.median( sregion )
	
	if not fit:
		# Simple case, not fitting required
		flux = map[innerPixels].max()
		#flux = healpy.pixelfunc.get_interp_val(map, numpy.pi/2-dec*numpy.pi/180, ra*numpy.pi/180)
		flux -= sky
		
		err = numpy.std( map[sringPixels] )
		
	else:
		# Do the Gaussian fit
		region = map[innerPixels] - sky
		coords = [healpy.pixelfunc.pix2ang(nSide, i) for i in innerPixels]
		coords = numpy.array(coords)
		coords[:,0] = numpy.pi/2 - coords[:,0]
		
		def _func(p, x):
			height = p[0]
			cRA, cDec = p[1], p[2]
			width = p[3]
			
			ra = x[:x.size/2]
			dec = x[x.size/2:]
			
			d = [ephem.separation((r,d), (cRA,cDec)) for r,d in zip(ra,dec)]
			d = numpy.array(d)
			
			return height*numpy.exp( -d**2 / 2.0 / width**2 )
			
		def _err(p, x, data):
			return data - _func(p, x)
			
		p = [region.max(), ra*numpy.pi/180, dec*numpy.pi/180, fwhm*numpy.pi/180/2/numpy.sqrt(2*numpy.log(2))]
		x = list(coords[:,1])
		x.extend(coords[:,0])
		x = numpy.array(x)
		out = leastsq(_err, p, (x, region), full_output=True)
		fit, cov = out[0], out[1]
		rChi2 = (_err(fit, x, region)**2).sum() / (region.size - len(p))
		try:
			err = numpy.sqrt( cov*rChi2 )
			err = [err[i,i] for i in xrange(err.shape[0])]
		except TypeError:
			err = fit*2.0
			
		#pylab.figure()
		#pylab.tripcolor(coords[:,1]*180/numpy.pi, coords[:,0]*180/numpy.pi, region)
		#pylab.plot(fit[1]*180/numpy.pi, fit[2]*180/numpy.pi, marker='x')
		#pylab.draw()
		#pylab.show()
		
		flux, err = fit[0], err[0]
		
	return flux, err


def getMapValueVirA(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of VirA.
	"""
	
	flux, err = getMapValue(map, 12.51361111, 12.39111111, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValueTauA(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of TauA.
	"""
	
	flux, err = getMapValue(map, 5.57555556, 22.01444444, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValueCygA(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of CygA.
	"""
	
	flux, err = getMapValue(map, 19.99122222, 40.73388889, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValueCasA(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of CasA.
	"""
	
	flux, err = getMapValue(map, 23.39055556, 58.80000000, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValueHydA(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of HydA.
	"""
	
	flux, err = getMapValue(map, 9.30166667, -12.09583333, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValueHerA(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of HerA.
	"""
	
	flux, err = getMapValue(map, 16.85226389, 4.99258889, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValue3C48(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of 3C48.
	"""
	
	flux, err = getMapValue(map, 1.62808333, 33.15888889, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValue3C123(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of 3C123.
	"""
	
	flux, err = getMapValue(map, 4.61787972, 29.6704638, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValue3C147(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of 3C147.
	"""
	
	flux, err = getMapValue(map, 5.71002778, 49.85194444, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValue3C196(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of 3C196.
	"""
	
	flux, err = getMapValue(map, 8.22666667, 48.21750000, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValue3C270(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of 3C270.
	"""
	
	flux, err = getMapValue(map, 12.32311678, 5.82521530, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValue3C295(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of 3C295.
	"""
	
	flux, err = getMapValue(map, 14.18902778, 52.20277778, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValue3C353(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of 3C353.
	"""
	
	flux, err = getMapValue(map, 17.34113889, -0.97972222, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValue3C380(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of 3C380.
	"""
	
	flux, err = getMapValue(map, 18.49216667, 48.74611111, fwhm=fwhm, hours=True, fit=fit)
	
	return flux, err


def getMapValue3C134(map, fwhm=2.2, fit=True):
	"""
	Return the map value at the location of 3C134.
	"""
	
	flux, err = getMapValue(map, 76.17579, 38.10316, fwhm=fwhm, hours=False, fit=fit)
	
	return flux, err


def plotRADec(ra, dec, label=None, hours=False, color='w', marker='x'):
	"""
	Plot a right ascension/decliation pair on a HEALPix map.
	"""
	
	theta = numpy.pi/2 - dec*numpy.pi/180
	if hours:
		ra *= 15.0
	phi = ra*numpy.pi/180
	if ra > 180:
		phi -= 2*numpy.pi
		
	if marker is not None:
		p = healpy.projscatter([theta,], [phi,], coord='C', marker=marker, color=color)
	else:
		p = healpy.projscatter([theta,], [phi,], coord='C', marker='', color=color)
	p.set_offset_position('data')
	xy = p.get_offsets()
	
	if ra > 180 and ra < 195:
		phi = ra*numpy.pi/180 + 0.37
	else:
		phi = ra*numpy.pi/180 - 0.1
		
	if label is not None:
		p = healpy.projtext([theta,], [phi,], label, coord='C', color=color, fontsize=12)
		
	return xy


def plotBrightSources(color='w', marker=None):
	"""
	Plot a collection of bright sources onto a HEALPix map.  The sources include:
	  * Cyg A
	  * Cas A
	  * Tau A
	  * Vir A
	  * Her A
	  * Hyd A
	  * Sgr A
	  * Cen A
	  * Vela
	  * 3C48
	  * 3C123
	  * 3C147
	  * 3C196
	  * 3C270
	  * 3C295
	  * 3C353
	  * 3C380
	"""
	
	plotRADec(23.39055556, 58.80000000, hours=True, label='Cas A', marker=marker, color=color)
	plotRADec(19.99122222, 40.73388889, hours=True, label='Cyg A', marker=marker, color=color)
	plotRADec( 5.57555556, 22.01444444, hours=True, label='Tau A', marker=marker, color=color)
	plotRADec(17.76111111,-29.00780556, hours=True, label='Sgr A', marker=marker, color=color)
	plotRADec(12.51361111, 12.39111111, hours=True, label='Vir A', marker=marker, color=color)
	plotRADec(16.85222222,  4.99250000, hours=True, label='Her A', marker=marker, color=color)
	plotRADec( 9.30155556,-12.09555556, hours=True, label='Hyd A', marker=marker, color=color)
	plotRADec(13.42433333,-43.01911111, hours=True, label='Cen A', marker=marker, color=color)
	plotRADec( 8.56666667,-45.83333333, hours=True, label='Vela',  marker=marker, color=color)
	plotRADec( 1.62808333, 33.15888889, hours=True, label='3C48',  marker=marker, color=color)
	plotRADec( 4.61787972, 29.67046380, hours=True, label='3C123', marker=marker, color=color)
	plotRADec( 5.71002778, 49.85194444, hours=True, label='3C147', marker=marker, color=color)
	plotRADec( 8.22666667, 48.21750000, hours=True, label='3C196', marker=marker, color=color)
	plotRADec(12.32311678,  5.82521530, hours=True, label='3C270', marker=marker, color=color)
	plotRADec(14.18902778, 52.20277778, hours=True, label='3C295', marker=marker, color=color)
	plotRADec(17.34113889, -0.97972222, hours=True, label='3C353', marker=marker, color=color)
	plotRADec(18.49216667, 48.74611111, hours=True, label='3C380', marker=marker, color=color)


#fermiBubbleNorth = numpy.array([[ -9.403  ,   4.271  ],
#					 [-15.24   ,  12.81   ],
#					 [-14.15   ,  16.33   ],
#					 [-16.78   ,  23.36   ],
#					 [-23.17   ,  28.89   ],
#					 [-23.69   ,  31.9    ],
#					 [-19.88   ,  43.96   ],
#					 [ -0.05508,  52.51   ],
#					 [ 10.6    ,  46.48   ],
#					 [ 19.62   ,  31.9    ],
#					 [ 20.6    ,  16.83   ],
#					 [ 12.52   ,   5.778  ],
#					 [ -1.404  ,   0.2512 ],
#					 [ -9.403  ,   4.271  ]])
#					 
#fermiBubbleSouth = numpy.array([[ -6.914  , -57.04   ],
#					 [-18.08   , -51.54   ],
#					 [-25.     , -46.52   ],
#					 [-28.19   , -38.46   ],
#					 [-25.53   , -26.32   ],
#					 [-13.82   , -12.12   ],
#					 [  4.787  ,  -5.971  ],
#					 [ 12.76   , -17.04   ],
#					 [ 15.95   , -28.13   ],
#					 [ 14.89   , -34.2    ],
#					 [ 14.89   , -41.27   ],
#					 [ 15.95   , -47.32   ],
#					 [ 12.23   , -51.38   ],
#					 [  5.851  , -54.95   ],
#					 [ -6.914  , -57.04   ]])


fermiBubbleNorth = numpy.array([[345.1  ,  17.4  ],
					 [342.0  ,  25.5  ],
					 [339.1  ,  35.3  ],
					 [342.5  ,  44.8  ],
					 [  3.1  ,  47.7  ],
					 [ 14.9  ,  37.5  ],
					 [ 18.3  ,  30.0  ],
					 [ 16.8  ,  16.8  ]])
					 
fermiBubbleSouth = numpy.array([[ 11.7  , -17.1  ],
					 [ 13.4  , -25.0  ],
					 [ 15.1  , -35.0  ],
					 [  5.6  , -51.1  ],
					 [347.8  , -50.3  ],
					 [337.1  , -39.5  ],
					 [337.1  , -30.9  ],
					 [340.3  , -23.3  ]])


def plotFermiBubbles(color='w', marker='x', linestyle='--'):
	"""
	Plot an outline of the Fermi bubbles on a HEALPix map.
	"""
	
	#dataN = numpy.array([[ -9.403  ,   4.271  ],
	#				 [-15.24   ,  12.81   ],
	#				 [-14.15   ,  16.33   ],
	#				 [-16.78   ,  23.36   ],
	#				 [-23.17   ,  28.89   ],
	#				 [-23.69   ,  31.9    ],
	#				 [-19.88   ,  43.96   ],
	#				 [ -0.05508,  52.51   ],
	#				 [ 10.6    ,  46.48   ],
	#				 [ 19.62   ,  31.9    ],
	#				 [ 20.6    ,  16.83   ],
	#				 [ 12.52   ,   5.778  ],
	#				 [ -1.404  ,   0.2512 ],
	#				 [ -9.403  ,   4.271  ]])
	#				 
	#dataS = numpy.array([[ -6.914  , -57.04   ],
	#				 [-18.08   , -51.54   ],
	#				 [-25.     , -46.52   ],
	#				 [-28.19   , -38.46   ],
	#				 [-25.53   , -26.32   ],
	#				 [-13.82   , -12.12   ],
	#				 [  4.787  ,  -5.971  ],
	#				 [ 12.76   , -17.04   ],
	#				 [ 15.95   , -28.13   ],
	#				 [ 14.89   , -34.2    ],
	#				 [ 14.89   , -41.27   ],
	#				 [ 15.95   , -47.32   ],
	#				 [ 12.23   , -51.38   ],
	#				 [  5.851  , -54.95   ],
	#				 [ -6.914  , -57.04   ]])
	#				 
	#for data in (dataN, dataS):
	#	data *= numpy.pi/180
	#	theta = numpy.pi/2 - data[:,1]
	#	phi = data[:,0]
	#	
	#	healpy.projplot((theta, phi), coord='G', marker=marker, linestyle=linestyle, color=color)
		
		
	dataN = numpy.array([[345.1  ,  17.4  ],
					 [342.0  ,  25.5  ],
					 [339.1  ,  35.3  ],
					 [342.5  ,  44.8  ],
					 [  3.1  ,  47.7  ],
					 [ 14.9  ,  37.5  ],
					 [ 18.3  ,  30.0  ],
					 [ 16.8  ,  16.8  ]])
					 
	dataS = numpy.array([[ 11.7  , -17.1  ],
					 [ 13.4  , -25.0  ],
					 [ 15.1  , -35.0  ],
					 [  5.6  , -51.1  ],
					 [347.8  , -50.3  ],
					 [337.1  , -39.5  ],
					 [337.1  , -30.9  ],
					 [340.3  , -23.3  ]])
					 
	for data in (dataN, dataS):
		data *= numpy.pi/180
		theta = numpy.pi/2 - data[:,1]
		phi = data[:,0]
		
		healpy.projplot((theta, phi), coord='G', marker=marker, linestyle=linestyle, color=color)


def mollviewPlus(map=None, fig=None, rot=None, coord=None, unit='', xsize=800, title='Mollweide view', nest=False, min=None, max=None, flip='astro', remove_dip=False, remove_mono=False, gal_cut=0, format='%g', format2='%g', cbar=True, cmap=None, notext=False, norm=None, hold=False, margins=None, sub=None, cbarOrientation='horizontal'):
	"""
	Plot an healpix map (given as an array) in Mollweide projection.  This function 
	differs from the original healpy.visufunc.mollview function in that it provides 
	nicer colorbars.
	
	Input:
		- map : an ndarray containing the map
				if None, use map with inf value (white map), useful for
				overplotting
	Parameters:
		- fig: a figure number. Default: create a new figure
		- rot: rotation, either 1,2 or 3 angles describing the rotation
				Default: None
		- coord: either one of 'G', 'E' or 'C' to describe the coordinate
				system of the map, or a sequence of 2 of these to make
				rotation from the first to the second coordinate system.
				Default: None
		- unit: a text describing the unit. Default: ''
		- xsize: the size of the image. Default: 800
		- title: the title of the plot. Default: 'Mollweide view'
		- nest: if True, ordering scheme is NEST. Default: False (RING)
		- min: the minimum range value
		- max: the maximum range value
		- flip: 'astro' (default, east towards left, west towards right) or 'geo'
		- remove_dip: if True, remove the dipole+monopole
		- remove_mono: if True, remove the monopole
		- gal_cut: galactic cut for the dipole/monopole fit
		- format: the format of the scale label. Default: '%g'
		- format2: format of the pixel value under mouse. Default: '%g'
		- cbar: display the colorbar. Default: True
		- notext: if True, no text is printed around the map
		- norm: color normalization, hist= histogram equalized color mapping, log=
			logarithmic color mapping, default: None (linear color mapping)
		- hold: if True, replace the current Axes by a MollweideAxes.
				use this if you want to have multiple maps on the same
				figure. Default: False
		- sub: use a part of the current figure (same syntax as subplot).
				Default: None
		- margins: either None, or a sequence (left,bottom,right,top)
				giving the margins on left,bottom,right and top
				of the axes. Values are relative to figure (0-1).
				Default: None
		- cbarOrientation: how the colorbar should be oriented, either horizontal
						or vertical.  Default: horizontal
	"""
	
	# Create the figure
	if not (hold or sub):
		## Start fresh
		f = pylab.figure(fig,figsize=(8.5,5.4))
		extent = (0.02,0.05,0.96,0.9)
	elif hold:
		## Use the existing figure
		f = pylab.gcf()
		left,bottom,right,top = numpy.array(f.gca().get_position()).ravel()
		extent = (left,bottom,right-left,top-bottom)
		f.delaxes(f.gca())
	else:
		## Use the existing figure with subplots
		f = pylab.gcf()
		if hasattr(sub,'__len__'):
			nrows, ncols, idx = sub
		else:
			nrows, ncols, idx = sub/100, (sub%100)/10, (sub%10)
		if idx < 1 or idx > ncols*nrows:
			raise ValueError('Wrong values for sub: %d, %d, %d' % (nrows, ncols, idx))
			c,r = (idx-1)%ncols, (idx-1)/ncols
			if not margins:
				margins = (0.01,0.0,0.0,0.02)
			extent = (c*1./ncols+margins[0], 
					1.-(r+1)*1./nrows+margins[1],
					1./ncols-margins[2]-margins[0],
					1./nrows-margins[3]-margins[1])
			extent = (extent[0]+margins[0],
					extent[1]+margins[1],
					extent[2]-margins[2]-margins[0],
					extent[3]-margins[3]-margins[1])
					
	# Starting to draw : turn interactive off
	wasinteractive = pylab.isinteractive()
	pylab.ioff()
	
	try:
		if map is None:
			map = numpy.zeros(12) + numpy.inf
			cbar = False
			
		ax = healpy.projaxes.HpxMollweideAxes(f, extent, coord=coord, rot=rot, format=format2, flipconv=flip)
		f.add_axes(ax)
		
		if remove_dip:
			map = healpy.pixelfunc.remove_dipole(map, gal_cut=gal_cut, nest=nest, copy=True, verbose=True)
		elif remove_mono:
			map = healpy.pixelfunc.remove_monopole(map, gal_cut=gal_cut, nest=nest, copy=True, verbose=True)
			
		ax.projmap(map, nest=nest, xsize=xsize, coord=coord, vmin=min, vmax=max, cmap=cmap, norm=norm)
		
		# Pull out the value limits as used
		im = ax.get_images()[0]
		min, max = im.norm.vmin, im.norm.vmax
		
		# Deal with negative values for logging by switching over to the SymLogNorm class
		if norm == 'log' and min <= 0.0:
			max = numpy.max([abs(min), abs(max)])
			min = -max
			norm = SymLogNorm(10.0, clip=True)
			
			del ax.images[0]
			ax.projmap(map, nest=nest, xsize=xsize, coord=coord, vmin=min, vmax=max, cmap=cmap, norm=norm)
			
		# Add the colorbar, if requested
		if cbar:
			b = im.norm.inverse(numpy.linspace(0,1,im.cmap.N+1))
			v = numpy.linspace(im.norm.vmin,im.norm.vmax,im.cmap.N)
			
			norm2 = None
			if type(norm) is SymLogNorm:
				vminTick = 1.0
				vmaxTick = 10.0**numpy.ceil(numpy.log10(numpy.max([abs(min), abs(max)])))
				lctr = LogLocator(base=10.0, subs=[1.0,1.5,2.0,3.0,4.0,6.0,8.0])
				ticks = lctr.tick_values(vminTick, vmaxTick)
				if min <= 0.0:
					ticks = numpy.concatenate([ticks, [-t for t in ticks]])
					ticks = numpy.append(ticks, 0)
					ticks.sort()
				ticks = ticks[ numpy.where( (ticks>=min) & (ticks<=max) )[0] ]
				
				## Fix to make sure 8 and 10 don't collide in the tick marks
				scale = 10**int(numpy.log10(ticks[0]))
				diff = (numpy.log10(ticks) - numpy.log10(8))
				has8 = (diff == diff.astype(numpy.int32)).any()
				if has8 and ticks.max() >= 10*scale:
					ticks = ticks[numpy.where( ticks != 8*scale )]
					
				lctr = FixedLocator(ticks, nbins=5)
				
			elif norm == 'log':
				vminTick = 10.0**numpy.floor(numpy.log10(min))
				vmaxTick = 10.0**numpy.ceil(numpy.log10(max))
				lctr = LogLocator(base=10.0, subs=[1.0,1.5,2.0,3.0,4.0,6.0,8.0])
				ticks = lctr.tick_values(vminTick, vmaxTick)
				ticks = ticks[ numpy.where( (ticks>=min) & (ticks<=max) )[0] ]
				
				## Fix to make sure 8 and 10 don't collide in the tick marks
				scale = 10**int(numpy.log10(ticks[0]))
				diff = (numpy.log10(ticks) - numpy.log10(8))
				has8 = (diff == diff.astype(numpy.int32)).any()
				if has8 and ticks.max() >= 10*scale:
					ticks = ticks[numpy.where( ticks != 8*scale )]
					
				lctr = FixedLocator(ticks, nbins=5)
				
			elif norm == 'hist':
				ticks = [im.norm.inverse(i/4.0) for i in xrange(5)]
				lctr = FixedLocator(ticks, nbins=5)
				
			else:
				lctr = healpy.projaxes.BoundaryLocator(N=5)
				if issubclass(map.dtype.type, numpy.integer):
					## Special case for integers where we want a blocky colorbar
					b = numpy.arange(min, max+2) - 0.5
					norm2 = BoundaryNorm(b, cmap.N)
					ticks = range(min, max+2)
					while len(ticks) > 10:
						start = ticks[0]
						stop = ticks[-1]
						middle = ticks[2:-2:2]
						ticks = [start,]
						ticks.extend(middle)
						ticks.append(stop)
						
			cb = f.colorbar(im, ax=ax, orientation=cbarOrientation, shrink=0.75, aspect=25, ticks=lctr, pad=0.05, fraction=0.1, boundaries=b, values=v, format=format)
			if norm2 is not None:
				cax = cb.ax
				cb = ColorbarBase(cb.ax, orientation=cbarOrientation, ticks=ticks, boundaries=b, cmap=cmap, norm=norm2, format='%i')
				
		else:
			junk, junk2 = make_axes(ax, orientation=cbarOrientation, fraction=0.1, shrink=0.75, aspect=25, pad=0.05)
			junk.axis('off')
			
		ax.set_title(title)
		
		if not notext:
			ax.text(0.86, 0.05, ax.proj.coordsysstr, fontsize=14, fontweight='bold', transform=ax.transAxes)
			
		if cbar:
			cb.set_label(unit)
			
		f.sca(ax)
		
	finally:
		pylab.draw()
		if wasinteractive:
			pylab.ion()


class RestrictedOrthographicProj(healpy.projector.SphericalProj):
	"""
	Modified version of the healpy.projector.OthrographicProj class that provides the 
	projection with a restricted field of view.
	"""
	
	name = "RestrictedOrthographic"
	
	def __init__(self, rot=None, coord=None, xsize=800, fov=numpy.pi, **kwds):
		super(RestrictedOrthographicProj, self).__init__(rot=rot, coord=coord, xsize=xsize, fov=fov, **kwds)
		
	def set_proj_plane_info(self, xsize, fov):
		super(RestrictedOrthographicProj, self).set_proj_plane_info(xsize=xsize, fov=fov)
		
	def vec2xy(self, vx, vy=None, vz=None, direct=False):
		if not direct:
			theta,phi=healpy.rotator.vec2dir(self.rotator(vx,vy,vz))
		else:
			theta,phi=healpy.rotator.vec2dir(vx,vy,vz)
		if self.arrayinfo is None:
			raise TypeError("No projection plane array information defined for this projector")
		fov = self.arrayinfo['fov']
		flip = self._flip
		# set phi in [-pi,pi]
		phi = flip*(phi+numpy.pi)%(2*numpy.pi)-numpy.pi
		lat = numpy.pi/2. - theta
		x = numpy.cos(lat)*numpy.sin(phi)
		y = numpy.sin(lat)
		# unfold back of sphere
		cosc = numpy.cos(lat)*numpy.cos(phi)
		if numpy.any(cosc<0):
			hmask = (cosc<0)
			if hasattr(x,'__len__'):
				x[hmask] = numpy.nan
			elif hmask:
				x = numpy.nan
				
		mask = (numpy.asarray(x)**2+numpy.asarray(y)**2>1)
		
		if mask.any():
			if not hasattr(x,'__len__'):
				x = numpy.nan
				y = numpy.nan
			else:
				x[mask] = numpy.nan
				y[mask] = numpy.nan
		return x,y
	vec2xy.__doc__ = healpy.projector.SphericalProj.vec2xy.__doc__ % (name,name)
	
	def xy2vec(self, x, y=None, direct=False):
		if y is None:
			x,y = x
		if hasattr(x,'__len__'):
			x,y = numpy.asarray(x),numpy.asarray(y)
		if self.arrayinfo is None:
			raise TypeError("No projection plane array information defined for this projector")
		fov = self.arrayinfo['fov']
		flip = self._flip
		# re-fold back of sphere
		mask = None
		r = numpy.sqrt(x**2+y**2)
		if hasattr(r,'__len__'):
			r[(r > 1) | (numpy.abs(x) > numpy.sin(fov/2)) | (numpy.abs(y) > numpy.sin(fov/2))] = numpy.nan
		elif r > 1 or numpy.abs(x)>numpy.sin(fov/2) or numpy.abs(y)>numpy.sin(fov)/2:
			r = numpy.nan
		c = numpy.arcsin(r)
		if hasattr(y,'__len__'):
			y[numpy.abs(y)>numpy.sin(fov/2)] = numpy.nan
		elif numpy.abs(y)>numpy.sin(fov/2):
			y = numpy.nan
		lat = numpy.arcsin(y)
		phi = numpy.arctan2(x,numpy.cos(c))
		phi *= flip
		if not mask is None:
			if hasattr(phi,'__len__'):
				phi[mask] = numpy.pi-phi[mask]
			else: phi = numpy.pi-phi
		theta = numpy.pi/2. - lat
		vec = healpy.rotator.dir2vec(theta,phi)
		if not direct:
			return self.rotator.I(vec)
		else:
			return vec
	xy2vec.__doc__ = healpy.projector.SphericalProj.xy2vec.__doc__ % (name,name)
	
	def ang2xy(self, theta, phi=None, lonlat=False, direct=False):
		return self.vec2xy(healpy.rotator.dir2vec(theta,phi,lonlat=lonlat),direct=direct)
	ang2xy.__doc__ = healpy.projector.SphericalProj.ang2xy.__doc__ % (name,name)
	
	def xy2ang(self, x, y=None, lonlat=False, direct=False):
		return healpy.rotator.vec2dir(self.xy2vec(x,y,direct=direct),lonlat=lonlat)
	xy2ang.__doc__ = healpy.projector.SphericalProj.xy2ang.__doc__ % (name,name)

	def xy2ij(self, x, y=None):
		if self.arrayinfo is None:
			raise TypeError("No projection plane array information defined for this projector")
		xsize = self.arrayinfo['xsize']
		fov = self.arrayinfo['fov']
		ratio = 1.0
		ysize = xsize/ratio
		if y is None:
			x,y = numpy.asarray(x)
		else:
			x,y = numpy.asarray(x), numpy.asarray(y)
		xc,yc = (xsize-1.)/2., (ysize-1.)/2.
		if hasattr(x,'__len__'):
			mask = (numpy.abs(x) > numpy.sin(fov/2)) | (numpy.abs(y) > numpy.sin(fov/2))
			if not mask.any():
				mask = numpy.ma.nomask
			j=numpy.ma.array(numpy.around(x*xc/ratio+xc).astype(long),mask=mask)
			i=numpy.ma.array(numpy.around(yc+y*yc).astype(long),mask=mask)
		else:
			if (numpy.abs(x) > numpy.sin(fov/2)) or (numpy.abs(y) > numpy.sin(fov/2)):
				i,j,=numpy.nan,numpy.nan
			else:
				j = numpy.around(x*xc/ratio+xc).astype(long)
				i = numpy.around(yc+y*yc).astype(long)
		return i,j
	xy2ij.__doc__ = healpy.projector.SphericalProj.xy2ij.__doc__ % (name,name)
	
	def ij2xy(self, i=None, j=None):
		if self.arrayinfo is None:
			raise TypeError("No projection plane array information defined for this projector")
		xsize = self.arrayinfo['xsize']
		fov = self.arrayinfo['fov']
		ratio = 1.0
		ysize=xsize/ratio
		xc,yc=(xsize-1.)/2.,(ysize-1.)/2.
		if i is None and j is None:
			idx = numpy.outer(numpy.arange(ysize),numpy.ones(xsize))
			y = (idx-yc)/yc * numpy.sin(fov/2)
			idx = numpy.outer(numpy.ones(ysize),numpy.arange(xsize))
			x = ratio*(idx-xc)/xc * numpy.sin(fov/2)
		elif i is not None and j is not None:
			y = (numpy.asarray(i)-yc)/yc * numpy.sin(fov/2)
			x = ratio*(numpy.asarray(j)-xc)/xc * numpy.sin(fov/2)
			# if numpy.mod(x,1.0)**2+y**2 > 1.0: x,y=numpy.nan,numpy.nan
		elif i is not None and j is None:
			i,j = i
			y=(numpy.asarray(i)-yc)/yc * numpy.sin(fov/2)
			x=ratio*(numpy.asarray(j)-xc)/xc * numpy.sin(fov/2)
			# if numpy.mod(x,1.0)**2.+y**2 > 1.: x,y=numpy.nan,numpy.nan
		else:
			raise TypeError("i and j must be both given or both not given")
		mask = (x**2+y**2>1) | (numpy.abs(x) > numpy.sin(fov/2)) | (numpy.abs(y) > numpy.sin(fov/2))
		if not mask.any():
			mask=numpy.ma.nomask
		x = numpy.ma.array(x,mask=mask)
		y = numpy.ma.array(y,mask=mask)
		if len(x)==0:
			x = x[0]
		if len(y)==0:
			y = y[0]
		return x,y
	ij2xy.__doc__ = healpy.projector.SphericalProj.ij2xy.__doc__ % (name,name)
	
	def get_extent(self):
		if self.arrayinfo is None:
			raise TypeError("No projection plane array information defined for this projector")
		fov = self.arrayinfo['fov']
		ratio = numpy.sin(fov/2)
		return (-ratio,ratio,-ratio,ratio)
	get_extent.__doc__ = healpy.projector.SphericalProj.get_extent.__doc__


class RestrictedSphericalProjAxes(healpy.projaxes.SphericalProjAxes):
	"""Define a special Axes to take care of spherical projection.

	Input:
		- projection : a SphericalProj class or a class derived from it.
		- rot=, coord= : define rotation and coordinate system. See rotator.
		- coordprec= : number of digit after floating point for coordinates display.
		- format= : format string for value display.
		
		Other keywords from Axes (see Axes).
	"""
	
	def __init__(self, ProjClass, *args, **kwds):
		if not issubclass(ProjClass, healpy.projector.SphericalProj):
			raise TypeError("First argument must be a SphericalProj class (or derived from)")
		self.proj = ProjClass(rot=kwds.pop('rot',None), coord = kwds.pop('coord',None), flipconv=kwds.pop('flipconv',None), fov=kwds.pop('fov',numpy.pi), **kwds.pop('arrayinfo', {}))
		
		kwds.setdefault('format','%g')
		kwds.setdefault('coordprec',2)
		kwds['aspect'] = 'equal'
		super(healpy.projaxes.SphericalProjAxes, self).__init__(*args, **kwds)
		self.axis('off')
		self.set_autoscale_on(False)
		xmin,xmax,ymin,ymax = self.proj.get_extent()
		self.set_xlim(xmin,xmax)
		self.set_ylim(ymin,ymax)
		
		dx,dy = self.proj.ang2xy(numpy.pi/2., 1.*numpy.pi/180, direct=True)
		self._segment_threshold = 16.*numpy.sqrt(dx**2+dy**2)
		self._segment_step_rad = 0.1*numpy.pi/180
		self._do_border = True
		self._gratdef = {}
		self._gratdef['local'] = False
		self._gratdef['dpar'] = 1.
		self._gratdef['dmer'] = 1.


class RestrictedOrthographicAxes(RestrictedSphericalProjAxes):
	"""Define a FOV-restricted orthographic Axes to handle orthographic projection.
	
	Input:
	- rot=, coord= : define rotation and coordinate system. See rotator.
	- fov= : define the field of view in radians
	- coordprec= : num of digits after floating point for coordinates display.
	- format= : format string for value display.
	
	Other keywords from Axes (see Axes).
	"""
	
	def __init__(self, *args, **kwds):
		kwds.setdefault('coordprec', 2)
		kwds.setdefault('fov', numpy.pi)
		super(RestrictedOrthographicAxes, self).__init__(RestrictedOrthographicProj, *args, **kwds)
		self._segment_threshold = 0.01
		self._do_border = False
		
	def projmap(self, map, vec2pix_func, xsize=800, **kwds):
		fov = self.proj.arrayinfo['fov']
		self.proj.set_proj_plane_info(xsize=xsize, fov=fov)
		super(RestrictedOrthographicAxes, self).projmap(map, vec2pix_func, **kwds)
		ratio = numpy.sin(fov/2)
		self.set_xlim(-ratio,ratio)
		self.set_ylim(-ratio,ratio)


class HpxRestrictedOrthographicAxes(RestrictedOrthographicAxes):
	def projmap(self, map, nest=False, **kwds):
		fov = self.proj.arrayinfo['fov']
		nside = healpy.pixelfunc.npix2nside(len(map))
		f = lambda x,y,z: healpy.pixelfunc.vec2pix(nside, x, y, z, nest=nest)
		return super(HpxRestrictedOrthographicAxes,self).projmap(map, f, **kwds)
	

def regionviewPlus(map=None, fov=numpy.pi, fig=None, rot=None, coord=None, unit='', xsize=800, title='Mollweide view', nest=False, min=None, max=None, flip='astro', remove_dip=False, remove_mono=False, gal_cut=0, format='%g', format2='%g', cbar=True, cmap=None, notext=False, norm=None, hold=False, margins=None, sub=None, cbarOrientation='horizontal'):
	"""
	Plot a region of a healpix map (given as an array) in Orthographic projection.  This function 
	is based on the healpy.visufunc.mollview function .
	
	Input:
		- map : an ndarray containing the map
				if None, use map with inf value (white map), useful for
				overplotting
	Parameters:
		- fov: field of view in radians. Default = pi = half the sky
		- fig: a figure number. Default: create a new figure
		- rot: rotation, either 1,2 or 3 angles describing the rotation
				Default: None
		- coord: either one of 'G', 'E' or 'C' to describe the coordinate
				system of the map, or a sequence of 2 of these to make
				rotation from the first to the second coordinate system.
				Default: None
		- unit: a text describing the unit. Default: ''
		- xsize: the size of the image. Default: 800
		- title: the title of the plot. Default: 'Mollweide view'
		- nest: if True, ordering scheme is NEST. Default: False (RING)
		- min: the minimum range value
		- max: the maximum range value
		- flip: 'astro' (default, east towards left, west towards right) or 'geo'
		- remove_dip: if True, remove the dipole+monopole
		- remove_mono: if True, remove the monopole
		- gal_cut: galactic cut for the dipole/monopole fit
		- format: the format of the scale label. Default: '%g'
		- format2: format of the pixel value under mouse. Default: '%g'
		- cbar: display the colorbar. Default: True
		- notext: if True, no text is printed around the map
		- norm: color normalization, hist= histogram equalized color mapping, log=
			logarithmic color mapping, default: None (linear color mapping)
		- hold: if True, replace the current Axes by a MollweideAxes.
				use this if you want to have multiple maps on the same
				figure. Default: False
		- sub: use a part of the current figure (same syntax as subplot).
				Default: None
		- margins: either None, or a sequence (left,bottom,right,top)
				giving the margins on left,bottom,right and top
				of the axes. Values are relative to figure (0-1).
				Default: None
		- cbarOrientation: how the colorbar should be oriented, either horizontal
						or vertical.  Default: horizontal
	"""
	
	# Create the figure
	if not (hold or sub):
		## Start fresh
		f = pylab.figure(fig,figsize=(8.5,5.4))
		extent = (0.02,0.05,0.96,0.9)
	elif hold:
		## Use the existing figure
		f = pylab.gcf()
		left,bottom,right,top = numpy.array(f.gca().get_position()).ravel()
		extent = (left,bottom,right-left,top-bottom)
		f.delaxes(f.gca())
	else:
		## Use the existing figure with subplots
		f = pylab.gcf()
		if hasattr(sub,'__len__'):
			nrows, ncols, idx = sub
		else:
			nrows, ncols, idx = sub/100, (sub%100)/10, (sub%10)
		if idx < 1 or idx > ncols*nrows:
			raise ValueError('Wrong values for sub: %d, %d, %d' % (nrows, ncols, idx))
			c,r = (idx-1)%ncols, (idx-1)/ncols
			if not margins:
				margins = (0.01,0.0,0.0,0.02)
			extent = (c*1./ncols+margins[0], 
					1.-(r+1)*1./nrows+margins[1],
					1./ncols-margins[2]-margins[0],
					1./nrows-margins[3]-margins[1])
			extent = (extent[0]+margins[0],
					extent[1]+margins[1],
					extent[2]-margins[2]-margins[0],
					extent[3]-margins[3]-margins[1])
					
	# Starting to draw : turn interactive off
	wasinteractive = pylab.isinteractive()
	pylab.ioff()
	
	try:
		if map is None:
			map = numpy.zeros(12) + numpy.inf
			cbar = False
			
		ax = HpxRestrictedOrthographicAxes(f, extent, fov=fov, coord=coord, rot=rot, format=format2, flipconv=flip)
		f.add_axes(ax)
		
		if remove_dip:
			map = healpy.pixelfunc.remove_dipole(map, gal_cut=gal_cut, nest=nest, copy=True, verbose=True)
		elif remove_mono:
			map = healpy.pixelfunc.remove_monopole(map, gal_cut=gal_cut, nest=nest, copy=True, verbose=True)
			
		ax.projmap(map, nest=nest, xsize=xsize, coord=coord, vmin=min, vmax=max, cmap=cmap, norm=norm)
		
		# Pull out the value limits as used
		im = ax.get_images()[0]
		min, max = im.norm.vmin, im.norm.vmax
		
		# Deal with negative values for logging by switching over to the SymLogNorm class
		if norm == 'log' and min <= 0.0:
			max = numpy.max([abs(min), abs(max)])
			min = -max
			norm = SymLogNorm(10.0, clip=True)
			
			del ax.images[0]
			ax.projmap(map, nest=nest, xsize=xsize, coord=coord, vmin=min, vmax=max, cmap=cmap, norm=norm)
			
		# Add the colorbar, if requested
		if cbar:
			b = im.norm.inverse(numpy.linspace(0,1,im.cmap.N+1))
			v = numpy.linspace(im.norm.vmin,im.norm.vmax,im.cmap.N)
			
			norm2 = None
			if type(norm) is SymLogNorm:
				vminTick = 1.0
				vmaxTick = 10.0**numpy.ceil(numpy.log10(numpy.max([abs(min), abs(max)])))
				lctr = LogLocator(base=10.0, subs=[1.0,2.0,3.0,4.0,6.0,8.0])
				ticks = lctr.tick_values(vminTick, vmaxTick)
				if min <= 0.0:
					ticks = numpy.concatenate([ticks, [-t for t in ticks]])
					ticks = numpy.append(ticks, 0)
					ticks.sort()
				ticks = ticks[ numpy.where( (ticks>=min) & (ticks<=max) )[0] ]
				
				## Fix to make sure 8 and 10 don't collide in the tick marks
				scale = 10**int(numpy.log10(ticks[0]))
				diff = (numpy.log10(ticks) - numpy.log10(8))
				has8 = (diff == diff.astype(numpy.int32)).any()
				if has8 and ticks.max() >= 10*scale:
					ticks = ticks[numpy.where( ticks != 8*scale )]
					
				lctr = FixedLocator(ticks, nbins=5)
				
			elif norm == 'log':
				vminTick = 10.0**numpy.floor(numpy.log10(min))
				vmaxTick = 10.0**numpy.ceil(numpy.log10(max))
				lctr = LogLocator(base=10.0, subs=[1.0,2.0,3.0,4.0,6.0,8.0])
				ticks = lctr.tick_values(vminTick, vmaxTick)
				ticks = ticks[ numpy.where( (ticks>=min) & (ticks<=max) )[0] ]
				
				## Fix to make sure 8 and 10 don't collide in the tick marks
				scale = 10**int(numpy.log10(ticks[0]))
				diff = (numpy.log10(ticks) - numpy.log10(8))
				has8 = (diff == diff.astype(numpy.int32)).any()
				if has8 and ticks.max() >= 10*scale:
					ticks = ticks[numpy.where( ticks != 8*scale )]
					
				lctr = FixedLocator(ticks, nbins=5)
				
			elif norm == 'hist':
				ticks = [im.norm.inverse(i/4.0) for i in xrange(5)]
				lctr = FixedLocator(ticks, nbins=5)
				
			else:
				lctr = healpy.projaxes.BoundaryLocator(N=5)
				if issubclass(map.dtype.type, numpy.integer):
					## Special case for integers where we want a blocky colorbar
					b = numpy.arange(min, max+2) - 0.5
					norm2 = BoundaryNorm(b, cmap.N)
					ticks = range(min, max+2)
					while len(ticks) > 10:
						start = ticks[0]
						stop = ticks[-1]
						middle = ticks[2:-2:2]
						ticks = [start,]
						ticks.extend(middle)
						ticks.append(stop)
						
			cb = f.colorbar(im, ax=ax, orientation=cbarOrientation, shrink=0.75, aspect=25, ticks=lctr, pad=0.05, fraction=0.1, boundaries=b, values=v, format=format)
			if norm2 is not None:
				cax = cb.ax
				cb = ColorbarBase(cb.ax, orientation=cbarOrientation, ticks=ticks, boundaries=b, cmap=cmap, norm=norm2, format='%i')
				
		ax.set_title(title)
		
		if not notext:
			ax.text(0.86, 0.05, ax.proj.coordsysstr, fontsize=14, fontweight='bold', transform=ax.transAxes)
			
		if cbar:
			cb.set_label(unit)
			
		f.sca(ax)
		
	finally:
		pylab.draw()
		if wasinteractive:
			pylab.ion()