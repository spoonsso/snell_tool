# Snell library
# Timothy W. Dunn 12/5/17

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from skimage.morphology import binary_erosion
from matplotlib.patches import Polygon
from six.moves import cPickle
#import seaborn as sns
import math
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def find_nearest(array,value):
	"""
	Helper function to speed up the search for the index of the "nearest" input value in a sorted array
	"""
	idx = np.searchsorted(array, value, side="left")
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
		return idx-1
	else:
		return idx

def make_display(h=1, dSc = 0.1, res = 100000, ang = 90, dispres = 2000, dispcm = 3, ddisp = 0.5, nw=4/3,na=1):
	"""
	inputs---
		h: distance between eye and bottom of dish
		dSc: distance between dish and screen. dSC/h is a critical parameter of the system,
		res: resolution of spatial lookup table
		ang: max angle for spatial LUT
		dispres: virtual display resolution
		dispcm: virtual display physical dimensions, for one quadrant
		ddisp: absolute distance to virtual display, in cm
	"""
	spatLUT = np.zeros((res,))
	tres = np.linspace(0,ang,res)
	fresnel = np.zeros(res,)
	for t in range(spatLUT.shape[0]):
		theta_ = tres[t] * np.pi/180
		theta_prime = np.arcsin(nw*np.sin(theta_)/na)

		x_w = h*np.tan(theta_)
		x_a = dSc*np.tan(theta_prime)
		x = x_a + x_w
		xtan = np.arctan(x/(h+dSc))*180/np.pi
		
		Rs = abs((nw*np.cos(theta_prime) - na*np.cos(theta_))/(nw*np.cos(theta_prime) + na*np.cos(theta_)))**2
		Rp = abs((nw*np.cos(theta_) - na*np.cos(theta_prime))/(nw*np.cos(theta_) + na*np.cos(theta_prime)))**2
		
		fresnel[t] = 1-(Rs+Rp)/2
		
		spatLUT[t] = find_nearest(tres,xtan)#np.argmin(abs(xtan-tres))

	spatLUT_track = spatLUT.copy()
	spatLUT_track[spatLUT_track==0] = len(tres) - 1
	spatLUT_track[0] = 0
		
	spatLUT_inv = np.zeros((res,))
	spatLUT[spatLUT==0] = np.inf
	spatLUT[0] = 0
	for i in range(spatLUT_inv.shape[0]):
		spatLUT_inv[i] = tres[find_nearest(spatLUT,i)]#tres[np.argmin(abs(spatLUT-i))]
		
	display = np.zeros((dispres,dispres,2))
	fresdisplay = np.zeros((dispres,dispres))
	
	#snellsave = np.zeros((dispres,))
	for i in range(dispres):
		for j in range(dispres):

			# Get virtual pixel position in cm
			x_cm = j*dispcm/dispres
			y_cm = i*dispcm/dispres

			# Calculate distance from origin / vector magnitude
			d_cm = np.sqrt(x_cm**2 + y_cm**2)

			# Calculate unit vector
			unit_x = x_cm/d_cm
			unit_y = y_cm/d_cm

			# Get angle to point from fish perspective
			d_ang = np.arctan(d_cm/ddisp)*180/np.pi

			# Get angle after distortion by snell's law
			ind = find_nearest(tres, d_ang)#np.argmin(abs(d_ang-tres))
			snell_ang = spatLUT_inv[ind]
			#snellsave[j] = snell_ang
			# Save fresnel transmittance
			ind = find_nearest(tres, snell_ang)
			fresdisplay[i,j] = fresnel[ind]

			# Get new vector magnitude along screen
			snell_cm = ddisp*np.tan(snell_ang*np.pi/180)

			snell_x_cm = unit_x*snell_cm
			snell_y_cm = unit_y*snell_cm

			# Record new coordinate positions in pixel-space
			display[i,j,0] = snell_x_cm*dispres/dispcm
			display[i,j,1] = snell_y_cm*dispres/dispcm
		#return snellsave
	return display, tres, spatLUT_track, spatLUT_inv, fresnel, fresdisplay

def read_display(filename,fresnel=True):
	"""
	inputs---
		filename: name of pickle file where display variables are stored
	outputs---
		display: raw virtual display array that represents one quadrant of the snell's spatial lookup table
		tres: 1-D numpy array of equally angles from 0 to 90 degrees
		spatLUT: 1-D forward spatial lookup. Given an output angle from a pinhole, 
			what is the final resulting angle after refraction?
		spatLUT_inv: 1-D reverse spatial lookup. Numerical inverse of spatLUT. Given an angle formed between a pinhole and
			a point on the screen, what is the apparent angle to the pinhole resulting from refraction?
		h: height of pinhole in water
		dSc: width of air interface (distance between water and screen)
		dz: full display array
	"""
	f = open(filename,'rb')
	
	display = cPickle.load(f)
	tres = cPickle.load(f)
	spatLUT = cPickle.load(f)
	spatLUT_inv = cPickle.load(f)
	h = cPickle.load(f)
	dSc = cPickle.load(f)
	dz = cPickle.load(f)
	if fresnel:
		fresnel = cPickle.load(f)
		fresdisplay = cPickle.load(f)
		f.close()
		return display, tres, spatLUT, spatLUT_inv, h, dSc, dz, fresnel, fresdisplay

	f.close()
	
	return display, tres, spatLUT, spatLUT_inv, h, dSc, dz

def write_display(filename,display,tres,spatLUT,spatLUT_inv,h,dSc,dz,fresnel=None, fresdisplay=None):
	"""
	Writes relevant arrays and variables to pickle file
	
	inputs--
		filename: name of pickle file where display variables will be stored
		display: raw virtual display array that represents one quadrant of the snell's spatial lookup table
		tres: 1-D numpy array of equally angles from 0 to 90 degrees
		spatLUT: 1-D forward spatial lookup. Given an output angle from a pinhole, 
			what is the final resulting angle after refraction?
		spatLUT_inv: 1-D reverse spatial lookup. Numerical inverse of spatLUT. Given an angle formed between a pinhole and
			a point on the screen, what is the apparent angle to the pinhole resulting from refraction?
		h: height of pinhole in water
		dSc: width of air interface (distance between water and screen)
		dz: full display array
	"""
	f = open(filename,'wb')
	
	cPickle.dump(display,f)
	cPickle.dump(tres,f)
	cPickle.dump(spatLUT,f)
	cPickle.dump(spatLUT_inv,f)
	cPickle.dump(h,f)
	cPickle.dump(dSc,f)
	cPickle.dump(dz,f)
	if fresnel is not None:
		cPickle.dump(fresnel,f)
		cPickle.dump(fresdisplay,f)
		
	f.close()

def border_to_sphere(bord, h=333.33,center=2000):
	"""
	Takes the outline of an image on the screen and projects it through a pinhole onto a curved sensor (Retina)
	input--
		bord: 2-D numpy array, binary image border
		h: distance to screen, in pixels
	"""
	# Call the fish pupil the origin, then:
	circPoly = np.zeros((3,len(bord[0])))
	psitrack = np.zeros((len(bord[0]),))
	thetatrack = np.zeros((len(bord[0]),))
	R = 10
	for i in range(len(bord[0])):
		
		v1 = np.array([bord[1][i], bord[0][i]-center, -h]) #vector point from the fish's eye to a point on the spot

		#Standard spherical coordinate conversion:
		theta = np.arccos(v1[2]/np.sqrt(np.sum(v1**2)))
		psi = np.arctan(v1[1]/v1[0])
		# Express spherical coordinates as cartesian x,y,z coordinates
		circPoly[0,i] = R*np.sin(theta)*np.cos(psi);
		circPoly[1,i] = R*np.sin(theta)*np.sin(psi);
		circPoly[2,i] = R*np.cos(theta);

		psitrack[i] = psi
		thetatrack[i] = theta
	return circPoly, psitrack, thetatrack

def border_to_sphere_z(bord, h=333.33,center=2000):
	"""
	Takes the outline of an image on the screen and projects it through a pinhole onto a curved sensor (Retina)
	For this version of the function, the image plane is a *side-view*
	input--
		bord: 2-D numpy array, binary image border
		h: distance to screen, in pixels
	"""
	# Call the fish pupil the origin, then:
	circPoly = np.zeros((3,len(bord[0])))
	psitrack = np.zeros((len(bord[0]),))
	thetatrack = np.zeros((len(bord[0]),))
	R = 10
	for i in range(len(bord[0])):
		
		v1 = np.array([h,bord[1][i]-center,bord[0][i]-center]) #vector point from the fish's eye to a point on the spot

		#Standard spherical coordinate conversion:
		theta = np.arccos(v1[2]/np.sqrt(np.sum(v1**2)))
		psi = np.arctan(v1[1]/v1[0])
		# Express spherical coordinates as cartesian x,y,z coordinates
		circPoly[0,i] = R*np.sin(theta)*np.cos(psi);
		circPoly[1,i] = R*np.sin(theta)*np.sin(psi);
		circPoly[2,i] = R*np.cos(theta);

		psitrack[i] = psi
		thetatrack[i] = theta
	return circPoly, psitrack, thetatrack

def get_solid_angle(psi, theta):
	"""
	Calculate surface area on the unit sphere using a sinusoidal projection
	"""
	y = theta;
	x = psi * np.cos(theta);
	
	area = 0
	for i in range(-1, len(x)-1):
		area += x[i] * (y[i+1] - y[i-1])
		
	return abs(area) / 2

def get_border(im):
	"""
	Takes a binary image, returns the coordinates of the border pixels
	"""
	xb = binary_erosion(im)
	circb = np.zeros(im.shape)
	circb[xb] = 1
	yyy = im-circb
	return np.where(yyy==1)



def genRadFromRV(scDist, RV, tSim):
	"""Takes scDist and RV and returns absRad, where
	scDist is the distance, in cm, between the fish and the screen
	RV is the desired |R/V| ratio, in seconds
	tSim is the time before collision where the trace should start (i.e., for
	   a stimulus with a 10 second duration, tSim should be 10

	absRad is the spot radius, in cm, over time, in ms, that should be
	dislayed on the screen in order to simulate the desired R/V
	"""

	#generate a vector representing time in s from -tSim (seconds before time of collision (0 seconds)

	absRadt = np.linspace(-tSim,0,tSim*1000) 
	theta = 2*np.arctan(-RV/absRadt);
	absRad = scDist*np.tan(theta/2)

	return absRad, theta

def drawCirc(circ,r,h,k, mode='half'):
	"""
	Clumsily draws a filled circle into a 2-D numpy array

	inputs--
		circ: 2-D numpy array of zeros. should be the size of half the screen
		r: int, radius of circle in pixels
		h: int, x position of circle center
		k: int, y position of circle center
	"""
	# x, y = np.meshgrid(np.arange(circ.shape[1]),np.arange(circ.shape[0]))
	if mode == 'half':
		acirc = circ[np.max([0,k-r]):np.min([k+r+1,circ.shape[0]]), \
					np.max([0,h-r]):np.min([h+r+1,circ.shape[1]])]
		x, y = np.meshgrid(np.arange(acirc.shape[1]),np.arange(acirc.shape[0]))
		acirc[(x-(acirc.shape[1]-1)/2)**2+(y-(acirc.shape[0]-1)/2)**2<=r**2] = 1
	elif mode == 'full':
		x, y = np.meshgrid(np.arange(circ.shape[1]),np.arange(circ.shape[0]))
		circ[(x-h)**2+(y-k)**2<=r**2] = 1

	# for i in range(np.max([0,k-r]),np.min([k+r+1,circ.shape[0]])):#circ.shape[0]):
	# 	for j in range(np.max([0,h-r]),np.min([h+r+1,circ.shape[1]])):
	# 		if (i-k)**2 + (j-h)**2 <= r**2:
	# 			circ[i,j] = 1

def snell_distort(map_,im_,dispres,flag='ones'):
	"""
	Takes a snell distortion map and an input image (of the same dimensions), and distorts the image accordingly
	"""
	if flag == 'ones':
		ok_rows = map_[:,:,1]
		ok_rows = np.round(ok_rows[im_==1]).astype('int')
		ok_cols = map_[:,:,0]
		ok_cols = np.round(ok_cols[im_==1]).astype('int')
		
		distort = np.zeros(im_.shape)
		distort[(ok_rows,ok_cols)] = 1
		
	else:
		ok_rows = map_[:,:,1]
		ok_rows = np.round(ok_rows).astype('int')
		ok_rows[ok_rows<0] = dispres
		ok_cols = map_[:,:,0]
		ok_cols = np.round(ok_cols).astype('int')
		ok_cols[ok_cols<0] = dispres

		inds = np.ravel_multi_index((ok_rows.flatten(),ok_cols.flatten()),(map_.shape[:2]))
	
		distort = np.zeros(im_.shape)
		distort[(ok_rows.flatten(),ok_cols.flatten())] = np.reshape(im_,(distort.shape[0]*distort.shape[1]))
	
	return distort

def temizer(theta,r=17.5,l=2,d=8, na = 1, nw = 4/3):
	"""
	Given a list of angles, will run the Snell's model forward for the Temizer et al. geometry.

	For example, to look for the Snell's apparent angle for when a looming stimulus is 21.7/2 degrees,
		run this function and look for the corresponding entry in theta when theta_tild reaches 21.7/2
		degrees.

	inputs--
		theta: numpy array of input angles to be transformed.
		r: float; radius of embedding dish, in mm
		l: float; distance between fish and edge of dish, in mm
		d: float; distance between edge of dish and screen, in mm

	output--
		theta_tild: snell transformed apparent angles
	"""

	# These equations can be found in my hand written notes
	theta = theta * np.pi / 180

	phi = np.arcsin((r-l)/r * np.sin(np.pi-theta))

	phi_tild = np.arcsin(nw*np.sin(phi)/na)

	gam = np.pi/2 - phi_tild - theta + phi

	a = (r + d)/np.cos(theta-phi) - r

	s = a * np.sin(phi_tild)/np.sin(gam)
	s_tild = (r + d) * np.tan(theta-phi)

	theta_tild = np.arctan((s+s_tild)/(d+l))

	return theta_tild

def temizer_plastic(theta, r=16.5, r_p=1, l=2, d=8, na = 1, nw = 4/3, np_ = 1.55):
	"""
	Given a list of angles, will run the Snell's model forward for the Temizer et al. geometry, given an additional plastic interface.

	inputs--
		theta: numpy array of input angles to be transformed.
		r: float; inner radius of embedding dish, in mm
		r_p: float; thickness of the plastic on the edge of the dish, in mm
		l: float; distance between fish and edge of dish, in mm
		d: float; distance between edge of dish and screen, in mm

	output--
		theta_tild: snell transformed apparent angles
	"""
	R = r + r_p
	da = d
	dw = l - r_p
	dp = r_p
	theta = theta * np.pi / 180

	psi_w = np.arcsin((r - dw)/r * np.sin(theta))
	psi_p =  np.arcsin(nw/np_ * np.sin(psi_w))
	psi_p_prime = np.arcsin(nw*(r-dw)/(R*np_) * np.sin(theta))
	psi_a = np.arcsin(nw*(r-dw)/(R*na) * np.sin(theta))

	beta = np.pi - psi_p + psi_w - theta
	alpha = np.pi - beta - psi_p_prime

	a = (R + da) / np.cos(alpha) - R
	phi = np.pi / 2 - alpha
	omega = phi - psi_a


	s = a * np.sin(psi_a) / np.sin(omega)
	s_prime = (R + da) * np.tan(alpha) 
	theta_prime = np.arctan((s+s_prime)/(da+dw+dp))

	return theta_prime


def RFmap_pixel_noloop(center, width, stimpts,fr=None):
	"""
	inputs---
		center: RF center, in pixels
		width: RF width, in pixels
		stimpts: pts on screen being stimulated.
	"""
	sig = width

	RFout = stats.norm.pdf(
	stimpts[:,:,0] - center[0], loc=0, scale=sig) * stats.norm.pdf(
	stimpts[:,:,1] - center[1], loc=0, scale=sig) 
		
	if fr is not None:
		RFout = RFout*fr
	
	return RFout

# def RFmap_pixel_solidangle(center, width, stimpts,fr=None):
# 	"""
# 	inputs---
# 		center: RF center, in degrees in our sinusoidal projection
# 		width: RF width, in degrees
# 		stimpts: pts on screen being stimulated, in units of psi and theta.
# 	"""
# 	sig = width

# 	RFout = stats.norm.pdf(
# 	stimpts[:,:,0] - center[0], loc=0, scale=sig) * stats.norm.pdf(
# 	stimpts[:,:,1] - center[1], loc=0, scale=sig) 
		
# 	if fr is not None:
# 		RFout = RFout*fr
	
# 	return RFout

def RFmap_pixel_noloop_degrees(center, centerd, width, stimpts,ddisp,dispcm,fr=None):
	"""
	inputs---
		center: RF center, in pixels
		centerd: RF center, in degrees
		width: RF width, in degrees -- normal degrees along the diagonal from center
		stimpts: pts on screen being stimulated.
	"""
	
	maxwid = np.min([90, centerd + width/2])
	minwid = np.max([centerd - width/2])
	pixmaxwid = np.tan(maxwid*np.pi/180)*ddisp*stimpts.shape[1]/dispcm
	pixminwid = np.tan(minwid*np.pi/180)*ddisp*stimpts.shape[1]/dispcm

	sig = pixmaxwid - pixminwid

	print("center - pixminwid: {}".format(np.sqrt(center[0]**2 + center[1]**2) - pixminwid))
	print("pixmaxwid-center: {}".format(pixmaxwid-np.sqrt(center[0]**2 + center[1]**2)))

	RFout = stats.norm.pdf(
	stimpts[:,:,0] - center[0], loc=0, scale=sig) * stats.norm.pdf(
	stimpts[:,:,1] - center[1], loc=0, scale=sig) 
		
	if fr is not None:
		RFout = RFout*fr
	
	return RFout

def FWHM(im):
	"""
	Returns area, in pixels, of RF at half-max of the peak response intensity
	"""
	
	return len(np.where(im>=np.nanmax(im)/2)[0])

def FWHM_border(im,ddisp,dispres,dispcm, thresh=None):
	"""
	Returns area, in pixels, of RF at half-max of the peak response intensity -- in solid angle steradian
	"""
	
	if thresh is None:
		im_ = im>=np.nanmax(im)/2
	else:
		im_ = np.logical_and(im>=np.nanmax(im)/2,im>=thresh)

	border = get_border(im_)
	xxx, psi_, theta_ = border_to_sphere(border,h = round(ddisp*dispres/dispcm),center=dispres)

	# Sort psi, theta
	mean_y = np.mean(border[0])
	mean_x = np.mean(border[1])
	diff_y = border[0]-mean_y
	diff_x = border[1]-mean_x
	a = np.argsort(np.arctan2(diff_y,diff_x))

	psi_ = psi_[a]
	theta_ = theta_[a]

	return get_solid_angle(psi_, theta_-np.pi/2)

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
	"""
	Make a scatter of circles plot of x vs y, where x and y are sequence 
	like objects of the same lengths. The size of circles are in data scale.

	Parameters
	----------
	x,y : scalar or array_like, shape (n, )
		Input data
	s : scalar or array_like, shape (n, ) 
		Radius of circle in data unit.
	c : color or sequence of color, optional, default : 'b'
		`c` can be a single color format string, or a sequence of color
		specifications of length `N`, or a sequence of `N` numbers to be
		mapped to colors using the `cmap` and `norm` specified via kwargs.
		Note that `c` should not be a single numeric RGB or RGBA sequence 
		because that is indistinguishable from an array of values
		to be colormapped. (If you insist, use `color` instead.)  
		`c` can be a 2-D array in which the rows are RGB or RGBA, however. 
	vmin, vmax : scalar, optional, default: None
		`vmin` and `vmax` are used in conjunction with `norm` to normalize
		luminance data.  If either are `None`, the min and max of the
		color array is used.
	kwargs : `~matplotlib.collections.Collection` properties
		Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
		norm, cmap, transform, etc.

	Returns
	-------
	paths : `~matplotlib.collections.PathCollection`

	Examples
	--------
	a = np.arange(11)
	circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
	plt.colorbar()

	License
	--------
	This code is under [The BSD 3-Clause License]
	(http://opensource.org/licenses/BSD-3-Clause)
	"""
	import numpy as np
	import matplotlib.pyplot as plt

	if np.isscalar(c):
		kwargs.setdefault('color', c)
		c = None
	if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
	if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
	if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
	if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

	patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
	collection = PatchCollection(patches, **kwargs)
	if c is not None:
		collection.set_array(np.asarray(c))
		collection.set_clim(vmin, vmax)

	ax = plt.gca()
	ax.add_collection(collection)
	ax.autoscale_view()
	if c is not None:
		plt.sci(collection)
	return collection

def drawChecker(circ,r,h,k,max_=1, min_=-1,grids=8):
	"""
	Takes a numpy image array, creates circle mask and outputs an updated array with a windowed 
		grids x grids checkerboard inserted into the masked area

	"""

	drawCirc(circ,r,h,k)

	check = np.kron([[max_, min_] * (grids//2), [min_, max_] * (grids//2)] * (grids//2), np.ones((2*r//grids, 2*r//grids)))

	checks = np.ones_like(circ)*0
	checks_ = np.ones_like(circ)*0
	checks[k-r:k+r,h-r:h+r] = check
	checks = checks_ * (~circ.astype('bool')) + checks * circ

	return checks