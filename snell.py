import numpy as np
from six.moves import cPickle
import math
import scipy.ndimage as ndi
from skimage.transform import downscale_local_mean, rescale

class snell:
	"""
	Utilities for visualizing distorted images from the perspective of a fish in a dish. 

	Accompanies Dunn & Fitzgerald (2018).
	"""

	def __init__(self, config = 'flat', dispres = 2000, dispcm = 2, ddisp = 0.5, nw = 4/3, na = 1,
					dadw = 0.1, R = 1.75, L = 0.2 , D = 0.8, angres = 100000, maxang = 90):
		"""
		inputs---
			config: string indiciating setup geometry. Valid inputs: 'flat' or 'curved'
			dispres: virtual display resolution, in pixels, for one qudrant. Virtual screen is created via tiling.
			dispcm: virtual display width, in cm, for one quadrant. Virtual screen is created via tiling.
			ddisp: absolute distance to virtual display from fish, in cm
			nw: refractive index for water
			na: refractive index for air
			dadw: Used for 'flat' configurations only
				Ratio between the distance from the dish to the screen and the distance from the fish to the bottom of dish.
			R: Used for 'curved' configurations only
				Radius of the dish, in cm
`			L: Used for 'curved' configurations only
				Distance between the fish and the closest edge of the dish along a radius, in cm
			D: Used for 'curved' configurations only
				Distance between the fish and the screen along an extended radius, in cm
			angres: resolution of spatial lookup table
			maxang: max angle for spatial lookup table
		"""

		self.config = config
		self.dispres = dispres
		self.dispcm = dispcm
		self.ddisp = ddisp
		self.nw = nw
		self.na = na
		self.dadw = dadw
		self.R = R
		self.L = L
		self.D = D
		self.angres = angres
		self.maxang = maxang
		self.display = None
		self.fresdisplay = None

		if self.config is not 'flat' and self.config is not 'curved' and self.config is not 'flat_stochastic':
			raise Exception('Invalid configuration')


	def make_display(self):

		if self.config is 'flat':
			self.make_display_flat()
		elif self.config is 'curved':
			self.make_display_curved()
		elif self.config is 'flat_stochastic':
			self.make_display_stochastic_flat()
		else:
			raise Exception('Invalid configuration')

	def spat_fresnel_flat(self):
		spatLUT = np.zeros((self.angres,))
		tres = np.linspace(0,self.maxang,self.angres)
		fresnel = np.zeros(self.angres,)
		for t in range(spatLUT.shape[0]):
			theta_ = tres[t] * np.pi/180
			theta_prime = np.arcsin(self.nw*np.sin(theta_)/self.na)

			x_w = np.tan(theta_)
			x_a = self.dadw * np.tan(theta_prime)
			x = x_a + x_w
			xtan = np.arctan(x/(1 + self.dadw))*180/np.pi
			
			Rs = abs((self.nw*np.cos(theta_prime) - self.na*np.cos(theta_))/(self.nw*np.cos(theta_prime) + self.na*np.cos(theta_)))**2
			Rp = abs((self.nw*np.cos(theta_) - self.na*np.cos(theta_prime))/(self.nw*np.cos(theta_) + self. na*np.cos(theta_prime)))**2
			
			fresnel[t] = 1-(Rs+Rp)/2
			
			spatLUT[t] = find_nearest(tres,xtan)

		spatLUT_track = spatLUT.copy()
		spatLUT_track[spatLUT_track==0] = len(tres) - 1
		spatLUT_track[0] = 0
			
		spatLUT_inv = np.zeros((self.angres,))
		spatLUT[spatLUT==0] = np.inf
		spatLUT[0] = 0
		for i in range(spatLUT_inv.shape[0]):
			spatLUT_inv[i] = tres[find_nearest(spatLUT,i)]

		return spatLUT_inv, fresnel, tres

	def grid_to_display(self,j,i, spatLUT_inv, fresnel, tres):

		display = np.zeros((self.dispres,self.dispres,2))
		fresdisplay = np.zeros((self.dispres,self.dispres))

		# Get virtual pixel position in cm
		x_cm = j.ravel()*self.dispcm/self.dispres
		y_cm = i.ravel()*self.dispcm/self.dispres

		# Calculate distance from origin / vector magnitude
		d_cm = np.sqrt(x_cm**2 + y_cm**2)

		# Calculate unit vector
		unit_x = x_cm/d_cm
		unit_y = y_cm/d_cm

		# Get angle to point from fish perspective
		d_ang = np.arctan(d_cm/self.ddisp)*180/np.pi

		# Get angle after distortion by snell's law
		ind = find_nearest(tres, d_ang)
		snell_ang = spatLUT_inv[ind]
		#return ind, snell_ang, d_ang, d_cm, x_cm, y_cm, j, i

		# Save fresnel transmittance
		ind = find_nearest(tres, snell_ang)
		fresdisplay = np.reshape(fresnel[ind],(self.dispres,self.dispres))

		# Get new vector magnitude along screen
		snell_cm = self.ddisp*np.tan(snell_ang*np.pi/180)

		snell_x_cm = unit_x*snell_cm
		snell_y_cm = unit_y*snell_cm

		# Record new coordinate positions in pixel-space
		display[:,:,0] = np.reshape(snell_x_cm*self.dispres/self.dispcm,(self.dispres,self.dispres))
		display[:,:,1] = np.reshape(snell_y_cm*self.dispres/self.dispcm,(self.dispres,self.dispres))

		self.display = self.tile_display(display)
		self.fresdisplay = self.tile_fresdisplay(fresdisplay)

	def make_display_stochastic_flat(self):
		if self.config is not 'flat' and self.config is not 'flat_stochastic':
			raise Exception('Cannot make display, configuration is not flat')

		spatLUT_inv, fresnel, tres = self.spat_fresnel_flat()	

		j, i = np.meshgrid(np.arange(self.dispres),np.arange(self.dispres))

		j = j.astype('float64')
		i = i.astype('float64')

		#Shift all x and y coordinates by random float between 0 and 1
		j += np.random.rand(self.dispres,self.dispres)
		i += np.random.rand(self.dispres,self.dispres)

		self.grid_to_display(j,i, spatLUT_inv, fresnel, tres)

	def make_display_flat(self):
		"""
		Given properties established during instantiation (via __init__()), creates display variables for distorting images.

		This should be called via make_display(), not directly.
		"""
		if self.config is not 'flat':
			raise Exception('Cannot make display, configuration is not flat')

		spatLUT_inv, fresnel, tres = self.spat_fresnel_flat()	
	
		
		# for i in range(self.dispres):
		# 	for j in range(self.dispres):
		#i = np.arange(self.dispres)
		#j = np.arange(self.dispres)
		j, i = np.meshgrid(np.arange(self.dispres),np.arange(self.dispres))

		self.grid_to_display(j,i, spatLUT_inv, fresnel, tres)


	def make_display_curved(self, filename):
		"""
		"""
	
	def write_display(self,filename):
		"""
		Writes display variables to pickle file

		inputs--
			filename: name of pickle file where display variables will be stored
		"""

		f = open(filename,'wb')

		cPickle.dump(self.config,f)
		cPickle.dump(self.dispres,f)
		cPickle.dump(self.dispcm,f)
		cPickle.dump(self.ddisp,f)
		cPickle.dump(self.nw,f)
		cPickle.dump(self.na,f)
		cPickle.dump(self.dadw,f)
		cPickle.dump(self.R,f)
		cPickle.dump(self.L,f)
		cPickle.dump(self.D,f)
		cPickle.dump(self.angres,f)
		cPickle.dump(self.maxang,f)
		cPickle.dump(self.display,f)
		cPickle.dump(self.fresdisplay,f)

		f.close()

	def load_display(self, filename):
		"""
		Loads display variables from pickle file

		inputs--
			filename: name of pickle file where display variables are stored
		"""

		f = open(filename,'rb')

		self.config = cPickle.load(f)
		self.dispres = cPickle.load(f)
		self.dispcm = cPickle.load(f)
		self.ddisp = cPickle.load(f)
		self.nw = cPickle.load(f)
		self.na = cPickle.load(f)
		self.dadw = cPickle.load(f)
		self.R = cPickle.load(f)
		self.L = cPickle.load(f)
		self.D = cPickle.load(f)
		self.angres = cPickle.load(f)
		self.maxang = cPickle.load(f)
		self.display = cPickle.load(f)
		self.fresdisplay = cPickle.load(f)

		f.close()

	def tile_display(self, display):
		"""
		Tiles the virtual display with Snell-corrected spatial offsets
		"""
		dy = np.concatenate((display[-2::-1,:,:],display),axis=0)
		dy[:self.dispres,:,1] = self.dispres - dy[:self.dispres,:,1]
		dy[self.dispres:,:,1] = self.dispres + dy[self.dispres:,:,1]

		dz = np.concatenate((dy[:,-2::-1,:],dy[:,:,:]),axis=1)
		dz[:,:self.dispres,0] = self.dispres - dz[:,:self.dispres,0]
		dz[:,self.dispres:,0] = self.dispres + dz[:,self.dispres:,0]

		dz[np.isnan(dz)] = self.dispres

		return dz

	def tile_fresdisplay(self, fresdisplay):
		"""
		Tiles the virtual display with fractions of transmittance (Fresnel)
		"""

		bright = fresdisplay
		bleft = fresdisplay[:,-2::-1]
		tleft = fresdisplay[-2::-1,::-1]
		tright = tleft[:,-2::-1]

		fr_b = np.concatenate((bleft,bright),axis=1)
		fr_t = np.concatenate((tleft,tright),axis=1)
		fr = np.concatenate((fr_t,fr_b),axis=0)

		return fr

	def transform_image(self, im, smooth=True, downsample=False,stochastic=False, upsample=False):
		"""
		Transforms/distorts input image using the spatial lookup table in self.display and Fresnel transmittance in
			self.fresdisplay

		inputs--
			im: 2-D numpy array. Note that image dimensions N x M must equal the dimensions
				of self.fresdisplay and the first two dimensions of self.display
		"""
		weights = im.flatten() * self.fresdisplay.ravel()

		if stochastic:
			weights[np.random.choice(len(weights),len(weights)//stochastic,replace=False)] = 0

		im_snell = np.histogram2d(np.round(self.display[:,:,0]).astype('int').ravel(),
                              np.round(self.display[:,:,1]).astype('int').ravel(),
                              np.arange(0,self.display.shape[0]+1),
                              weights = weights)

		#im_snell = im_snell[0].T * self.fresdisplay
		im_snell = im_snell[0].T

		im_snell[im_snell==0] = np.mean(im)

		if smooth:
			return np.log(ndi.filters.gaussian_filter(im_snell,5))
		if downsample and not upsample:
			return np.log(downscale_local_mean(im_snell,(downsample,downsample)))
		if downsample and upsample:
			return np.log(rescale(downscale_local_mean(im_snell,(downsample,downsample)),(upsample,upsample)))
		else:
			return np.log(im_snell)

	def init_inverse_transform(self):
		"""
		Initializes data structure necessary for performing an inverse image transform
		"""

		self.inverse = {}

		source_display = np.round(self.display).astype('int')

		for i in range(source_display.shape[0]):
			for j in range(source_display.shape[1]):
				this_pos_x = source_display[i,j,0]
				this_pos_y = source_display[i,j,1]
				if (this_pos_x, this_pos_y) in self.inverse.keys():
					self.inverse[(this_pos_x, this_pos_y)][0].append(j)
					self.inverse[(this_pos_x, this_pos_y)][1].append(i)
				else:
					self.inverse[(this_pos_x, this_pos_y)] = [[j],[i]]

	def inverse_transform_image(self,im,smooth=True,downsample=False,upsample=False):
		"""
		Given a target image post-distortion, returns the input image that should be projected onto the screen

		inputs--
			im: 2-D numpy array. Note that image dimensions N x M must equal the dimensions
				of self.fresdisplay and the first two dimensions of self.display
		"""

		input_im = np.zeros_like(im)

		#Only loop over pixels in the Snell window
		snell_w = int(round(np.tan(49*np.pi/180)*self.ddisp*self.dispres/self.dispcm))
		center_x = int(round(input_im.shape[1]/2))
		center_y = int(round(input_im.shape[0]/2))

		x_min = center_x - snell_w
		x_max = center_x + snell_w + 1
		y_min = center_y - snell_w
		y_max = center_y + snell_w + 1

		source_display = np.round(self.display).astype('int')
		for i in range(x_min,x_max):
			for j in range(y_min,y_max):
				#source = np.where(np.logical_and(source_display[:,:,1]==i,source_display[:,:,0]==j))
				if (i,j) in self.inverse.keys():
					source = self.inverse[(i,j)]
					input_im[source] += im[j,i]/len(source[0])

		if smooth:
			return ndi.filters.gaussian_filter(input_im.T,5)
		if downsample and not upsample:
			return downscale_local_mean(input_im.T,(downsample,downsample))
		if downsample and upsample:
			return rescale(downscale_local_mean(input_im.T,(downsample,downsample)),(upsample,upsample))
		else:
			return input_im.T

def find_nearest(array,value):
	"""
	Helper function to speed up the search for the index of the "nearest" input value in a sorted array
	"""
	idx = np.searchsorted(array, value, side="left")

	if type(value) is np.ndarray:
		corr = np.logical_or(idx==len(array),np.less(np.abs(value-array[idx-1]),np.abs(value-array[idx])))
		corr = np.logical_and(corr,idx>0)
		idx[corr] = idx[corr] - 1

		return idx
	else:
		if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
			return idx-1
		else:
			return idx