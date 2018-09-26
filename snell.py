import numpy as np
from six.moves import cPickle
import math
import scipy.ndimage as ndi
from skimage.transform import downscale_local_mean, rescale
import sys

import warnings

class snell:
	"""
	Utilities for visualizing distorted images from the perspective of a fish in a dish. 

	Accompanies Dunn & Fitzgerald (2018).
	"""

	def __init__(self, config = 'flat', dispres = 2000, dispcm = 2, ddisp = 0.5, nw = 4/3, na = 1,
					dadw = 0.1, R = 1.75, L = 0.2 , D = 0.8, angres = 100000, maxang = 90, stoch_range=1, 
					num_stochastic=1,supersample_deg = 4):
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
			stoch_range: the range over which random offsets are added to pixel indicies to combat anti-aliasing.
				Only used when when config == 'flat_stochastic'
			num_stochastic: number of stochastic displays to use for transformation in order to reduce noise
		"""

		self.config = config
		self.supersample_deg = supersample_deg
		self.dispres = dispres*supersample_deg//2 #This sets our degree of supersampling
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

		self.stoch_range = stoch_range
		self.num_stochastic = num_stochastic
		self.stoch_count = 0
		self.display = np.zeros((self.dispres*2-1,self.dispres*2-1,2,num_stochastic))
		self.fresdisplay = np.zeros((self.dispres*2-1,self.dispres*2-1,num_stochastic))

		if self.config is not 'flat' and self.config is not 'curved' and self.config is not 'flat_stochastic':
			raise Exception('Invalid configuration')


	def make_display(self):

		if self.config is 'flat':
			self.make_display_flat()
		elif self.config is 'curved':
			self.make_display_curved()
		elif self.config is 'flat_stochastic':
			while self.stoch_count < self.num_stochastic:
				sys.stdout.write('start[' + 'X'*(self.stoch_count+1) + ' '*(self.num_stochastic-self.stoch_count+1) + ']end\r')
				sys.stdout.flush()
				self.make_display_stochastic_flat()
		else:
			raise Exception('Invalid configuration')

	def spat_fresnel_flat(self):
		spatLUT = np.zeros((self.angres,))
		tres = np.linspace(0,self.maxang,self.angres)
		fresnel = np.zeros(self.angres,)
		for t in range(spatLUT.shape[0]):
			theta_ = tres[t] * np.pi/180
			if theta_ <= 48.5*np.pi/180: # size of the snell window
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

		self.display[:,:,:,self.stoch_count] = self.tile_display(display)
		self.fresdisplay[:,:,self.stoch_count] = self.tile_fresdisplay(fresdisplay)
		self.stoch_count = self.stoch_count + 1

	def make_display_stochastic_flat(self):
		if self.config is not 'flat' and self.config is not 'flat_stochastic':
			raise Exception('Cannot make display, configuration is not flat')

		spatLUT_inv, fresnel, tres = self.spat_fresnel_flat()	

		j, i = np.meshgrid(np.arange(self.dispres),np.arange(self.dispres))

		j = j.astype('float64')
		i = i.astype('float64')

		#Shift all x and y coordinates by random float between -stoch_range//2 and stoch_range//2
		j += np.random.rand(self.dispres,self.dispres)*self.stoch_range #- self.stoch_range//2
		i += np.random.rand(self.dispres,self.dispres)*self.stoch_range #- self.stoch_range//2

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
		cPickle.dump(self.stoch_range,f)
		cPickle.dump(self.num_stochastic,f)
		cPickle.dump(self.stoch_count,f)

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
		self.stoch_range = cPickle.load(f)
		self.num_stochastic = cPickle.load(f)
		self.stoch_count = cPickle.load(filename)

		f.close()

	def write_inverse_init(self,filename):

		f = open(filename,'wb')

		cPickle.dump(self.inverses,f)

		f.close()

	def load_inverse_init(self,filename):

		f = open(filename,'rb')

		self.inverses = cPickle.load(f)

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

	def transform_images_loop(self, im, smooth=False, downsample=False,stochastic=False, upsample=False, noexp=False, surround=True):
		# make sure image has no negative values, and also transform to exponential space
		if not noexp:
			im = process_im(im)

		# here we also need to upsample the image to match the backend display size
		im = rescale(im,(self.supersample_deg,self.supersample_deg))
		im = im[:-1,:-1]

		im_avg = np.zeros((self.display.shape[0], self.display.shape[1], self.display.shape[3]))

		if downsample and not upsample:
			im_avg = downscale_local_mean(im_avg,(downsample,downsample,1))
		for i in range(self.display.shape[3]):
			sys.stdout.write('start[' + 'X'*(i+1) + ' '*(self.display.shape[3]-i+1) + ']end\r')
			sys.stdout.flush()
			im_avg[:,:,i] = self.transform_image(im,smooth=smooth,downsample=downsample,
				stochastic=stochastic,upsample=upsample, disp_ind=i, surround=surround)

		return downscale_local_mean(np.mean(im_avg,axis=2),(self.supersample_deg,self.supersample_deg))

	def transform_image(self, im, smooth=True, downsample=False,stochastic=False, upsample=False,disp_ind = 0,surround=True):
		"""
		Transforms/distorts input image using the spatial lookup table in self.display and Fresnel transmittance in
			self.fresdisplay

		inputs--
			im: 2-D numpy array. Note that image dimensions N x M must equal the dimensions
				of self.fresdisplay and the first two dimensions of self.display
		"""
		weights = im.flatten() * self.fresdisplay[:,:,disp_ind].ravel()

		if stochastic:
			weights[np.random.choice(len(weights),len(weights)//stochastic,replace=False)] = 0

		im_snell = np.histogram2d(np.round(self.display[:,:,0,disp_ind]).astype('int').ravel(),
                              np.round(self.display[:,:,1,disp_ind]).astype('int').ravel(),
                              np.arange(0,self.display.shape[0]+1),
                              weights = weights)

		#im_snell = im_snell[0].T * self.fresdisplay
		im_snell = im_snell[0].T

		if surround:
			im_snell[im_snell==0] = np.mean(im)

		if smooth:
			return np.log(ndi.filters.gaussian_filter(im_snell,5))
		if downsample and not upsample:
			return np.log(downscale_local_mean(im_snell,(downsample,downsample)))
		if downsample and upsample:
			return np.log(rescale(downscale_local_mean(im_snell,(downsample,downsample)),(upsample,upsample)))
		if surround:
			return np.log(im_snell)
		else:
			return im_snell

	def init_inverse_transform_loop(self):
		"""
		Calls init_inverse_transform in a loop for each of the stochastic sub displays
		"""
		self.inverses = []
		print("Creating inverse lookup tables...")
		for i in range(self.display.shape[3]):
			print("{}/{} ...".format(i+1,self.display.shape[3]))
			self.inverses.append(self.init_inverse_transform(i))

	def init_inverse_transform(self, disp_ind = 0):
		"""
		Initializes data structure necessary for performing an inverse image transform
		"""

		inverse = {}

		source_display = np.round(self.display).astype('int')
		max_cnt = source_display.shape[0]//100
		for i in range(source_display.shape[0]):
			cnt = i//100

			sys.stdout.write('start[' + 'X'*(cnt+1) + ' '*(max_cnt-cnt+1) + ']end\r')
			sys.stdout.flush()
			for j in range(source_display.shape[1]):
				this_pos_x = source_display[i,j,0, disp_ind]
				this_pos_y = source_display[i,j,1, disp_ind]
				if (this_pos_x, this_pos_y) in inverse.keys():
					inverse[(this_pos_x, this_pos_y)][0].append(j)
					inverse[(this_pos_x, this_pos_y)][1].append(i)
				else:
					inverse[(this_pos_x, this_pos_y)] = [[j],[i]]
		sys.stdout.write('\n')
		sys.stdout.flush()			
		return inverse


	def check_inverse(self,im):
		# First make sure there are no negative values, as with forward transforms:
		im = process_im(im)

		# Easier to check for non-zero values in linear space
		im = np.log(im)

		#Proceed if no exception was thrown by process_im, check if anything non-zero outside of snell window
		# Convert every screen position to an angle from the center
		xx, yy = np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[0]))
		dist = np.sqrt((yy-im.shape[0]//2)**2 + (xx-im.shape[1]//2)**2)
		ang = np.arctan(dist*self.dispcm/self.dispres/self.ddisp)*180/np.pi
		if np.sum(im*(ang>48)) > 0:
			warnings.warn("Attempting to generate images outside of the Snell window. Returning cropped input for inspection")
			return np.exp(im*(ang<=48))

		#If all good:
		return None

	def recon_error(self, out, ref):
		"""
		Transforms "out" forward and returns mean squared error between the transformed image and the reference (i.e. target)
		"""
		out = self.transform_images_loop(out,noexp=True, surround=False)

		return np.mean(np.abs(out-ref))/np.mean(ref)

	def inverse_transform_image_loop(self,im,smooth=False,downsample=False,upsample=False):

		ref = im.copy()
		im = rescale(im,(self.supersample_deg,self.supersample_deg))
		im = im[:-1,:-1]

		im_avg = np.zeros((self.display.shape[0], self.display.shape[1], self.display.shape[3]))

		if downsample and not upsample:
			im_avg = downscale_local_mean(im_avg,(downsample,downsample,1))

		print("inverting image...")
		for i in range(self.display.shape[3]):
			print("{}/{} ...".format(i+1,self.display.shape[3]))
			im_avg[:,:,i] = self.inverse_transform_image(im,smooth=smooth,downsample=downsample,
				upsample=upsample, disp_ind=i)
		print("Done.")

		output = np.mean(im_avg,axis=2)
		output = downscale_local_mean(output,(self.supersample_deg,self.supersample_deg))
		print("Calculating reconstruction error...")
		print("mean absolute error (proportion of target):{}".format(self.recon_error(output,ref)))


		return output

	def inverse_transform_image(self,im,smooth=True,downsample=False,upsample=False,disp_ind=0):
		"""
		Given a target image post-distortion, returns the input image that should be projected onto the screen

		inputs--
			im: 2-D numpy array. Note that image dimensions N x M must equal the dimensions
				of self.fresdisplay and the first two dimensions of self.display
		"""

		#First, perform several checks on the target image to make sure, e.g. it is all inside the snell window
		im_crop = self.check_inverse(im)
		if im_crop is not None:
			#Return cropped image. im_crop should be None if everything is OK
			return im_crop

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
		max_cnt = (x_max-x_min)//100
		for i in range(x_min,x_max):
			cnt = (i-x_min)//100
			sys.stdout.write('start[' + 'X'*(cnt+1) + ' '*(max_cnt-cnt+1) + ']end\r')
			sys.stdout.flush()
			for j in range(y_min,y_max):
				if (i,j) in self.inverses[disp_ind].keys():
					source = self.inverses[disp_ind][(i,j)]
					input_im[source] += im[j,i]/len(source[0])

		sys.stdout.write('\n')
		sys.stdout.flush()

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

def process_im(im):
	"""
	Make sure im has no negative values, then transform to exp space
	"""

	if np.min(im) < 0:
		raise Exception("Detected negative values in input image. All pixels values must be >= 0")

	return np.exp(im)

