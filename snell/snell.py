import numpy as np
from six.moves import cPickle
import math
import scipy.ndimage as ndi
from skimage.transform import downscale_local_mean, rescale
from skimage.measure import label
from skimage.filters import gaussian
import sys

import matplotlib.pyplot as plt

import warnings

class snell:
	"""
	Utilities for visualizing distorted images from the perspective of a fish in a dish. 

	Accompanies Dunn & Fitzgerald (2019).
	"""

	def __init__(self, config = 'flat_stochastic', dispres = 2000, dispcm = 2, nw = 4/3, na = 1.,
					da = 0.1, dw = 1., angres = 100000, maxang = 90, stoch_range=2, 
					num_stochastic=5,supersample_deg = 4):
		"""
		inputs---
			config: string indiciating setup geometry. Valid inputs: 'flat' or 'flat_stochastic'
			dispres: virtual display resolution, in pixels.
			dispcm: virtual display width, in cm.
			nw: refractive index for water
			na: refractive index for air
			da: distance, in cm, from the dish to the screen
			dw: distance, in cm, from the fish to the bottom of dish.
			angres: resolution of spatial lookup table
			maxang: max angle for spatial lookup table
			stoch_range: the range over which random offsets are added to pixel indicies to combat anti-aliasing.
				Only used when when config == 'flat_stochastic'
			num_stochastic: number of stochastic displays to use for transformation in order to reduce noise
		"""

		self.config = config
		self.supersample_deg = supersample_deg
		self.dispres = dispres*supersample_deg//2 #This sets our degree of supersampling
		self.dispcm = dispcm/2
		self.ddisp = da + dw
		self.nw = nw
		self.na = na
		self.dadw = da/dw
		self.da = da
		self.dw = dw
		self.angres = angres
		self.maxang = maxang

		self.stoch_range = stoch_range
		self.num_stochastic = num_stochastic
		self.stoch_count = 0
		self.display = np.zeros((self.dispres*2-1,self.dispres*2-1,2,num_stochastic))
		self.fresdisplay = np.zeros((self.dispres*2-1,self.dispres*2-1,num_stochastic))

		if self.config is not 'flat' and self.config is not 'flat_stochastic':
			raise Exception('Invalid configuration')

	def make_display(self):

		if self.config is 'flat':
			self.make_display_flat()
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
				
				Rp = abs((self.nw*np.cos(theta_prime) - self.na*np.cos(theta_))/(self.nw*np.cos(theta_prime) + self.na*np.cos(theta_)))**2
				Rs = abs((self.nw*np.cos(theta_) - self.na*np.cos(theta_prime))/(self.nw*np.cos(theta_) + self. na*np.cos(theta_prime)))**2
				
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

	def grid_to_display(self,j,i, spatLUT_inv, fresnel, tres, trans_matrix = False):

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

	def add_scale(self,lw=1):
		"""
		Once called, adds a scale bar to the active figure based on the virtual display geometry
		"""

		#The size of our image/display:
		native_dispres = self.dispres*2//self.supersample_deg

		#Plot bar with length equal to 1/5 native_dispres
		# plt.plot([4*native_dispres//5-1, native_dispres-1],[9*native_dispres//10,9*native_dispres//10],'w',linewidth=5)
		# ang_scale = 2*np.arctan(native_dispres//5/2/(self.ddisp*native_dispres/(self.dispcm*2)))
		# plt.title("Transformed image. Scale bar = {0:.1f} degrees".format(ang_scale*180/np.pi))

		snell_rad_pix = self.ddisp*np.tan(97.2/2*np.pi/180)*native_dispres/(self.dispcm*2)
		sn_window = plt.Circle((500,500),snell_rad_pix, edgecolor='y',fill=False,lw=lw)
		fig = plt.gcf()
		ax = plt.gca()
		ax.add_artist(sn_window)

	def make_display_stochastic_flat(self):
		if self.config is not 'flat' and self.config is not 'flat_stochastic':
			raise Exception('Cannot make display, configuration is not flat')

		spatLUT_inv, fresnel, tres = self.spat_fresnel_flat()	

		j, i = np.meshgrid(np.arange(self.dispres),np.arange(self.dispres))

		j = j.astype('float64')
		i = i.astype('float64')

		#Shift all x and y coordinates by random float between -stoch_range//2 and stoch_range//2
		j += np.random.rand(self.dispres,self.dispres)*self.stoch_range - self.stoch_range//2
		i += np.random.rand(self.dispres,self.dispres)*self.stoch_range - self.stoch_range//2

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

	def transform_images_loop(self, im, min_lux,max_lux, smooth=False, downsample=False, lux=False, noupsample=False,gma=2.2):
		# make sure image has no negative values, and also transform to exponential space

		# first, we need to scale im according to min_lux and max_lux
		# normalize to [0, 1]
		im = im.copy()

		# Solve for gamma functions params given low and high steps
		self.find_ab(min_lux,max_lux,gma)

		if not lux:
			im = self.process_im(im)

		# here we also need to upsample the image to match the backend display size
		if not noupsample:
			im = rescale(im,(self.supersample_deg,self.supersample_deg),mode='constant')
			im = im[:-1,:-1]

		im_avg = np.zeros((self.display.shape[0], self.display.shape[1], self.display.shape[3]))

		if downsample and not upsample:
			im_avg = downscale_local_mean(im_avg,(downsample,downsample,1))
		for i in range(self.display.shape[3]):
			sys.stdout.write('start[' + 'X'*(i+1) + ' '*(self.display.shape[3]-i+1) + ']end\r')
			sys.stdout.flush()
			im_avg[:,:,i] = self.transform_image(im,smooth=smooth,downsample=downsample, disp_ind=i)

		return self.degamma(downscale_local_mean(np.mean(im_avg,axis=2),(self.supersample_deg,self.supersample_deg)))

	def transform_image(self, im, smooth=False, downsample=False, disp_ind = 0):
		"""
		Transforms/distorts input image using the spatial lookup table in self.display and Fresnel transmittance in
			self.fresdisplay

		inputs--
			im: 2-D numpy array. Note that image dimensions N x M must equal the dimensions
				of self.fresdisplay and the first two dimensions of self.display
		"""
		weights = im.flatten()* self.fresdisplay[:,:,disp_ind].ravel()

		im_snell = np.histogram2d(np.round(self.display[:,:,0,disp_ind]).astype('int').ravel(),
							  np.round(self.display[:,:,1,disp_ind]).astype('int').ravel(),
							  np.arange(0,self.display.shape[0]+1),
							  weights = weights)

		im_snell = im_snell[0].T

		if smooth:
			im_snell =  self.degamma(ndi.filters.gaussian_filter(im_snell,5))
		if downsample:
			im_snell =  self.degamma(downscale_local_mean(im_snell,(downsample,downsample)))
		
		return im_snell

	def inverse_transform_image_loop(self,im,min_lux,max_lux,idealized=False,gma=2.2):

		# first, we need to scale im according to min_lux and max_lux
		# normalzie to [0, 1]
		ref_input = im.copy()
		
		self.find_ab(min_lux,max_lux, gma)

		im = self.gamma(im)

		im = rescale(im,(self.supersample_deg,self.supersample_deg),mode='constant')
		im = im[:-1,:-1]

		#return self.inverse_transform_image_fast(im, disp_ind=0)


		im_avg = np.zeros((self.display.shape[0], self.display.shape[1], self.display.shape[3]))

		print("inverting image...")
		for i in range(self.display.shape[3]):
			print("{}/{} ...".format(i+1,self.display.shape[3]))
			im_avg[:,:,i] = self.inverse_transform_image_fast(im, disp_ind=i)
		print("Done.")

		output = np.mean(im_avg,axis=2)
		#output = self.degamma(downscale_local_mean(output,(self.supersample_deg,self.supersample_deg)))
		output = self.degamma(output)
		# print("Calculating reconstruction error...")
		if np.min(output) < 0 and not idealized:
			print("Warning: target image would require pixels darker than what your projector can produce")
			print("Warning: setting the pixels in question to 0")
			output[output<0] = 0

		if np.max(output) > 255 and not idealized:
			print("Warning: target image would require pixels brighter than what your projector can produce")
			print("Warning: setting the pixels in question to 255")
			output[output>255] = 255

		print ("Performing forward transformation...")

		fwd = self.transform_images_loop(output,min_lux=min_lux,max_lux=max_lux, noupsample=True,gma=gma)

		return output, fwd

	def inverse_transform_image_fast(self,im,disp_ind=0):
		inds = np.ravel_multi_index((np.round(self.display[:,:,0,disp_ind].ravel()).astype('int'),
			np.round(self.display[:,:,1,disp_ind].ravel()).astype('int')),
			self.display.shape[:2])

		uniques, inverse_inds, counts = np.unique(inds,return_inverse=True,return_counts=True)

		imshape = im.shape

		# im_ = im.ravel().copy()
		im_norav = im.copy()

		#before I distribute the photoms from im_, i need to distribute pixel values from "holes" to fileld neighbors
		imm = np.zeros_like(im)
		imm[np.unravel_index(uniques, im.shape[:2])] = 1
		labels = label(imm,connectivity=1,background=1)
		# now all non 0 and 1 labels are the holes I want to distribute to neighbors
		labels = labels>1
		were = np.where(labels)
		#The problem is that it is a hard programming problem to find the closest non-zero neighbor
		# So what I will do is just look in an 8-neighborhood for a non-zero. If I find one, distribute, otherwise
		# do nothing...
		labels_inv = labels <=1 #All non-holes
		#print(labels_inv.shape)

		wtrack = np.zeros_like(im_norav)
		#print(wtrack.shape)

		#Calulcate how many non-zero neighbors
		for ii in range(-1,2):
			for jj in range(-1,2):
				if ii != 0 and jj != 0:
					wtrack[were] += labels_inv[(were[0]+ii,were[1]+jj)]

		# Now distribute and divide by # of neighbors
		for ii in range(-1,2):
			for jj in range(-1,2):
				if ii != 0 and jj != 0:
					im_norav[(were[0]+ii,were[1]+jj)] += im_norav[were]/wtrack[were]
					#wtrack[were] += labels_inv[(were[0]+ii,were[1]+jj)]

		im_ = im_norav.ravel().copy()

		im_[uniques] = im_[uniques]/np.round(counts)

		im_out = im_[inds]

		im_out = np.reshape(im_out*(1/self.fresdisplay[:,:,disp_ind].ravel()),imshape).T

		return im_out

	def find_ab(self,low,high,gamma):
		"""
		Solves for parameters in pixel transfer function,
		
		g(x) = a(x+b)^gamma


		"""
		ginv = 1/gamma
		b = (255*low**ginv)/(high**ginv-low**ginv)
		#a= low/b**gamma
		a = high/(255+b)**gamma
		self.a = a
		self.b = b
		self.g = gamma

	def gamma(self, px):
		return self.a*(px+self.b)**self.g

	def degamma(self, px):
		ginv = 1/self.g
		return px**ginv/self.a**ginv - self.b

	def process_im(self,im):
		"""
		Make sure im has no negative values, then transform to exp space
		"""

		# if np.min(im) < 0:
		# 	raise Exception("Detected negative values in input image. All pixels values must be >= 0")
		return self.gamma(im)

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

