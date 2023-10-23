# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist
#=============================================================================
def show_box(box, grains=None):
	"""
	Function to show the box filled with grains.

	Parameters
	----------
	box : numpy.ndarray
		A square (2-dimensional) array. The state of the cells of the array
		can be equal to 0 which mean that there are no grain on this cell. If
		the value is equal to c, c beeing superior to 0, this mean that the
		cell is part of the c grain.
	grains : numpy.ndarray, optional
		A 2-dimensional array that store the list of the grains putted into
		the box. The default is None.

	Returns
	-------
	None.

	"""
	plt.figure(figsize=(12, 12))
	plt.imshow(box, interpolation='Nearest')
	if type(grains) != type(None):
		plt.plot(grains[:, 2], grains[:, 1], '.r')

	plt.show()

def repart_grains(grains, plot=True):
	"""
	Function to calculate (and plot if asked) the distribution of the grains
	put into the box.

	Parameters
	----------
	grains : numpy.ndarray
		A 2-dimensional array that store the list of the grains putted into
		the box.
	plot : bool, optional
		Indicate if we want to plot the distribution. The default is True.

	Returns
	-------
	values : numpy.ndarray
		A 1-dimensional array that store the sorted and unique list of drawed
		grains ray.
	counts : numpy.ndarray
		Number of exemples for each values.

	"""
	values, counts = np.unique(grains[:, 0], return_counts=True)
	if plot:
		plt.figure()
		plt.title('Ray distribution')
		plt.vlines(values, 0, counts, lw=5)
		plt.xlabel('Rays')
		plt.ylabel('Count')
		plt.ylim(0)
		plt.show()

	return values, counts

def show_trying(trying):
	"""
	Function to show the evolution of the sucess/try ratio.

	Parameters
	----------
	trying : numpy.ndarray
		Evolution of the ratio between the sucessful/try.

	Returns
	-------
	None.

	"""
	plt.figure(figsize=(12, 5))
	plt.grid(True, zorder=1)
	plt.plot(trying, zorder=2)
	plt.ylabel('putted/try')
	plt.ylim(-0.01, 1.02)
	plt.xlim(-len(trying)*0.02, len(trying)+len(trying)*0.02)
	plt.show()

def granular_filling(size, ray_range, ratio, method, verbose=True):
	"""
	Function to randomly fill a blank space with granular particules.

	Parameters
	----------
	size : int
		Size of the plate to fill.
	ray_range : list
		Limites rays size.
	ratio : float
		Try over win ratio.
	method : str
		'lrqc' or 'uniform'.
	verbose : bool
		Parameter to indicate the verbose of the code.

	Returns
	-------
	plate : numpy.ndarray
		A 2d array of the filled box. The 1-infinite values indicate the
		pixels tooked by the i-th grain.
	disques : numpy.ndarray
		List of the disques positionned, their ray and their position.
	tent : numpy.ndarray
		List of the evolution of the ratio between trys and succeeds
		particules placed.

	Exemple
	-------
	In[0] : granular_filling(19, [4, 6], 0.001, 'uniform')
	Out[0] : array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
					[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0],
					[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
					]),

			 [0.0, 0.6666666666666666, 0.5, 0.6, 0.6666666666666666],

			 [[4, 11, 5], [4, 4, 9], [4, 13, 14]]

	Note
	----
	The time cunsumption is going exponetialy with the size of the box. With
	ray range [4, 6], it wil take ~ xxx0.00 s to fill a 50*50 cells box
	the ~ 50s to fill a 100*100 cells box.

	"""
	if verbose:
		print('Please wait while filling the tray, this may take a few '+
			  'moments to several tens of minutes...')

	# Create the box to fill
	plate = np.zeros((size, size))
	# Range of the possible rays
	delta_r = ray_range[1]-ray_range[0]
	# Number of trial, start at 1 to avoid 0-division
	t = 1
	# Creating all the possible coordinates
	y_grid, x_grid = np.meshgrid(np.arange(0, size, 1.),
								 np.arange(0, size, 1.))
	grid_coord = np.array([x_grid, y_grid]).T
	# Evolution of the ratio number of succeful trial over the total number of
	# trial
	tent = []
	# Lists of grains and their parameters
	disques = []
	# Parameter to stop the while loop
	stop = False
	# Number of successful trial, start at 1 to avoid 0-division
	c = 1
	# Integer that count the number of time that there aren't any possible
	# place for the drawn grain
	toobig = 0
	if verbose:
		pbar = tqdm(total=1, desc='progression')

	while stop != True:
		# Method to draw rays following an exponentional law where the
		# probability increase with the size of the ray.
		if method == 'lrqc':
			# Ray exentent
			n = ray_range[1]-ray_range[0]+1
			# Range of ray
			x_range = np.arange(ray_range[0]-1, ray_range[1]+1)
			# Degree angle for the approximation of the exonentional law
			degre = np.arccos((x_range-ray_range[0]-1)/n)
			ray_rang_cos = np.cos(degre)*n+ray_range[0]-1
			# Probability density function
			pdf = np.sin(np.pi+degre)*ray_range[1]+ray_range[1]
			ray_rang_cos = ray_rang_cos[1:].astype(int)
			pdf = pdf[1:]
			pdf = pdf/np.sum(pdf)
			# Drawing the rays
			ray = np.random.choice(ray_rang_cos, size=1, replace=False, p=pdf)[0]

		# Method to draw uniform rays
		elif method == 'uniform':
			ray = np.random.randint(ray_range[0], ray_range[1]+1)

		# Looking the position where we can put the grain
		psz = np.argwhere(plate == 0)
		psz = psz[(psz[:, 0] >= ray)&(psz[:, 1] >= ray)&
				  (psz[:, 0] < size-ray)&(psz[:, 1] < size-ray)]

		# Trick to ease the calculation of the position of the grain
		diag = int((ray**2/2)**.5)
		kernel = np.array([[[0, ray]], [[ray, 0]], [[0, -ray]], [[-ray, 0]],
						   [[diag, diag]], [[-diag, diag]], [[diag, -diag]],
						   [[-diag, -diag]]])

		projec = psz+kernel
		cond1 = plate[projec[0, :, 0], projec[0, :, 1]] == 0
		cond2 = plate[projec[1, :, 0], projec[1, :, 1]] == 0
		cond3 = plate[projec[2, :, 0], projec[2, :, 1]] == 0
		cond4 = plate[projec[3, :, 0], projec[3, :, 1]] == 0
		cond5 = plate[projec[4, :, 0], projec[4, :, 1]] == 0
		cond6 = plate[projec[5, :, 0], projec[5, :, 1]] == 0
		cond7 = plate[projec[6, :, 0], projec[6, :, 1]] == 0
		cond8 = plate[projec[7, :, 0], projec[7, :, 1]] == 0
		psz = psz[cond1&cond2&cond3&cond4&cond5&cond6&cond7&cond8]
		# This mean that there are at least one position where the grain can
		# be put
		if len(psz) > 0:
			# Random position
			x, y = psz[np.random.randint(len(psz))]
			dist = cdist(grid_coord, np.array([[x, y]])) # ((Xx-x)**2+(Yy-y)**2)**.5
			# This mean that the random position is available
			if np.sum(plate[dist <= ray]) == 0:
				plate[dist <= ray] = c
				disques.append([ray, x, y])
				# +1 for the succesfull trial
				c += 1

			# +1 for the trial
			t += 1
			tent.append(c/t)
			# An early stopping
			if tent[-1] <= ratio:
				stop = True
				break

		# If there aren't any possible position for the grain due to its size
		else:
			toobig += 1
			ray_range[1] = ray_range[1]-1
			# an early stoping for when there aren't any possible position for
			# any grain size
			if toobig > delta_r:
				stop = True
				break

		if verbose:
			pbar.update(1)

	if verbose:
		pbar.close()

	plate = plate.astype(int)
	disques = np.array(disques)
	tent = np.array(tent)
	if verbose:
		print('The table is filled to '+str(len(plate[plate > 0])/size**2*100)+
		  '%. There is/are '+str(len(disques))+' grains.')

	return plate, disques, tent

def dictio_range_ray(size, ray_range):
	"""
	Function to create a dictionnarie of the relative positions of the bordure
	point of a dissk of ray n.

	Parameters
	----------
	size : int
		Size of the plate to fill.
	ray_range : list
		Upper and lower limits rays size.

	Returns
	-------
	disp_ray : dict
		Dictionaries that countain the relative positions of the bordure point
		of a dissk of ray n.

	Exemple
	-------
	In[0] : dictio_range_ray(19, [4, 6])
	Out[0] : {'8': array([[ 0,  9], [ 1,  5], [ 1,  6], [ 1,  7], [ 1,  8],
						  [ 1, 10], [ 1, 11], [ 1, 12], [ 1, 13], [ 2,  4],
						  [ 2, 14], [ 3,  3], [ 3, 15], [ 4,  2], [ 4, 16],
						  [ 5,  1], [ 5, 17], [ 6,  1], [ 6, 17], [ 7,  1],
						  [ 7, 17], [ 8,  1], [ 8, 17], [ 9,  0], [ 9, 18],
						  [10,  1], [10, 17], [11,  1], [11, 17], [12,  1],
						  [12, 17], [13,  1], [13, 17], [14,  2], [14, 16],
						  [15,  3], [15, 15], [16,  4], [16, 14], [17,  5],
						  [17,  6], [17,  7], [17,  8], [17, 10], [17, 11],
						  [17, 12], [17, 13], [18,  9]], dtype=int64),
			  '9': array([[-1,  9], [ 0,  5], [ 0,  6], [ 0,  7], [ 0,  8],
						  [ 0, 10], [ 0, 11], [ 0, 12], [ 0, 13], [ 1,  3],
						  [ 1,  4], [ 1, 14], [ 1, 15], [ 2,  2], [ 2, 16],
						  [ 3,  1], [ 3, 17], [ 4,  1], [ 4, 17], [ 5,  0],
						  [ 5, 18], [ 6,  0], [ 6, 18], [ 7,  0], [ 7, 18],
						  [ 8,  0], [ 8, 18], [ 9, -1], [ 9, 19], [10,  0],
						  [10, 18], [11,  0], [11, 18], [12,  0], [12, 18],
						  [13,  0], [13, 18], [14,  1], [14, 17], [15,  1],
						  [15, 17], [16,  2], [16, 16], [17,  3], [17,  4],
						  [17, 14], [17, 15], [18,  5], [18,  6], [18,  7],
						  [18,  8], [18, 10], [18, 11], [18, 12], [18, 13],
						  [19,  9]], dtype=int64),
			 '10': array([[-2,  9], [-1,  5], [-1,  6], [-1,  7], [-1,  8],
						  [-1, 10], [-1, 11], [-1, 12], [-1, 13], [ 0,  3],
						  [ 0,  4], [ 0, 14], [ 0, 15], [ 1,  2], [ 1, 16],
						  [ 2,  1], [ 2, 17], [ 3,  0], [ 3, 18], [ 4,  0],
						  [ 4, 18], [ 5, -1], [ 5, 19], [ 6, -1], [ 6, 19],
						  [ 7, -1], [ 7, 19], [ 8, -1], [ 8, 19], [ 9, -2],
						  [ 9, 20], [10, -1], [10, 19], [11, -1], [11, 19],
						  [12, -1], [12, 19], [13, -1], [13, 19], [14,  0],
						  [14, 18], [15,  0], [15, 18], [16,  1], [16, 17],
						  [17,  2], [17, 16], [18,  3], [18,  4], [18, 14],
						  [18, 15], [19,  5], [19,  6], [19,  7], [19,  8],
						  [19, 10], [19, 11], [19, 12], [19, 13], [20,  9]],
						  dtype=int64),
			 '11': array([[-3,  9], [-2,  5], [-2,  6], [-2,  7], [-2,  8],
						  [-2, 10], [-2, 11], [-2, 12], [-2, 13], [-1,  3],
						  [-1,  4], [-1, 14], [-1, 15], [ 0,  2], [ 0, 16],
						  [ 1,  1], [ 1, 17], [ 2,  0], [ 2, 18], [ 3, -1],
						  [ 3, 19], [ 4, -1], [ 4, 19], [ 5, -2], [ 5, 20],
						  [ 6, -2], [ 6, 20], [ 7, -2], [ 7, 20], [ 8, -2],
						  [ 8, 20], [ 9, -3], [ 9, 21], [10, -2], [10, 20],
						  [11, -2], [11, 20], [12, -2], [12, 20], [13, -2],
						  [13, 20], [14, -1], [14, 19], [15, -1], [15, 19],
						  [16,  0], [16, 18], [17,  1], [17, 17], [18,  2],
						  [18, 16], [19,  3], [19,  4], [19, 14], [19, 15],
						  [20,  5], [20,  6], [20,  7], [20,  8], [20, 10],
						  [20, 11], [20, 12], [20, 13], [21,  9]],
						  dtype=int64),
			 '12': array([[-4,  9], [-3,  4], [-3,  5], [-3,  6], [-3,  7],
						  [-3,  8], [-3, 10], [-3, 11], [-3, 12], [-3, 13],
						  [-3, 14], [-2,  3], [-2, 15], [-1,  1], [-1,  2],
						  [-1, 16], [-1, 17], [ 0,  0], [ 0, 18], [ 1, -1],
						  [ 1, 19], [ 2, -1], [ 2, 19], [ 3, -2], [ 3, 20],
						  [ 4, -3], [ 4, 21], [ 5, -3], [ 5, 21], [ 6, -3],
						  [ 6, 21], [ 7, -3], [ 7, 21], [ 8, -3], [ 8, 21],
						  [ 9, -4], [ 9, 22], [10, -3], [10, 21], [11, -3],
						  [11, 21], [12, -3], [12, 21], [13, -3], [13, 21],
						  [14, -3], [14, 21], [15, -2], [15, 20], [16, -1],
						  [16, 19], [17, -1], [17, 19], [18,  0], [18, 18],
						  [19,  1], [19,  2], [19, 16], [19, 17], [20,  3],
						  [20, 15], [21,  4], [21,  5], [21,  6], [21,  7],
						  [21,  8], [21, 10], [21, 11], [21, 12], [21, 13],
						  [21, 14], [22,  9]], dtype=int64)}

	"""
	y_range, x_range = np.meshgrid(np.arange(0, 6*ray_range[1]),
								   np.arange(0, 6*ray_range[1]))

	disp_ray = {}
	center = ((x_range-3*ray_range[1])**2+(y_range-3*ray_range[1])**2)**.5
	kernel = np.array([[[0, 1]], [[1, 0]], [[0, -1]], [[-1, 0]]])
	for i in range(ray_range[0]*2, 2*ray_range[1]+1):
		mask = (center <= i+1)
		pf = np.argwhere(Mask == True)
		pk = pf+kernel
		c1 = mask[pk[0, :, 0], pk[0, :, 1]] == False
		c2 = mask[pk[1, :, 0], pk[1, :, 1]] == False
		c3 = mask[pk[2, :, 0], pk[2, :, 1]] == False
		c4 = mask[pk[3, :, 0], pk[3, :, 1]] == False
		disp_ray[str(i)] = pf[c1|c2|c3|c4]-3*ray_range[1]

	return disp_ray

def compac_granular(size, ray_range, verbose=True):
	"""
	A more compact method to fill a box with circular particules of random
	rays.

	Parameters
	----------
	size : int
		Taille du plateau à remplire.
	ray_range : list
		Limites de taille des rayons possible.

	Returns
	-------
	table : numpy.ndarray
		A square (2-dimensional) array. The state of the cells of the array
		can be equal to 0 which mean that there are no grain on this cell. If
		the value is equal to c, c beeing superior to 0, this mean that the
		cell is part of the c grain.
	disques : numpy.ndarray
		List of the disques positionned, their ray and their position.
	Verbose : bool, optional
		If you want that the algorithm be verbose. The default is True.

	Exemple
	-------
	In[0] : compac_granular(19, [4, 6])
	Out[0] : array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 0, 0],
					[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 0],
					[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3],
					[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 0],
					[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0],
					[0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 0],
					[0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
					[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0],
					[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 4, 4, 4, 4, 4, 0, 0],
					[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 4, 4, 4, 4, 4, 4, 4, 0],
					[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 0],
					[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4],
					[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 4, 4, 4, 4, 4, 4, 4, 0],
					[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 4, 4, 4, 4, 4, 4, 4, 0],
					[0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0],
					[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0]]),
			 array([[ 4,  4,  6],
					[ 5, 13,  5],
					[ 4,  5, 14],
					[ 4, 14, 14]], dtype=int64))

	Note
	----
	The computation time goes exponentially with the size of the table. You
	should start with small model (~100) the increase the size until you are
	satified.

	"""
	if verbose:
		print('Please wait while filling the tray, this may take a few '+
			  'moments to several tens of minutes...')

	disques = np.array([[]], dtype=int)
	stop = False
	table = np.zeros((size, size))
	range_ = ray_range[1]-ray_range[0]
	# To ease the caluclations
	y_grid, x_grid = np.meshgrid(np.arange(0, size, 1.), np.arange(0, size, 1.))
	grid_coord = np.array([x_grid, y_grid]).T
	disp_ray = dictio_range_ray(size, ray_range)
	rand = np.random.randint(ray_range[0], ray_range[1]+1)
	x, y = np.random.randint(rand, size-rand, 2)
	disques = np.concatenate((disques, np.array([[rand, x, y]])), axis=1)
	dist = cdist(grid_coord, np.array([[x, y]])) # ((XX-x)**2 +(YY-y)**2)**.5
	table[dist <= rand] = 1
	c = 2
	toobig = 0
	if verbose:
		pbar = tqdm(total=1, desc='progression')

	while stop != True:
		plug = []
		rand = np.random.randint(ray_range[0], ray_range[1]+1-toobig)
		# Get the possible places for all of the discs with the drawed ray
		for i in range(len(disques)):
			plus = np.copy(disp_ray[str(disques[i, 0]+rand)])
			plus += disques[i, 1:]
			plus = plus[(plus[:, 0] > rand)&(plus[:, 1] > rand)&
						(plus[:, 0] < size-rand)&(plus[:, 1] < size-rand)]

			plus = plus[table[plus[:, 0], plus[:, 1]] == 0]
			pf = np.argwhere(table > 0)
			#dist = ((pf[:, 0]-plus[:, 0, np.newaxis])**2 +
			#		(pf[:, 1]-plus[:, 1, np.newaxis])**2)**.5
			# cdist is higly more fast than self numpy implementation
			dist = cdist(pf, plus)
			plus = plus[np.sum(dist > rand, axis=1) == pf.shape[0]]
			plug.append(plus)

		plug = np.concatenate(plug)
		if len(plug) > 0:
			# To put the new disc the closest of the other discs
			xmean, ymean = np.mean(disques[:, 1]), np.mean(disques[:, 2])
			dist = ((plug[:, 0]-xmean)**2 +(plug[:, 1]-ymean)**2)**.5
			x, y = plug[np.argmin(dist)]
			disques = np.concatenate((disques, np.array([[rand, x, y]])),
									 axis=0)

			dist = cdist(grid_coord, np.array([[x, y]])) # ((XX-x)**2 +(YY-y)**2)**.5
			table[dist <= rand] = c
			c += 1

		else:
			toobig += 1
			if toobig > range_:
				stop = True

		if verbose:
			pbar.update(1)

	if verbose:
		pbar.close()

	table = table.astype(int)
	if verbose:
		print('The table is filled to '+str(len(table[table > 0])/size**2*100)+
		  '%. There is/are '+str(len(disques))+' grains.')

	return table, disques
