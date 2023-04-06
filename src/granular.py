# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:11:42 2023

@author: Matthieu Nougaret
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def ShowBox(box, grains=None):
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

def RepartGrains(grains, Plot=True):
	"""
	Function to calculate (and plot if asked) the distribution of the grains
	put into the box.

	Parameters
	----------
	grains : numpy.ndarray
		A 2-dimensional array that store the list of the grains putted into
		the box.
	Plot : bool, optional
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
	if Plot:
		plt.figure()
		plt.title('Ray distribution')
		plt.vlines(values, 0, counts, lw=5)
		plt.xlabel('Rays')
		plt.ylabel('Count')
		plt.ylim(0)
		plt.show()
	return values, counts

def ShowTrying(trying):
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

def GranularFilling(Size, RayRange, ratio, method, verbose=True):
	"""
	Function to randomly fill a blank space with granular particules.

	Parameters
	----------
	Size : int
		Size of the plate to fill.
	RayRange : list
		Limites rays size.
	ratio : float
		Try over win ratio.
	method : str
		'lrqc' or 'uniform'.
	verbose : bool
		Parameter to indicate the verbose of the code.

	Returns
	-------
	Plate : numpy.ndarray
		A 2d array of the filled box. The 1-infinite values indicate the
		pixels tooked by the i-th grain.
	disques : numpy.ndarray
		List of the disques positionned, their ray and their position.
	tent : numpy.ndarray
		List of the evolution of the ratio between trys and succeeds
		particules placed.

	Exemple
	-------
	In[0] : GranularFilling(19, [4, 6], 0.001, 'uniform')
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
	ray range [4, 6], it wil take ~ 0.00 s to fill a 50*50 cells box
	the ~ 50s to fill a 100*100 cells box.

	"""
	if verbose:
		print('Please wait while filling the tray, this may take a few '+
			  'moments to several tens of minutes...')
	# Create the box to fill
	Plate = np.zeros((Size, Size))
	# Range of the possible rays
	Dr = RayRange[1]-RayRange[0]
	# Number of trial, start at 1 to avoid 0-division
	t = 1
	# Creating all the possible coordinates
	Yy, Xx = np.meshgrid(np.arange(0, Size, 1.),
						 np.arange(0, Size, 1.))
	# Evolution of the ratio number of succeful trial over the total number of
	# trial
	tent = []
	# Lists of grains and their parameters
	disques = []
	# Parameter to stop the while loop
	Stop = False
	# Number of successful trial, start at 1 to avoid 0-division
	c = 1
	# Integer that count the number of time that there aren't any possible
	# place for the drawn grain
	toobig = 0
	if verbose:
		pbar = tqdm(total=1, desc='progression')

	while Stop != True:
		# Method to draw rays following an exponentional law where the
		# probability increase with the size of the ray.
		if method == 'lrqc':
			# Ray exentent
			n = RayRange[1]-RayRange[0]+1
			# Range of ray
			Xs = np.arange(RayRange[0]-1, RayRange[1]+1)
			# Degree angle for the approximation of the exonentional law
			degre = np.arccos((Xs-RayRange[0]-1)/n)
			rayRang = np.cos(degre)*n+RayRange[0]-1
			# Probability density function
			pdf = np.sin(np.pi+degre)*RayRange[1]+RayRange[1]
			rayRang = rayRang[1:].astype(int)
			pdf = pdf[1:]
			pdf = pdf/np.sum(pdf)
			# Drawing the rays
			Ray = np.random.choice(rayRang, size=1, replace=False, p=pdf)[0]

		# Method to draw uniform rays
		elif method == 'uniform':
			Ray = np.random.randint(RayRange[0], RayRange[1]+1)

		# Looking the position where we can put the grain
		Pz = np.argwhere(Plate == 0)
		Pz = Pz[(Pz[:, 0] >= Ray)&(Pz[:, 1] >= Ray)&
				(Pz[:, 0] < Size-Ray)&(Pz[:, 1] < Size-Ray)]
		# Trick to ease the calculation of the position of the grain
		diag = int((Ray**2/2)**.5)
		kernel = np.array([[[0, Ray]], [[Ray, 0]], [[0, -Ray]], [[-Ray, 0]],
						   [[diag, diag]], [[-diag, diag]], [[diag, -diag]],
						   [[-diag, -diag]]])
		Projec = Pz+kernel
		cond1 = Plate[Projec[0, :, 0], Projec[0, :, 1]] == 0
		cond2 = Plate[Projec[1, :, 0], Projec[1, :, 1]] == 0
		cond3 = Plate[Projec[2, :, 0], Projec[2, :, 1]] == 0
		cond4 = Plate[Projec[3, :, 0], Projec[3, :, 1]] == 0
		cond5 = Plate[Projec[4, :, 0], Projec[4, :, 1]] == 0
		cond6 = Plate[Projec[5, :, 0], Projec[5, :, 1]] == 0
		cond7 = Plate[Projec[6, :, 0], Projec[6, :, 1]] == 0
		cond8 = Plate[Projec[7, :, 0], Projec[7, :, 1]] == 0
		Pz = Pz[cond1&cond2&cond3&cond4&cond5&cond6&cond7&cond8]
		# This mean that there are at least one position where the grain can
		# be put
		if len(Pz) > 0:
			# Random position
			x, y = Pz[np.random.randint(len(Pz))]
			Dist = ((Xx-x)**2+(Yy-y)**2)**.5
			# This mean that the random position is available
			if np.sum(Plate[Dist <= Ray]) == 0:
				Plate[Dist <= Ray] = c
				disques.append([Ray, x, y])
				# +1 for the succesfull trial
				c += 1

			# +1 for the trial
			t += 1
			tent.append(c/t)
			# An early stopping
			if tent[-1] <= ratio:
				Stop = True
				break

		# If there aren't any possible position for the grain due to its size
		else:
			toobig += 1
			RayRange[1] = RayRange[1]-1
			# an early stoping for when there aren't any possible position for
			# any grain size
			if toobig > Dr:
				Stop = True
				break

		if verbose:
			pbar.update(1)

	if verbose:
		pbar.close()

	Plate = Plate.astype(int)
	disques = np.array(disques)
	tent = np.array(tent)
	if verbose:
		print('The table is filled to '+str(len(Plate[Plate > 0])/Size**2*100)+
		  '%. There is/are '+str(len(Disques))+' grains.')

	return Plate, disques, tent

def DictioRangeRay(Size, RayRange):
	"""
	Function to create a dictionnarie of the relative positions of the bordure
	point of a dissk of ray n.

	Parameters
	----------
	Size : int
		Size of the plate to fill.
	RayRange : list
		Upper and lower limits rays size.

	Returns
	-------
	DispRay : dict
		Dictionaries that countain the relative positions of the bordure point
		of a dissk of ray n.

	Exemple
	-------
	In[0] : DictioRangeRay(19, [4, 6])
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
	y_range, x_range = np.meshgrid(np.arange(0, 6*RayRange[1]),
								   np.arange(0, 6*RayRange[1]))
	DispRay = {}
	CentD = ((x_range-3*RayRange[1])**2+(y_range-3*RayRange[1])**2)**.5
	kernel = np.array([[[0, 1]], [[1, 0]], [[0, -1]], [[-1, 0]]])
	for i in range(RayRange[0]*2, 2*RayRange[1]+1):
		Mask = (CentD <= i+1)
		pf = np.argwhere(Mask == True)
		pk = pf+kernel
		c1 = Mask[pk[0, :, 0], pk[0, :, 1]] == False
		c2 = Mask[pk[1, :, 0], pk[1, :, 1]] == False
		c3 = Mask[pk[2, :, 0], pk[2, :, 1]] == False
		c4 = Mask[pk[3, :, 0], pk[3, :, 1]] == False
		DispRay[str(i)] = pf[c1|c2|c3|c4]-3*RayRange[1]
	return DispRay

def CompacGranular(Size, RayRange, verbose=True):
	"""
	A more compact method to fill a box with circular particules of random
	rays.

	Parameters
	----------
	Size : int
		Taille du plateau à remplire.
	RayRange : list
		Limites de taille des rayons possible.

	Returns
	-------
	Table : numpy.ndarray
		A square (2-dimensional) array. The state of the cells of the array
		can be equal to 0 which mean that there are no grain on this cell. If
		the value is equal to c, c beeing superior to 0, this mean that the
		cell is part of the c grain.
	Disques : numpy.ndarray
		List of the disques positionned, their ray and their position.
	Verbose : bool, optional
		If you want that the algorithm be verbose. The default is True.

	Exemple
	-------
	In[0] : CompacGranular(19, [4, 6])
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
	The computation time goes exponentially with the size of the Table. You
	should start with small model (~100) the increase the size until you are
	satified.

	"""
	if verbose:
		print('Please wait while filling the tray, this may take a few '+
			  'moments to several tens of minutes...')
	Disques = np.array([[]], dtype=int)
	Stop = False
	Table = np.zeros((Size, Size))
	Range = RayRange[1]-RayRange[0]
	# To ease the caluclations
	YY, XX = np.meshgrid(np.arange(0, Size, 1.), np.arange(0, Size, 1.))
	DispRay = DictioRangeRay(Size, RayRange)
	rand = np.random.randint(RayRange[0], RayRange[1]+1)
	x, y = np.random.randint(rand, Size-rand, 2)
	Disques = np.concatenate((Disques, np.array([[rand, x, y]])), axis=1)
	Dist = ((XX-x)**2 +(YY-y)**2)**.5
	Table[Dist <= rand] = 1
	c = 2
	toobig = 0
	if verbose:
		pbar = tqdm(total=1, desc='progression')
	while Stop != True:
		Plug = []
		rand = np.random.randint(RayRange[0], RayRange[1]+1-toobig)
		# Get the possible places for all of the discs with the drawed ray
		for i in range(len(Disques)):
			Plus = np.copy(DispRay[str(Disques[i, 0]+rand)])
			Plus += Disques[i, 1:]
			Plus = Plus[(Plus[:, 0] > rand)&(Plus[:, 1] > rand)&
						(Plus[:, 0] < Size-rand)&(Plus[:, 1] < Size-rand)]
			Plus = Plus[Table[Plus[:, 0], Plus[:, 1]] == 0]
			pf = np.argwhere(Table > 0)
			Dist = ((pf[:, 0]-Plus[:, 0, np.newaxis])**2 +
					(pf[:, 1]-Plus[:, 1, np.newaxis])**2)**.5
			Plus = Plus[np.sum(Dist > rand, axis=1) == pf.shape[0]]
			Plug.append(Plus)

		Plug = np.concatenate(Plug)
		if len(Plug) > 0:
			# To put the new disc the closest of the other discs
			xmean, ymean = np.mean(Disques[:, 1]), np.mean(Disques[:, 2])
			Dist = ((Plug[:, 0]-xmean)**2 +(Plug[:, 1]-ymean)**2)**.5
			x, y = Plug[np.argmin(Dist)]
			Disques = np.concatenate((Disques, np.array([[rand, x, y]])),
									 axis=0)
			Dist = ((XX-x)**2 +(YY-y)**2)**.5
			Table[Dist <= rand] = c
			c += 1

		else:
			toobig += 1
			if toobig > Range:
				Stop = True

		if verbose:
			pbar.update(1)

	if verbose:
		pbar.close()

	Table = Table.astype(int)
	if verbose:
		print('The table is filled to '+str(len(Table[Table > 0])/Size**2*100)+
		  '%. There is/are '+str(len(Disques))+' grains.')

	return Table, Disques
