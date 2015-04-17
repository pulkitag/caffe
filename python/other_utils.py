## @package other_utils
#  Miscellaneous Util Functions
#
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt
import copy
import os
##
# Verify if all the keys are present recursively in the dict
def verify_recursive_key(data, keyNames):
	'''
		data    : dict like data['a']['b']['c']...['l']
		keyNames: list of keys 
	'''
	assert isinstance(keyNames, list), 'keyNames is required to be a list'
	assert data.has_key(keyNames[0]), '%s not present' % keyNames[0]
	for i in range(1,len(keyNames)):
		dat = reduce(lambda dat, key: dat[key], keyNames[0:i], data)
		assert isinstance(dat, dict), 'Wrong Keys'
		assert dat.has_key(keyNames[i]), '%s key not present' % keyNames[i]
	return True

##
# Set the value of a recursive key. 
def set_recursive_key(data, keyNames, val):
	if verify_recursive_key(data, keyNames):
		dat = reduce(lambda dat, key: dat[key], keyNames[:-1], data)
		dat[keyNames[-1]] = val
	else:
		raise Exception('Keys not present')

##
# Get the item from a recursive key
def get_item_recursive_key(data, keyNames):
	if verify_recursive_key(data, keyNames):
		dat = reduce(lambda dat, key: dat[key], keyNames[:-1], data)
		return dat[keyNames[-1]]
	else:
		print "Not found:", keyNames
		return None

##
# Find the path to the key in a recursive dictionary. 
def find_path_key(data, keyName):
	'''
		Returns path to the first key of name keyName that is found.
		if keyName is a list - [k1, k2 ..kp] then find in data[k1][k2]...[kp-1] the key kp
	'''
	path    = []
	prevKey = []
	if not isinstance(data, dict):
		return path
	#Find if all keys except the last one exist or not. 
	if isinstance(keyName, list):
		data = copy.deepcopy(data)
		for key in keyName[0:-1]:
			if key not in data:
				return []
			else:
				data = data[key]
		prevKey = keyName[0:-1]
		keyName = keyName[-1]

	if data.has_key(keyName):
		return [keyName]
	else:
		for key in data.keys():
			pathFound = find_path_key(data[key], keyName)
			if len(pathFound) > 0:
				return prevKey + [key] + pathFound
	return path

##
# Find an item in dict
# keyName should be a string or an list of a single name. 
def get_item_dict(data, keyName):
	keyPath = find_path_key(data, keyName)
	if len(keyPath)==0:
		return None
	else:
		return get_item_recursive_key(data, keyPath)

##
# Read the image
def read_image(imName, color=True, isBGR=False, imSz=None):
	'''
		color: True - if a gray scale image is encountered convert into color
	'''
	im = plt.imread(imName)
	if color:
		if im.ndim==2:
			print "Converting grayscale image into color image"
			im = np.tile(im.reshape(im.shape[0], im.shape[1],1),(1,1,3))
		if isBGR:
			im = im[:,:,[2,1,0]]
	#Resize if needed
	if imSz is not None:
		assert isinstance(imSz,int)
		im = scm.imresize(im, (imSz,imSz))
	return im			


##
# Crop the image
def crop_im(im, bbox, **kwargs):
	'''
		The bounding box is assumed to be in the form (xmin, ymin, xmax, ymax)
		kwargs:
			imSz: Size of the image required
	'''
	cropType = kwargs['cropType']
	imSz  = kwargs['imSz']
	x1,y1,x2,y2 = bbox
	x1 = max(0, x1)
	y1 = max(0, y1)
	x2 = min(im.shape[1], x2)
	y2 = min(im.shape[0], y2)
	if cropType=='resize':
		imBox = im[y1:y2, x1:x2]
		imBox = scm.imresize(imBox, (imSz, imSz))
	if cropType=='contPad':
		contPad = kwargs['contPad']
		x1 = max(0, x1 - contPad)
		y1 = max(0, y1 - contPad)
		x2 = min(im.shape[1], x2 + contPad)
		y2 = min(im.shape[0], y2 + contPad)	
		imBox = im[y1:y2, x1:x2]
		imBox = scm.imresize(imBox, (imSz, imSz))
	else:
		raise Exception('Unrecognized crop type')
	return imBox		

##
# Read and crop the image. 
def read_crop_im(imName, bbox, **kwargs):
	if kwargs.has_key('color'):
		im = read_image(imName, color=kwargs['color'])
	else:
		im = read_image(imName)
	return crop_im(im, bbox, **kwargs)	


##
# Makes a table from dict
def make_table(**kwargs):
	'''
		kwargs should contains keys and lists as the values.
		Each dictionaty will be plotted as a column.
	'''
	for i,key in enumerate(kwargs.keys()):
		if i==0:
			L = len(kwargs[key])
		else:
			assert L == len(kwargs[key]), 'Wrong length for %s' % key

	N = len(kwargs.keys())
	formatStr = "{:<15} "
	lines = []
	lines.append(''.join(formatStr.format(k) for k in kwargs.keys()) + '\n')
	lines.append('-' * 15 * N + '\n')

	for i in range(L):
		line = ''
		for key in kwargs.keys():
			if isinstance(kwargs[key][i], int):
				fStr = '%d'
			elif isinstance(kwargs[key][i], np.float32) or isinstance(kwargs[key][i], np.float64):
				fStr = '%.3f'
			elif isinstance(kwargs[key][i], str):
				fStr = '%s'
			else:
				fStr = '%s'
			line = line + formatStr.format(fStr % kwargs[key][i])
		line = line + '\n'
		lines.append(line)	
	
	for l in lines:
		print l


#I will make the rows. 
def make_table_rows(**kwargs):
	#Find the maximum length of the key. 
	maxKeyLen = 0
	for key,val in kwargs.iteritems():
		maxKeyLen = max(maxKeyLen, len(key))
	keyLen = maxKeyLen + 15
	keyStr = "{:<%d} " % keyLen
	formatStr = "{:<15} "
	#Lets start printing
	lines = []
	count = 0	
	for key,val in kwargs.iteritems():
		line = ''
		line = line + keyStr.format('%s' % key)
		for v in val:
			if isinstance(v, int):
				fStr = '%d'
			elif isinstance(v, np.float32) or isinstance(v, np.float64):
				fStr = '%.3f'
			elif isinstance(v, str):
				fStr = '%s'
			else:
				fStr = '%s'
			line = line + formatStr.format(fStr % v)
		line = line + '\n'
		lines.append(line)
		if count == 0:
			lines.append('-' * 100 + '\n')
			count += 1
			
	for l in lines:
		print l


##
# Count the things.
def count_unique(arr, maxVal=None):
	if maxVal is None:
		elms = np.unique(arr)
	else:
		elms = range(maxVal+1)
	count = np.zeros((len(elms),))
	for i,e in enumerate(elms):
		count[i] = np.sum(arr==e)

	return count
	 

##
# Create dir
def create_dir(dirName):
	if not os.path.exists(dirName):
		os.makedirs(dirName)
