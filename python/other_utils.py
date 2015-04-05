## @package other_utils
#  Miscellaneous Util Functions
#
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as plt

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
		raise Exception('Keys not present')

##
# Find the path to the key in a recursive dictionary. 
def find_path_key(data, keyName):
	path = []
	if not isinstance(data, dict):
		return path
	if data.has_key(keyName):
		return [keyName]
	else:
		for keys in data.keys():
			path = path + find_path_key(data[key], keyName)
	return path

##
# Read the image
def read_image(imName, color=True, isBGR=False):
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


