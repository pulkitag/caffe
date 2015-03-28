##
#
# The way module operates
# First convert the raw annotation to processed annotations
#		- discard files which donot have euler angle annotations. 
#   - run: save_processed_annotations()
#		- All further processign uses this data. All bounding boxes are listed as seperate images. 
#
# Define an experiment using get_exp_prms
#   - Save the annotation data appropriate for the experiment
#		- run: save_exp_annotations()
#   - If no processing is required then the previously computed processed_annotations will be used. 
#
# Visualize the processed, experiment data
# 
# Save the LMDBs
#
# Create Caffe experiment files. 

import numpy as np
import scipy.io as sio
import h5py as h5
import os
import pdb
import scipy.misc as scm
import shutil
import matplotlib.pyplot as plt
import my_pycaffe_io as mpio
import time

CLASS_NAMES = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus',
							'car', 'chair', 'diningtable', 'motorbike', 'sofa',
							'train', 'tvmonitor']
##
# Get the basic paths
def get_basic_paths():
	#For saving stuff
	saveDir  = '/data1/pulkitag/pascal3d/'
	data0Dir = '/data0/pulkitag/pascal3d/' 
	
	paths            = {}
	#Data
	paths['data']    = '/data1/pulkitag/data_sets/pascal_3d/PASCAL3D+_release1.1/'
	paths['myData']  = '/data1/pulkitag/data_sets/pascal_3d/my/'
	paths['pascalDir'] = os.path.join(paths['data'], 'PASCAL','VOCdevkit','VOC2012')
	paths['pascalSet'] = os.path.join(paths['pascalDir'], 'ImageSets', 'Main', '%s_%s.txt') 
	paths['imDir']     = os.path.join(paths['data'], 'Images')
	paths['rawAnnData'] = os.path.join(paths['data'], 'Annotationsv73', '%s_%s') #Fill with class_dataset
	paths['procAnnData']    = os.path.join(paths['myData'], 'Annotation', '%s_%s.hdf5') #class_setname
	#Snapshots
	paths['snapDir']   = os.path.join(saveDir, 'snapshots')
	#Lmdb Stuff
	paths['lmdbDir']   = os.path.join(data0Dir, 'lmdb-store')
	#Visualization data
	paths['visDir']    = os.path.join(saveDir, 'vis')
	return paths


##
# Experiment parameters
def get_exp_prms(imSz=128, lblType='diff', cropType='resize', numSamples=20000, mnBbox=50):
	'''
	 imSz      : Size of the image to use
	 lblType   : Type of labels to use
						   'diff' -- difference in euler angles. 
	 cropType  : 'resize'-- just resize the bbox to imSz
	 numSamples:  Number of samples per class
	 mnBbox    :  Minimum size of bbox to consider. 
	'''

	prms  = {}
	prms['paths']      =  paths
	prms['imSz']       =  imSz
	prms['lblType']    =  lblType
	prms['cropType']   =  cropType
	prms['numSamples'] =  numSamples
	prms['mnBbox']     =  mnBbox 
	prms['expName']    = 'pascal3d_imSz%d_lbl%s_crp%s_ns%d_mb%d' % \
												(imSz, lblType, cropType, numSamples, mnBbox)

	paths = get_basic_paths()
	#Save the annotation data for the experiment. 
	if mnBbox > 0:
		paths['annData']    = os.path.join(paths['myData'], 'Annotation', '%s_%s_mb%d.hdf5' % mnBbox)
	else:
		paths['annData']    = os.path.join(paths['myData'], 'Annotation', '%s_%s.hdf5') #class_setname
		#LMDB Paths
	paths['lmbd-im'] = '%s_images_pascal3d_imSz%d_crp%s_ns%d_mb%d-lmdb' %\
											 ('%s', imSz, cropType, numSamples, mnBbox)
	paths['lmdb-lb'] = '%s_labels_pascal3d_lbl%s_ns%d-lmdb' % ('%s',lblType, numSamples)

	prms['paths'] = paths
	return prms


##
# Helper function for read_file
# Reads the fields
def read_field_(data, key):
	arr = None
	if key in ['viewpoint']:
		if isinstance(data, h5._hl.dataset.Dataset):
			#print "Empty data instance encountered"
			return arr
		arr = np.zeros((3,))
		if 'azimuth' in data.keys():
			arr[0], arr[1], arr[2] = data['azimuth'][:].squeeze(), \
															 data['elevation'][:].squeeze(), data['distance'][:].squeeze()
		else:
			arr[0], arr[1] = data['azimuth_coarse'][:].squeeze(), data['elevation_coarse'][:].squeeze()
			if 'distance' in data.keys():
				arr[2]  = data['distance'][:].squeeze()
			else:
				arr[2]  = None
	elif key in ['class']:
		arr = ''.join(unichr(c) for c in data[:].squeeze())
	elif key in ['bbox']:	
		arr = data[:].squeeze()
	else:
		raise Exception('Unrecognized field %s' % key)
	return arr


##
# Reads the raw annotation file and finds the relevant data
# Helper function for save_processed_data
def read_raw_annotation_file_(fileName):
	#print fileName
	dat        = h5.File(fileName, 'r')
	vals       = []
	refFlag    = False
	refs       = None
	readFields = ['viewpoint', 'class', 'bbox'] 
	if '#refs#' in dat.keys():
		refs 		= dat['#refs#']
		refFlag = True
	objs   = dat['record']['objects'] 
	if refFlag:
		N = objs['viewpoint'].shape[0]
		for i in range(N):
			valObj = {}
			for rf in readFields:
				valObj[rf] = read_field_(refs[objs[rf][i][0]], rf)
			if valObj['viewpoint'] is not None:
				vals.append(valObj)	
	else:
		valObj = {}
		for rf in readFields:
			valObj[rf] = read_field_(objs[rf], rf)
		if valObj['viewpoint'] is not None:
			vals.append(valObj)
	dat.close()
	return vals


##
# From a list of names - find the image names
# Helper function for save_processed_annotations
def ann2imnames_(annNames):
	paths   = get_basic_paths()
	imNames = []
	for name in annNames:
		bName = os.path.basename(name)
		bName,_ = os.path.splitext(bName)
		dName   = name.split('/')[-2]
		if 'pascal' in dName:
			ext = '.jpg'
		elif 'imagenet' in dName:
			ext = '.JPEG'
		imName  = os.path.join(paths['imDir'], dName, bName + ext)
		imNames.append(imName)
	return imNames

##
#Find the filenames for train and val
def get_raw_ann_files(className, setName='train'):
	assert className in CLASS_NAMES, '% class not found' % className
	paths    = get_basic_paths()	
	fileName = paths['pascalSet'] % (className, setName)
	fid   = open(fileName, 'r')
	lines = fid.readlines()
	fid.close()		

	#Pascal Data	
	pascalAnnDir = paths['rawAnnData'] % (className, 'pascal')
	pascalNames  = []
	for l in lines:
		data = l.split()
		idx = int(l.split()[1])
		if idx == 1:
			pascalNames.append(os.path.join(pascalAnnDir, l.split()[0] + '.mat'))
			assert os.path.exists(pascalNames[-1]), '%s doesnot exist' % pascalNames[-1]

	#Imagenet Data
	imgAnnDir = paths['rawAnnData'] % (className, 'imagenet')
	imgNames  = [os.path.join(imgAnnDir, p) for p in os.listdir(imgAnnDir) if '.mat' in p]

	allNames = pascalNames + imgNames
	return allNames				


##
# Get class statistics of how many images and annotations per class.
def get_class_statistics_raw(setName='train'):
	imCount  = {}
	annCount = {}
	for cl in CLASS_NAMES:
		fileNames = get_raw_ann_files(cl, setName=setName)
		imCount[cl] = len(fileNames) 
		clCount = 0
		for f in fileNames:
			ann = read_file(f)
			clCount += len(ann)
		annCount[cl] = clCount
	return imCount, annCount


##
# Number of boxes in processed data per class
def get_class_statistics_processed(setName='train'):
	paths    = get_basic_paths()
	annCount = {}
	for cl in CLASS_NAMES:
		annFile      = paths['procAnnData'] % (cl, setName)
		fid          = h5.File(annFile,'r')
		annCount[cl] = fid['euler'].shape[0]
		fid.close()
	return annCount


##
# For a specific experiment get number of boxes per class
def get_class_statistics_exp(prms, setName='train'):
	



##
# Process and save the annotations.
def save_processed_annotations(forceSave=False):
	'''
		Reads the raw annotations and saves the processed ones. 
		If the image has multiple bounding boxes then list them as seperate example.
		Fields:
		euler
		distance
		class
		bbox
		imName: Path to the image. 
	'''
	paths    = prms['paths']
	setNames = ['train', 'val']
	dt       = h5.special_dtype(vlen=unicode)
	for s in setNames:
		print "Set: %s" % s
		_,annClassCounts = get_class_statistics_raw(setName=s)
		for cl in CLASS_NAMES:
			print "Processing class: %s" % cl
			outFile  = paths['procAnnData'] % (cl, s)
			if os.path.exists(outFile)
				if forceSave:
					print "Deleting old file: %s" % outFile
					os.remove(outFile)	
				else:
					print 'File: %s exists, skipping computation' % outFile
					continue
			annNames = get_raw_ann_files(cl, setName=s)
			imNames  = ann2imnames(annNames)

			#Prepare the file.
			N        = annClassCounts[cl] 
			outFid   = h5.File(outFile,'w')
			euler    = outFid.create_dataset("/euler", (N, 2), dtype='f')
			dist     = outFid.create_dataset("/distance", (N, 1), dtype='f')
			bbox     = outFid.create_dataset("/bbox", (N,4), dtype='f')
			imgName  = outFid.create_dataset("/imgName", (N,), dt)
			clsName  = outFid.create_dataset("/className", (N,), dt) 
			count = 0
			for (f, i) in zip(annNames, imNames):
				annDat = read_raw_annotation_file_(f)
				for dat in annDat:
					euler[count,:] = dat['viewpoint'][0:2]
					dist[count]    = dat['viewpoint'][2]
					bbox[count,:]  = dat['bbox']	
					imgName[count] = i
					clsName[count] = cl
					count += 1	
			outFid.close()

##
# Uses processed annotations and converts into annotations
# useful for an experiment. 
def save_exp_annotations(prms, forceSave=False):
	paths    = prms['paths']
	setNames = ['train','val']



##
# Gets the indexes of examples that should be used for obtaining
# for the pairs of images.
def get_indexes(className, numSamples, setName):
	assert setName in ['train', 'val'], 'Inappropriate setName'
	#Get annotation data. 
	paths     = get_paths()
	annFile   = paths['annData'] % (className, setName)
	annData   = h5.File(annFile, 'r') 
	N         = annData['euler'].shape[0]
	annData.close()
	
	#Set the seed.
	randState = np.random.get_state() 
	clIdx = CLASS_NAMES.index(className)
	seed  = 3 + (clIdx * 2) + 1 
	np.random.seed(seed)	

	#Get the indexes
	idx1 = np.random.choice(range(N),size=(numSamples,))
	idx2 = np.random.choice(range(N),size=(numSamples,))

	#Set the state back. 
	np.random.set_state(randState)
	return idx1, idx2


##
# Get the label pairs.
def get_pair_labels(prms, className, setName):
	numSamples, lblType = prms['numSamples'], prms['lblType']
	idx1, idx2 = get_indexes(className, numSamples, setName)
	assert len(idx1)==len(idx2), 'Example mismatch'
	
	#Get the data
	paths     = get_paths()
	annData   = h5.File(paths['annData'] % (className, setName),'r')
	euler     = annData['euler']
	
	#Get label dimension. 
	if lblType=='diff':
		lD = 2
		lbType = np.float32
	else:
		raise Exception('Label Type: %s is not recognized' % lblType)

	N   =  len(idx1)
	lbl =  np.zeros((N,lD)).astype(lbType) 
	count = 0
	for (i1, i2) in zip(idx1, idx2):
		ann1 = euler[i1]
		ann2 = euler[i2]
		if lblType=='diff':
			lbl[count,:] = ann2[0:lD] - ann1[0:lD]
		count += 1

	annData.close()	
	return lbl, idx1, idx2


##
# Crop the image
def crop_im(im, bbox, cropType, **kwargs):
	'''
		The bounding box is assumed to be in the form (xmin, ymin, xmax, ymax)
		kwargs:
			imSz: Size of the image required
	'''
	x1,y1,x2,y2 = bbox
	if cropType=='resize':
		imSz  = kwargs['imSz']
		imBox = im[y1:y2, x1:x2]
		imBox = scm.imresize(imBox, (imSz, imSz))
	else:
		raise Exception('Unrecognized crop type')

	return imBox		


##
# Read the image
def read_image(imName, color=True):
	'''
		color: True - if a gray scale image is encountered convert into color
	'''
	im = plt.imread(imName)
	if color:
		if im.ndim==2:
			print "Converting grayscale image into color image"
			im = np.tile(im.reshape(im.shape[0], im.shape[1],1),(1,1,3))
	return im			

##
# Extract the pair of images. 
def get_pair_images(prms, className, setName):
	numSamples, imSz, cropType = prms['numSamples'], prms['imSz'], prms['cropType']
	idx1, idx2 = get_indexes(className, numSamples, setName)
	N = len(idx1)
	assert len(idx1)==len(idx2), 'Example mismatch'

	#Get the data
	paths     = get_paths()
	annData   = h5.File(paths['annData'] % (className, setName),'r')
	imgName   = annData['imgName']
	bbox      = annData['bbox']

	#Load all the imgData
	print "Pre-Loading the images"
	imgDat = []
	for i in range(imgName.shape[0]):
		imgDat.append(read_image(imgName[i], color=True))

	print "Extracting desired crops"
	ims = np.zeros((N,2,imSz,imSz,3)).astype(np.uint8)
	count = 0
	for (i1,i2) in zip(idx1, idx2):
		ims[count,0] = crop_im(imgDat[i1], bbox[i1], cropType, imSz=imSz)
		ims[count,1]  = crop_im(imgDat[i2], bbox[i2], cropType, imSz=imSz)
		count += 1

	annData.close()		
	return ims, idx1, idx2


##
# Visualize the data
def vis_pairs(prms, className, setName, isShow=True, saveFile=None):
	ims, idxIm1, idxIm2 = get_pair_images(prms, className, setName)
	lbls, idx1 , idx2   = get_pair_labels(prms, className, setName)
	assert all(idx1==idxIm1) and all(idx2==idxIm2), 'Idx mismatch'
	plt.ion()
	fig = plt.figure()
	print "Visualizing pairs ..."
	for i in range(prms['numSamples']):
		plt.subplot(1,2,1)
		plt.imshow(ims[i,0])
		plt.subplot(1,2,2)
		plt.imshow(ims[i,1])
		plt.title('Azimuth: %f, Elevation: %f' % (lbls[i][0],lbls[i][1]))
		plt.axis('off')
		if isShow:
			plt.show()
			plt.draw()
			time.sleep(3.0)
		else:
			plt.savefig(saveFile % i)


##
# Save the visualization of the data.
def save_vis_pairs(setName='val'):
	prms   = get_exp_prms(imSz=128, lblType='diff', cropType='resize', numSamples=100)
	visDir  = os.path.join(prms['paths']['visDir'], prms['expName'], setName) 
	for cl in CLASS_NAMES:
		clDir = os.path.join(visDir, cl)
		if not os.path.exists(clDir):
			os.makedirs(clDir)
		clFile = os.path.join(clDir, 'sample_%d.jpg')
		vis_pairs(prms, cl, setName, isShow=False, saveFile = clFile) 
	

##
# Get the statistics of labels
def get_pair_label_stats(prms, setName='val'):
	numSamples, lblType = prms['numSamples'], prms['lblType']
	labels = {}
	for cl in CLASS_NAMES:	
		labels[cl],_,_ = get_pair_labels(cl, numSamples, setName, lblType='diff')
	return labels


def save_image_lmdb(prms, setName='train'):
	ims,idx1,idx2 = get_pair_images(prms, className, setName):



