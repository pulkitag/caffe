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
import myutils as mu

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
def get_exp_prms(imSz=128, lblType='diff', cropType='contPad',
								 numSamples=20000, mnBbox=50, contPad=16,
								 azBin=30, elBin=10):
	'''
	 imSz      : Size of the image to use
	 lblType   : Type of labels to use
						   'diff'    -- difference in euler angles.
							 'uniform' -- bin the angles uniformly. 
	 azBin     : Bin size for azimuth
	 elBin     : Bin size for euler 
	 cropType  : 'resize' -- just resize the bbox to imSz
							 'contPad'-- pad the bounding box with some context.
	 contPad   :  Amount of context padding.  
	 numSamples:  Number of samples per class
	 mnBbox    :  Minimum size of bbox to consider. 
	'''
	assert lblType  in  ['diff', 'uniform'], 'Unknown lblType encountered'
	assert cropType in  ['resize', 'contPad'], 'Unknown cropType encountered' 

	prms  = {}
	prms['imSz']       =  imSz
	#Label info
	prms['lblType']    =  lblType
	prms['azBin']      =  azBin
	prms['elBin']      =  elBin
	prms['cropType']   =  cropType
	prms['numSamples'] =  numSamples
	prms['mnBbox']     =  mnBbox
	prms['contPad']    = contPad
	if cropType in ['contPad']:
		cropStr  = '%s%d' % (cropType, contPad)
	else:
		cropStr  = '%s' % (cropType)

	if lblType in ['uniform']:
		lblStr = 'uni-az%del%d' % (azBin, elBin)
	else:
		lblStr  = '%s' % lblType

	prms['expName']    = 'pascal3d_imSz%d_lbl-%s_crp-%s_ns%.0e_mb%d' % \
												(imSz, lblStr, cropStr, numSamples, mnBbox)

	paths = get_basic_paths()
	#Save the annotation data for the experiment. 
	if mnBbox > 0:
		paths['annData']    = os.path.join(paths['myData'], 'Annotation',
													 '%s_%s_mb%d.hdf5' % ('%s','%s',mnBbox))
	else:
		paths['annData']    = os.path.join(paths['myData'], 'Annotation', '%s_%s.hdf5') #class_setname
		#LMDB Paths
	paths['lmdb-im'] = os.path.join(paths['lmdbDir'], 
											'%s_images_pascal3d_imSz%d_crp-%s_ns%.0e_mb%d-lmdb' %\
											 ('%s', imSz, cropStr, numSamples, mnBbox))
	paths['lmdb-lb'] = os.path.join(paths['lmdbDir'], 
											'%s_labels_pascal3d_lbl-%s_ns%.0e-lmdb' % ('%s', lblStr, numSamples))

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
	assert setName in ['train','val'], 'Incorrect set name'
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

	if setName == 'train':
		#Imagenet Data only used for training.
		imgAnnDir = paths['rawAnnData'] % (className, 'imagenet')
		imgNames  = [os.path.join(imgAnnDir, p) for p in os.listdir(imgAnnDir) if '.mat' in p]
		allNames = pascalNames + imgNames
	else:
		allNames = pascalNames

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
			ann = read_raw_annotation_file_(f)
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
# Determined a bbox is valid for the experiment or not. 
def is_valid_bbox_(prms, bbox):
	assert bbox.ndim ==2
	assert bbox.shape[1]==4
	if prms['mnBbox'] > 0:
		xSz  = bbox[:,2] - bbox[:,0]
		ySz  = bbox[:,3] - bbox[:,1]
		minSz   = np.minimum(xSz.squeeze(), ySz.squeeze())
		isValid = (minSz >= prms['mnBbox']).squeeze()
		#maxSz = np.maximum(xSz.squeeze(), ySz.squeeze())
		#isValid = (maxSz >= prms['mnBbox']).squeeze()
	else:
		isValid = np.ones((bbox.shape[0],), dtype=bool)
	return isValid


##
# For a specific experiment get number of boxes per class
def get_class_statistics_exp(prms, setName='train'):
	paths = prms['paths']
	annCount = {}		
	for cl in CLASS_NAMES:
		annFile      = paths['procAnnData'] % (cl, setName)
		fid          = h5.File(annFile,'r')
		if prms['mnBbox'] > 0:
			isValid      = is_valid_bbox_(prms, fid['bbox'][:])
			annCount[cl] = sum(isValid)
		else:
			annCount[cl] = fid['bbox'].shape[0]
		fid.close()
	return annCount


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
	paths    = get_basic_paths()
	#setNames = ['train', 'val']
	setNames = ['val']
	dt       = h5.special_dtype(vlen=unicode)
	for s in setNames:
		print "Set: %s" % s
		_,annClassCounts = get_class_statistics_raw(setName=s)
		for cl in CLASS_NAMES:
			print "Processing class: %s" % cl
			outFile  = paths['procAnnData'] % (cl, s)
			if os.path.exists(outFile):
				if forceSave:
					print "Deleting old file: %s" % outFile
					os.remove(outFile)	
				else:
					print 'File: %s exists, skipping computation' % outFile
					continue
			annNames = get_raw_ann_files(cl, setName=s)
			imNames  = ann2imnames_(annNames)

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
	for s in setNames:
		print "Set: %s" % s
		#Computed the number of example
		annClassCounts = get_class_statistics_exp(prms, setName=s)
		for cl in CLASS_NAMES:
			print "Processing class: %s" % cl
			outFile  = paths['annData'] % (cl, s)
			inFile   = paths['procAnnData'] % (cl, s)
			if os.path.exists(outFile):
				if forceSave:
					print "Deleting old file: %s" % outFile
					os.remove(outFile)	
				else:
					print 'File: %s exists, skipping computation' % outFile
					continue
			outFid = h5.File(outFile, 'w')
			inFid  = h5.File(inFile,  'r')
			N      = annClassCounts[cl]
			count  = 0
			if prms['mnBbox'] > 0:
				isValid = is_valid_bbox_(prms, inFid['bbox'][:])
			else:
				assert N == inFid['bbox'].shape[0]
				isValid  = np.ones((N,), dtype=bool)
	
			for key in inFid.keys():
				inDat    = inFid[key][:]
				outShp   = tuple([N] + [shp for shp in inDat.shape[1:]])
				if key in ['imgName', 'className']:
					outType = h5.special_dtype(vlen=unicode)
				else:
					outType = inDat.dtype
				outDat = outFid.create_dataset("/%s" % key, outShp, dtype=outType)
				outDat[:] = inDat[isValid]
			outFid.close()
			inFid.close()  


##
# Gets the indexes of examples that should be used for obtaining
# for the pairs of images.
def get_indexes(prms, className, setName):
	assert setName in ['train', 'val'], 'Inappropriate setName'
	#Get annotation data. 
	paths      = prms['paths']
	numSamples = prms['numSamples'] 
	annFile   = paths['annData'] % (className, setName)
	annData   = h5.File(annFile, 'r') 
	N         = annData['euler'].shape[0]
	annData.close()
	
	#Set the seed.
	randState = np.random.get_state() 
	clIdx = CLASS_NAMES.index(className)
	seed  = 3 + (clIdx * 2) + 1 
	newRandState = np.random.RandomState(seed)	

	#Get the indexes
	idx1  = newRandState.choice(range(N),size=(numSamples,))
	idx2  = newRandState.choice(range(N),size=(numSamples,))

	#Set the state back. 
	np.random.set_state(randState)
	return idx1, idx2


##
# Get labels for single images.
def get_labels(prms, className, setName):
	numSamples, lblType = prms['numSamples'], prms['lblType']
	idx1, idx2 = get_indexes(prms, className, setName)
	N = len(idx1)
	#Get the data
	paths     = prms['paths']
	annData   = h5.File(paths['annData'] % (className, setName),'r')
	euler     = annData['euler'][:]
	lbl       = np.zeros((N,2)).astype(np.float32)
	for (i,idx) in enumerate(idx1):
		lbl[i] =  euler[idx]
	return lbl


##
# Get the label pairs.
def get_pair_labels(prms, className, setName):
	numSamples, lblType = prms['numSamples'], prms['lblType']
	idx1, idx2 = get_indexes(prms, className, setName)
	assert len(idx1)==len(idx2), 'Example mismatch'
	
	#Get the data
	paths     = prms['paths']
	annData   = h5.File(paths['annData'] % (className, setName),'r')
	euler     = annData['euler']
	
	#Get label dimension. 
	if lblType=='diff':
		lD      = 2
		lbClass = np.float32
	if lblType =='uniform':
		lD      = 2
		lbClass = np.int32
		azBins  = np.array(range(0,360, prms['azBin']))
		elBins  = np.array(range(0,180, prms['elBin']))
	else:
		raise Exception('Label Type: %s is not recognized' % lblType)

	N   =  len(idx1)
	lbl =  np.zeros((N,lD)).astype(lbClass) 
	count = 0
	
	for (i1, i2) in zip(idx1, idx2):
		ann1 = euler[i1]
		ann2 = euler[i2]
		az  = np.mod(ann2[0] - ann1[0], 360)
		el  = np.mod(ann2[1] - ann1[1], 180)
		if lblType == 'diff':
			#The Azimuth will be converted to (-pi, pi]
			if az > 180:
				az = -(360 - az)
			#The Elevation will be from [-pi/2 to pi/2]
			if el > 90:
				el = -(180 - el)
			lbl[count,:] = np.array([az, el]).reshape(1,2)
		elif lblType == 'uniform':
			lbl[count,0] = mu.find_bin(az, azBins)
			lbl[count,1] = mu.find_bin(el, elBins)
		else:
			raise Exception('Label Type: %s is not recognized' % lblType)
		count += 1
	annData.close()	
	return lbl, idx1, idx2


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
	idx1, idx2 = get_indexes(prms, className, setName)
	N = len(idx1)
	assert len(idx1)==len(idx2), 'Example mismatch'

	#Get the data
	paths     = prms['paths']
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
		ims[count,0] = crop_im(imgDat[i1], bbox[i1], **prms)
		ims[count,1] = crop_im(imgDat[i2], bbox[i2], **prms)
		count += 1

	annData.close()		
	return ims, idx1, idx2


##
# Visualize the data
def vis_image_labels(prms, className, setName, isShow=True, saveFile=None, isPairLabel=True):
	ims, idxIm1, idxIm2 = get_pair_images(prms, className, setName)
	if isPairLabel:
		lbls, idx1 , idx2   = get_pair_labels(prms, className, setName)
		assert all(idx1==idxIm1) and all(idx2==idxIm2), 'Idx mismatch'
	else:
		lbls = get_labels(prms, className, setName)
	#plt.ion()
	fig = plt.figure()
	ax1 = plt.subplot(1,2,1)
	ax2 = plt.subplot(1,2,2)
	print "Visualizing pairs ..."
	for i in range(prms['numSamples']):
		print i
		ax1.imshow(ims[i,0])
		if isPairLabel:
			ax2.imshow(ims[i,1])
		plt.title('Azimuth: %f, Elevation: %f' % (lbls[i][0],lbls[i][1]))
		plt.axis('off')
		if isShow:
			ax1.figure.canvas.draw()
			plt.ion()
			plt.show()
			#plt.draw()
			#time.sleep(3.0)
			raw_input()
		else:
			plt.savefig(saveFile % i)
	plt.close(fig)

##
# Save the visualization of the data.
def save_vis_pairs(setName='val', prms=None):
	if prms is None:
		prms    = get_exp_prms(imSz=128, lblType='diff', cropType='resize', numSamples=100, mnBbox=50)
	visDir  = os.path.join(prms['paths']['visDir'], prms['expName'], setName) 
	for cl in CLASS_NAMES:
		clDir = os.path.join(visDir, cl)
		if not os.path.exists(clDir):
			os.makedirs(clDir)
		clFile = os.path.join(clDir, 'sample_%d.jpg')
		vis_image_labels(prms, cl, setName, isShow=False, saveFile = clFile, isPairLabel=True) 
	

##
# Get the statistics of labels
def get_pair_label_stats(prms, setName='val'):
	labels = {}
	for cl in CLASS_NAMES:	
		labels[cl],_,_ = get_pair_labels(prms, cl, setName)
	return labels

##
# Converts a siamese img into caffe compatible format. 
def imSiamese2imCaffe(ims):
	'''
		imS: N * 2 * rows * cols * channels
	'''
	N,_,rows, cols, chnls = ims.shape
	ims = ims.transpose((0,1,4,2,3))
	ims = ims.reshape((N, 2 * chnls, rows, cols))
	return ims 


def save_lmdb(prms, setName='train'):
	nCl     = len(CLASS_NAMES)
	nsPerCl = prms['numSamples']
	N       = nsPerCl * nCl

	#Set random state
	oldRandState = np.random.get_state()
	randState    = np.random.RandomState(7)
	svIdx        = randState.permutation(N)

	db      = mpio.DoubleDbSaver(prms['paths']['lmdb-im'] % setName, prms['paths']['lmdb-lb'] % setName)
	batchSz = 1000		 
	count   = 0

	if prms['lblType'] in ['uniform']:
		lblAsFloat = False
		lbType     = np.int
		lblSz      = 2 #2 for azimuth, el bins
	else:
		raise Exception('lblType not recognized')	

	count = 0
	for (icl,cl) in enumerate(CLASS_NAMES):
		print 'Saving for class: %s' % cl
		ims,idx1,idx2     = get_pair_images(prms, cl, setName)
		lbs,idx1Lb,idx2Lb = get_pair_labels(prms, cl, setName)
		assert all(idx1==idx1Lb) and all(idx2==idx2Lb), 'idx mismatch'	
		for i in range(0,len(idx1),batchSz):
			st = i
			en = min(st + batchSz, len(idx1))
			#Form the batch
			imBatch  = imSiamese2imCaffe(ims[st:en]).astype(np.uint8)
			lbBatch  = lbs[st:en].astype(lbType)
			lbBatch  = lbBatch.reshape((en - st, lblSz, 1, 1)) 
			clsLabel = icl * np.ones((en-st,)).astype(np.int)
			batchIdx = svIdx[count : count + (en-st)]
			#Save the batch
			db.add_batch((imBatch, lbBatch), (clsLabel, None),
									  imAsFloat=(False,lblAsFloat), svIdx=(batchIdx, batchIdx))
			count = count + (en-st)
			print count, imBatch.shape, lbBatch.shape, clsLabel.shape
	db.close()


def vis_lmdb(prms, setName):
	db = mpio.DoubleDbReader((prms['paths']['lmdb-im'] % setName, prms['paths']['lmdb-lb'] % setName))
	fig = plt.figure()
	ax1 = plt.subplot(1,2,1)
	ax2 = plt.subplot(1,2,2)
	plt.ion()
	for i in range(100):
		imDat, rotLbl, clLbl, _ = db.read_batch_data_label(1)
		im1 = imDat[0,0:3].transpose((1,2,0))
		im2 = imDat[0,3:].transpose((1,2,0))
		ax1.imshow(im1)
		ax1.axis('off')
		ax2.imshow(im2)
		ax2.axis('off')
		rotLbl = rotLbl.squeeze()
		clLbl  = clLbl[0][0]
		clName = CLASS_NAMES[clLbl]
		plt.title('Cls: %s, Az: %d, Eu: %d' % (clName, rotLbl[0], rotLbl[1]))
		plt.show()
		raw_input()
	db.close()	

