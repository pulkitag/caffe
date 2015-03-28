import numpy as np
import scipy.io as sio
import h5py as h5
import os
import pdb
import shutil

CLASS_NAMES = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus',
							'car', 'chair', 'diningtable', 'motorbike', 'sofa',
							'train', 'tvmonitor']

##
# Paths for dealing with PASCAL-3D
def get_paths():
	paths = {}
	paths['data']    = '/data1/pulkitag/data_sets/pascal_3d/PASCAL3D+_release1.1/'
	paths['myData']  = '/data1/pulkitag/data_sets/pascal_3d/my/'
	paths['annData']    = os.path.join(paths['myData'], 'Annotation', '%s_%s.hdf5') #class_setname
	paths['rawAnnData'] = os.path.join(paths['data'], 'Annotationsv73', '%s_%s') #Fill with class_dataset
	paths['pascalDir'] = os.path.join(paths['data'], 'PASCAL','VOCdevkit','VOC2012')
	paths['pascalSet'] = os.path.join(paths['pascalDir'], 'ImageSets', 'Main', '%s_%s.txt') 
	paths['imDir']     = os.path.join(paths['data'], 'Images')
	return paths


##
# Helper function for read_file
# Reads the fields
def read_field(data, key):
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
# Reads a file and finds the relevant data
def read_file(fileName):
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
				valObj[rf] = read_field(refs[objs[rf][i][0]], rf)
			if valObj['viewpoint'] is not None:
				vals.append(valObj)	
	else:
		valObj = {}
		for rf in readFields:
			valObj[rf] = read_field(objs[rf], rf)
		if valObj['viewpoint'] is not None:
			vals.append(valObj)
	dat.close()
	return vals


##
#Find the filenames for train and val
def get_raw_ann_files(className, setName='train'):
	assert className in CLASS_NAMES, '% class not found' % className
	paths = get_paths()	
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
def get_class_statistics(setName='train'):
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
# From a list of names - find the image names
def ann2imnames(annNames):
	paths   = get_paths()
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
# Process and save the annotations.
def save_processed_annotations():
	'''
		If the image has multiple bounding boxes then list them as seperate example.
		Fields:
		euler
		distance
		class
		bbox
		imName: Path to the image. 
	'''
	paths = get_paths()
	setNames = ['train', 'val']
	dt = h5.special_dtype(vlen=unicode)
	for s in setNames:
		print "Set: %s" % s
		_,annClassCounts = get_class_statistics(setName=s)
		for cl in CLASS_NAMES:
			print "Processing class: %s" % cl
			annNames = get_raw_ann_files(cl, setName=s)
			imNames  = ann2imnames(annNames)
			outFile  = paths['annData'] % (cl, s)
			print outFile
			if os.path.exists(outFile):
				print "Deleting old file: %s" % outFile
				os.remove(outFile)	

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
				annDat = read_file(f)
				for dat in annDat:
					euler[count,:] = dat['viewpoint'][0:2]
					dist[count]    = dat['viewpoint'][2]
					bbox[count,:]  = dat['bbox']	
					imgName[count] = i
					clsName[count] = cl
					count += 1	
			outFid.close()



##
# Get the annotation data
def get_annotation_data(annNames):
	'''
		annNames: iterable over annotation file-names 
	'''
	annData = []
	for f in annNames:
		annData.append(read_file(f))
	return annData


##
# Get the image data. 
def get_image_data(imNames):
	'''
		imNAmes: iterable over image file-names. 
	'''
	imData = []
	for f in imNames:
		imData.append(plt.imread(f))
	return imData

	

##
# Gets the indexes of examples that should be used for obtaining
# for the pairs of images.
def get_indexes(className, numSamples, setName):
	assert setName in ['train', 'val'], 'Inappropriate setName'
	#Get annotation data. 
	annNames  = get_class_files(className, setName=setName) 
	annData   = get_annotation_data(annNames)
	N         = len(annData)
	
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
def get_pair_labels(className, numSamples, setName, lblType='diff'):
	idx1, idx2 = get_indexes(className, numSamples, setName)
	assert len(idx1)==len(idx2), 'Example mismatch'
	annNames  = get_class_files(className, setName=setName) 
	annData   = get_annotation_data(annNames)
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
		ann1 = annData[i1]
		ann2 = annData[i2]
		if lblType=='diff':
			lbl[count,:] = ann2[0:lD] - ann1[0:lD]
		count += 1
	
	return lbl


##
# Extract the pairwise data for a single class. 
def get_class_data(className, numSamples, setName, seed=None):
	'''
		numSamples: Number of samples to extract
		setName   : train or val set. 
	'''
	assert setName in ['train', 'val'], 'Inappropriate setName'
	#Get annotation data. 
	annNames  = get_class_files(className, setName=setName) 
	annData   = get_annotation_data(annNames)
	#Get imag data 
	imNames   = ann2imnames(annNames)
	imData    = get_image_data(imNames)
	assert len(annData) == len(imData), 'Size Mismatch'
	N = len(annData)

	#Set the seed. 	
			
	
	
	


	
	
