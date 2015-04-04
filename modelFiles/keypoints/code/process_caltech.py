import my_pycaffe as mp
import my_pycaffe_utils as mpu
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import collections as co
import my_pycaffe_io as mpio
import scipy.misc as scm
import other_utils as ou

##
# Get the basic paths
def get_basic_paths():
	paths            = {}
	paths['dataset']   = 'caltech101' 
	paths['imDir']     = '/data1/pulkitag/data_sets/caltech101/101_ObjectCategories/'
	paths['splitsDir'] = '/data1/pulkitag/caltech101/train_test_splits/'
	paths['lmdbDir']   = '/data1/pulkitag/caltech101/lmdb-store/'
	paths['expDir']    = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/'
	paths['snapDir']   = '/data1/pulkitag/caltech101/snapshots/'
	return paths	

##
# Get the experiment prms
def get_prms(imSz=128, numTrain=30, numTest=20, runNum=0):
	prms  = {}
	paths = get_basic_paths()

	prms['numTrain'] = numTrain
	prms['numTest']  = numTest
	prms['runNum']   = runNum
	prms['imSz']     = imSz

	expStr = 'imSz%d_ntr%d_nte%d_run%d' % (imSz, numTrain, numTest, runNum)

	paths['splitsFile'] = {}
	paths['splitsFile']['train'] = os.path.join(paths['splitsDir'],
																 'train_ntr%d_nte%d_run%d.txt' % (numTrain, numTest, runNum))
	paths['splitsFile']['test']  = os.path.join(paths['splitsDir'],
																 'test_ntr%d_nte%d_run%d.txt' % (numTrain, numTest, runNum))

	paths['lmdb'] = {}
	paths['lmdb']['train'] = os.path.join(paths['lmdbDir'], 'train-%s-lmdb' % expStr)
	paths['lmdb']['test']  = os.path.join(paths['lmdbDir'], 'test-%s-lmdb' % expStr)
	prms['paths'] = paths
	return prms 

##
# Read the class names
def get_classnames(prms):
	classNames = [cl for cl in os.listdir(prms['paths']['imDir']) \
											if (cl is not None) and ('BACKGROUND' not in cl)]
	if prms['paths']['dataset'] in ['caltech101']:
		assert len(classNames)==101, 'Caltech 101 should have 101 classes, not %d' % len(classNames)
	else:
		raise Exception('Something went wrong')
	return sorted(classNames, key=lambda s: s.lower()) 

##
# Read the number of examples
def get_class_statistics_raw(prms):
	clNames     = get_classnames(prms)
	clCount     = co.OrderedDict()
	minCount    = np.inf
	maxCount    = -np.inf
	fileNames,_ = get_class_imagenames(prms)
	for cl in clNames:
		clCount[cl] = len(fileNames[cl])
		if clCount[cl] > maxCount:
			maxCount = clCount[cl]
		if clCount[cl] < minCount:
			minCount = clCount[cl]
	return clCount, maxCount, minCount


##
# Get the names of images
def get_class_imagenames(prms):
	clNames   = get_classnames(prms)
	fullNames = co.OrderedDict()
	baseNames = co.OrderedDict()
	for cl in clNames:
		dirName       = os.path.join(prms['paths']['imDir'], cl)
		fullNames[cl] = [os.path.join(dirName, p) for p in os.listdir(dirName) if '.jpg' in p]
		baseNames[cl] = [os.path.join(cl, p) for p in os.listdir(dirName) if '.jpg' in p]
	return fullNames, baseNames

##
# Form train and test splits
def save_train_test_splits(prms, forceSave=False):
	'''
		The following strategy will be used:
		1. The test number of images will be first selected from the available collection.
		2. Then, out of reamining images - a "maximum" of prms['numTrain'] will
			 be considered for training. 
	'''
	
	setNames = ['train', 'test']
	#Set the seeds
	randSeed = 2 * prms['runNum'] + 1
	oldRandState = np.random.get_state()
	randState    = np.random.RandomState(randSeed)

	#Initialize the files for the output.
	fid = {}
	for s in setNames:
		fName  = prms['paths']['splitsFile'][s]
		if os.path.exists(fName) and not forceSave:
			print 'FileName: %s existing, quitting' % fName
			return
		fid[s] = open(fName, 'w')	

	_, imFiles = get_class_imagenames(prms)
	for (i,cl) in enumerate(imFiles.keys()):
		N        = len(imFiles[cl])
		perm     = randState.permutation(N)
		ids      = {}
		ids['test']  = perm[0:prms['numTest']]
		numTrain =   min(N - prms['numTest'], prms['numTrain'])
		ids['train'] = perm[prms['numTest']:prms['numTest'] + numTrain]
		for s in setNames:
			for idx in ids[s]:
				fid[s].write('%s %d\n' % (imFiles[cl][idx], i))	

	#Close the files
	for s in setNames:
		fid[s].close()	
	np.random.set_state(oldRandState)

	#Sanity Check
	#Verify that train and test files have no common images.
	trainFid = open(prms['paths']['splitsFile']['train'],'r')
	testFid  = open(prms['paths']['splitsFile']['test'],'r')	
	trainLines = trainFid.readlines()
	testLines  = testFid.readlines()
	trainFid.close()
	testFid.close()
	for trnL in trainLines:
		for teL in testLines:
			assert not trnL.split()[0] == teL.split()[0], 'Train and test names match'

##
# Save the lmdbs
def save_lmdb(prms):
	setNames  = ['train', 'test']
	randSeeds = [101, 103]
	batchSz  = 1000
	imSz     = prms['imSz']
	imDir    = prms['paths']['imDir']
	for (i,s) in enumerate(setNames):
		print s
		#Set Seed.
		oldRandState = np.random.get_state()
		randState    = np.random.RandomState(randSeeds[i])
		#Initialize stuff. 
		splitFile = prms['paths']['splitsFile'][s]
		db        = mpio.DbSaver(prms['paths']['lmdb'][s])
		ims       = np.zeros((batchSz, imSz, imSz, 3)).astype(np.uint8)
		lbs       = np.zeros((batchSz,)).astype(int)
		#Read the data
		fid 			= open(splitFile, 'r')
		lines     = fid.readlines()
		fid.close()
		#Randomize the ordering in which the lmdb is created.
		perm      =  randState.permutation(len(lines)) 
		count     = 0
		prevCount = 0
		for l in lines:
			imName, label = l.split()
			lbs[count]    = int(label)
			ims[count]    = scm.imresize((ou.read_image(os.path.join(imDir, imName),isBGR=True)),(imSz,imSz)) 
			count += 1
			if count == batchSz:
				print "Processed %d examples" % (count + prevCount)
				ims = ims.transpose((0, 3, 1, 2))
				db.add_batch(ims, lbs, svIdx = perm[prevCount:prevCount+count])
				count = 0 
				prevCount += count
				ims       = np.zeros((batchSz, imSz, imSz, 3)).astype(np.uint8)
				lbs       = np.zeros((batchSz,)).astype(int)
		if count > 0:
			ims = ims.transpose((0, 3, 1, 2))
			db.add_batch(ims[0:count], lbs[0:count], svIdx = perm[prevCount:prevCount+count])
		db.close()
		#Set the random state back. 
		np.random.set_state(oldRandState)


def vis_lmdb(prms, setName='train'):
	#### TO CHECK RGB v/s BGR #############
	fig = plt.figure()
	plt.ion()
	ax  = plt.subplot(1,1,1)
	db  = mpio.DbReader(prms['paths']['lmdb'][setName])
	clNames = get_classnames(prms)
	for i in range(0,1000):
		im, lb = db.read_next()
		#pdb.set_trace()
		plt.imshow(im.transpose((1,2,0)))
		plt.title('Class: %s' % clNames[lb])
		ax.axis('off')
		raw_input()	
	db.close()

##
#  Info for pretrained model used for initializing the experiment. 
def get_pretrain_info(preTrainStr):
	'''
		preTrainStr: The weights to use. 
	'''
	if preTrainStr == 'alex':
		netFile = '/data1/pulkitag/caffe_models/caffe_imagenet_train_iter_310000'
	elif preTrainStr == 'rotObjs_kmedoids30_20_iter60K':
		snapshotDir   = '/data1/pulkitag/snapshots/keypoints/'
		imSz          = 128
		numIterations = 60000
		modelName  =  'keypoints_siamese_scratch_iter_%d.caffemodel' % numIterations
		netFile = os.path.join(snapshotDir, 'exprotObjs_lblkmedoids30_20_imSz%d'% imSz, modelName) 
		defFile = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/caltech101/exp/keynet_full.prototxt' 
	else:
		raise Exception('Unrecognized preTrainStr: %s' % preTrainStr)
	return netFile, defFile

#List the layers to modify in order to get the network. 
def get_modify_layers(maxLayer):
	assert maxLayer >= 1, 'All layers cannot be deleted'
	#Layers that need to be deleted so that maxLayer can be achieved.
	delLayers = {}
	delLayer['l6'] = ['drop6','relu6','fc6']
	delLayer['l5'] = ['pool5','relu5','conv5']
	delLayer['l4'] = ['relu4', 'conv4']
	delLayer['l3'] = ['relu3', 'conv3','norm2']
	delLayer['l2'] = ['pool2', 'relu2', 'conv2', 'norm1']
	allDelLayers = []
	if maxLayer < 6:
		for l in range(6,maxLayer,-1):
			allDelLayers = allDelLayers + delLayer['l%d' % l]
	allDelLayers = allDelLayers + ['accuracy', 'loss']	

	#The last feature layer.
	lfLayer['l6'] = 'drop6'
	lfLayer['l5'] = 'pool5'
	lfLayer['l4'] = 'relu4'
	lfLayer['l3'] = 'relu3'
	lfLayer['l2'] = 'pool2'
	lfLayer['l1'] = 'pool1' 
	lastLayer = lfLayer['l%d' % maxLayer]

	return allDelLayers, lastLayer	
 		

def get_caffe_prms(isPreTrain=False, maxLayer=5, fineLast=True, preTrainStr=None)
	'''
		isPreTrain: Use a pretrained network for performing classification.  		
		maxLayer  : How many layers to consider in the alexnet train.
		fineLast  : Whether to finetune only the last layer or all layers. 
	'''
	cPrms  = {}
	expStr = []
	if isPreTrain:
		assert preTrainStr is not None, 'preTrainStr cannot be none'
		expStr.append('pre-%s' % preTrainStr)
	else:
		expStr.append('pre-random')

	if fineLast:
		expStr.append('ft-last')
	else:
		expStr.append('ft-all')

	#Max-Layer
	expStr.append('mxl-%d' % maxLayer)
	#Final str
	expStr = ''.join(s + '_' for s in expStr)
	expStr = expStr[0:-1]
	cPrms['expStr']    = expStr
	cPrms['delLayers'], cPrms['lastLayer'] =   	


def make_experiment() 
