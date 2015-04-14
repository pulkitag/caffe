import scipy.misc as scm
import pickle
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
import my_pycaffe_io as mpio
import my_pycaffe_utils as mpu
import pdb
import myutils as myu

SET_NAMES = ['test', 'train']

def get_lmdb_name_old(prms, setName):
	dbDir  = prms['paths']['lmdbDir']
	repNum = prms['repNum']
	if setName == 'train':
		dbDir = os.path.join(dbDir, 'train')
	elif setName == 'test':
		dbDir = os.path.join(dbDir, 'test')
	else:
		raise Exception('Unrecognized Set Name')

	if repNum is not None:
		expName = expName + ('_rep%d' % repNum)

	if expName in ['normal', 'null_transform']:
		imFile = os.path.join(dbDir, 'images_labels_' + expName + '-lmdb')
		lbFile = []
	else:
		imFile = os.path.join(dbDir, 'images_' + expName + '-lmdb')
		lbFile = os.path.join(dbDir, 'labels_' + expName + '-lmdb')
	return imFile, lbFile


def get_paths(isOld=False):
	paths = {}
	paths['dataset'] = 'mnist'
	if isOld:
		paths['lmdbDir'] = '/data1/pulkitag/mnist/lmdb-store/'
	else:
		paths['lmdbDir'] = '/data0/pulkitag/mnist/lmdb-store/'
	paths['expDir']  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/exp/'
	paths['snapDir'] = '/data1/pulkitag/mnist/snapshots/'
	paths['resDir']  = '/data1/pulkitag/mnist/results/'
	paths['visDir']  = '/data1/pulkitag/mnist/vis/'
	return paths


def get_prms(transform='rotTrans', maxRot=10, maxDeltaRot=10, maxDeltaTrans=3, 
						isOld=False, baseLineMode=False, runNum=0, numTrainEx=1e+06, numTestEx=1e+04,
						lossType='regress'):
	'''
		Ideally I should finetune both for classification + rotation training
		But, I don't think I have time to try to make that work. 
		First what I will try is to learn good filters using just unsupervised training.  
	'''
	paths = get_paths(isOld)
	prms  = {}
	prms['transform']     = transform
	prms['maxRot']        = maxRot
	prms['maxDeltaRot']   = maxDeltaRot
	prms['maxDeltaTrans'] = maxDeltaTrans
	prms['isOld']         = isOld
	prms['baseLineMode']  = baseLineMode
	prms['repNum']        = runNum
	prms['numEx']          = {}
	prms['numEx']['train'] = numTrainEx
	prms['numEx']['test']  = numTestEx 
	prms['lossType']       = lossType

	if prms['lossType'] == 'classify':
		prms['numTrnBins'] = int(2 * maxDeltaTrans + 1)
		prms['numRotBins'] = int(round((2 * maxDeltaRot + 1) / float(3)))
		prms['trnBins']    = np.linspace(-maxDeltaTrans - 0.2, maxDeltaTrans + 0.2, prms['numTrnBins'],
													endpoint=False)
		prms['rotBins']    = np.linspace(-maxDeltaRot, maxDeltaRot, prms['numRotBins'], endpoint=False)
		prms['trnLblSz']   = prms['numTrnBins'] + 1 
		prms['rotLblSz']   = prms['numRotBins'] + 1

	prms['paths']         = paths	

	if isOld:  
		if transform == 'normal':
			expName = 'normal'
		else:
			if baseLineMode:
				expName = 'mnist_baseline_transform_classify_%s_dRot%d_dTrn%d_mxRot%d_nLbl%.0e_numEx%.0e'\
									% (setName, maxDeltaRot, maxDeltaTrans, maxRot, numLabels, numEx)
			else:
				expName = 'mnist_transform_classify_%s_dRot%d_dTrn%d_mxRot%d_nLbl%.0e_numEx%.0e'\
									 % (setName, maxDeltaRot, maxDeltaTrans, maxRot, numLabels, numEx)
	else:
		expName = 'tf-%s' % prms['transform']
		if lossType=='classify':
			expName = expName + '_' + 'cls'
		if transform=='rotTrans':
			expName = expName + '_dRot%d_dTrn%d_mxRot%d_tr%.0e_run%d' %\
							 (maxDeltaRot, maxDeltaTrans, maxRot, numTrainEx, runNum)
			teExpName = expName + '_dRot%d_dTrn%d_mxRot%d_te%.0e_run%d' %\
							 (maxDeltaRot, maxDeltaTrans, maxRot, numTestEx, runNum)
		elif transform=='normal':
			expStr    = 'normal'
			assert numTestEx==1e+04, 'There should be 10K test examples for normal exp'
			expName   = expStr + 'tr%.0e_run%d' % (numTrainEx, runNum)
			teExpName = expStr + 'te%.0e_run%d' % (numTestEx, runNum)
		else:
			raise Exception('Unrecognized transform type')		
	prms['expName'] = expName

	paths['lmdb'] = {}
	paths['lmdb']['train'] = {}
	paths['lmdb']['test'] = {}
	if not isOld:
		paths['lmdb']['train']['im'] = os.path.join(paths['lmdbDir'], 'train','images_%s-lmdb' % expName)
		paths['lmdb']['train']['lb'] = os.path.join(paths['lmdbDir'], 'train','labels_%s-lmdb' % expName)
		paths['lmdb']['test']['im'] = os.path.join(paths['lmdbDir'], 'test', 'images_%s-lmdb' % teExpName)
		paths['lmdb']['test']['lb'] = os.path.join(paths['lmdbDir'], 'test', 'labels_%s-lmdb' % teExpName)
	else:
		paths['lmdb']['train']['im'], paths['lmdb']['train']['lb'] = \
					get_lmdb_name_old(prms, 'train')
		paths['lmdb']['test']['im'], paths['lmdb']['test']['lb'] = \
					get_lmdb_name_old(prms, 'test')

	paths['resFile'] = os.path.join(paths['resDir'], expName, '%s.h5') #%s by caffePrms

	prms['paths'] = paths
	return prms


def load_images(setName = 'train'):
	#dataPath = '/data1/pulkitag/mnist/raw/'
	dataPath = '/data1/pulkitag/data_sets/mnist/'
	if setName == 'train':
		dataFile  = dataPath + 'trainImages.pkl'
		labelFile = dataPath + 'trainLabels.pkl' 
	elif setName=='test':
		dataFile  = dataPath + 'testImages.pkl'
		labelFile = dataPath + 'testLabels.pkl' 
	else:
		raise Exception('Unrecognized dataset')

	with open(dataFile,'r') as f:
		im = pickle.load(f)
	
	with open(labelFile, 'r') as f:
		label = pickle.load(f)
	
	return im, label

##
def make_normal_lmdb(prms, setName):
	'''
		The Standard MNIST LMDBs
	'''
	assert prms['transform'] == 'normal'
	db  = mpio.DbSaver(prms['paths']['lmdb'][setName]['im'])
	im, label = load_images(setName)
	N, nr, nc = im.shape
	im        = im.reshape((N, 1, nr, nc))
	label     = label.squeeze()
	label     = label.astype(np.int)
	repNum    = prms['repNum']

	if setName=='train':
		oldState  = np.random.get_state()
		randSeed  = np.random.seed(3 * (2 * 1* 100 + 1) + (repNum + 1) * 2)
		randState = np.random.RandomState(randSeed)
		assert prms['numEx'][setName] <= N
		perm      = randState.permutation(N)
		perm      = perm[0:prms['numEx'][setName]]
		np.random.set_state(oldState)
	else:
		perm = range(N)

	db.add_batch(im[perm], label[perm])	
	db.close()

##
def make_null_transform_lmdb(prms, setName='test'):
	'''
		The LMDB with standard Images - but in a siamese way. 
	'''
	assert prms['expName'] == 'null_transform'
	db  = mpio.DbSaver(prms['paths'][setName]['im'])
	expName = 'null_transform'
	im, label = load_images(setName)
	#Arrange the labels
	label     = label.squeeze()
	label     = label.astype(np.int)
	#Arrange the images
	N, nr, nc = im.shape
	ims       = np.zeros((2, N, nr, nc)).astype(im.dtype)
	ims[0]    = im
	ims[1]    = im
	ims       = ims.transpose((1,0,2,3))
	print ims.shape
	ims       = ims.reshape((N, 2, nr, nc))
	db.add_batch(ims, label)	
	db.close()


def make_same_transform_lmdb(setName='test', maxDeltaTrans=3, maxRot=10):
	'''
		The LMDB with standard Images - but in a siamese way and transformed by the same parameter. 
	'''
	expName = 'same_transform_trn%d_mxRot%d' % (maxDeltaTrans, maxRot)
	dbFile,_ = get_lmdb_name(expName, setName)
	db  = mpio.DbSaver(dbFile)
	im, label = load_images(setName)
	#Arrange the labels
	label     = label.squeeze()
	label     = label.astype(np.int)
	#Arrange the images
	N, nr, nc = im.shape
	ims       = np.zeros((N, 2, nr, nc)).astype(im.dtype)
	for i in range(N):
		x1, y1, r1 = sample_transform(maxDeltaTrans), sample_transform(maxDeltaTrans), sample_transform(maxRot)
		ims[i,0,:,:]  = transform_im(im[i], x1, y1, r1)
		ims[i,1,:,:]  = transform_im(im[i], x1, y1, r1)
	
	db.add_batch(ims, label)	


def transform_im(im, deltaX, deltaY, theta):
	'''
		Rotate by theta
		Translate by deltaX, deltaY
		First Rotate and then translate
		Translation is by done by padding. 	
		Currently only works for MNIST/Gray scale images
	'''
	assert im.ndim == 2, 'Only 2D images are considered'
	nr, nc = im.shape
	
	#Rotate the image
	im     = scm.imrotate(im, theta)

	#Translate the image
	#deltaY	
	if deltaY > 0:
		col    = im[0, :].reshape((1, nc))
		tileIm = np.tile(col, (deltaY, 1))
		im[deltaY:,:]  = im[0:-deltaY,:]
		im[0:deltaY,:] = tileIm
	elif deltaY < 0:
		deltaY = -deltaY
		col    = im[-1, :].reshape((1, nc))
		tileIm = np.tile(col, (deltaY, 1))
		im[0:-deltaY,:]  = im[deltaY:,:]
		im[-deltaY:,:]   = tileIm

	if deltaX > 0:
		col    = im[:, 0].reshape((nr, 1))
		tileIm = np.tile(col, (1, deltaX))
		im[:, deltaX:]  = im[:, 0:-deltaX]
		im[:, 0:deltaX] = tileIm
	elif deltaX < 0:
		deltaX = -deltaX
		col    = im[:, -1].reshape((nr, 1))
		tileIm = np.tile(col, (1, deltaX))
		im[:, 0:-deltaX]  = im[:, deltaX:]
		im[:, -deltaX:]   = tileIm

	return im


def sample_transform(maxVal, randState = None):
	if randState is None:
		val = int(round(np.random.random() * maxVal))
		rnd = np.random.random()
	else:
		val = int(round(randState.rand() * maxVal))
		rnd = randState.rand()
	if rnd > 0.5:
		sgn = 1
	else:
		sgn = -1
	return sgn * val


def make_transform_db(prms):
	for sNum,s in enumerate(SET_NAMES):
		numEx  = prms['numEx'][s]
		repNum = prms['repNum']
		imFile, lbFile = prms['paths']['lmdb'][s]['im'], prms['paths']['lmdb'][s]['lb'] 	
		db     = mpio.DoubleDbSaver(imFile, lbFile)
		#Image and label data	
		ims, clLbs  = load_images(setName=s)
		N, nr, nc = ims.shape
	
		oldState  = np.random.get_state()
		randSeed  = np.random.seed(3 * (2 * sNum * 100 + 1) + (repNum + 1) * 2)
		randState = np.random.RandomState(randSeed)
		idxs = randState.random_integers(0, N-1, numEx)

		#Start Writing the data
		count   = 0
		batchSz = 10000
		imBatch = np.zeros((batchSz, 2, nr, nc)).astype(np.uint8)
		lbBatch = np.zeros((batchSz, 3, 1, 1)).astype(np.float)

		maxDeltaTrans, maxDeltaRot, maxRot = prms['maxDeltaTrans'],\
														 prms['maxDeltaRot'], prms['maxRot'] 
		for (i,idx) in enumerate(idxs):
			im = ims[idx]
			#lb = lbs[idx]	
			#Get the Transformation
			x1, y1, r1 = sample_transform(maxDeltaTrans, randState),\
									 sample_transform(maxDeltaTrans, randState), sample_transform(maxRot, randState)
			delx, dely, delr = sample_transform(maxDeltaTrans, randState),\
									 sample_transform(maxDeltaTrans, randState), sample_transform(maxDeltaRot, randState)
			x2, y2, r2 = x1 + delx, y1 + dely, r1 + delr
			delx, dely, delr = np.float32(delx), np.float32(dely), np.float32(delr)
		
			if prms['lossType'] == 'classify':
				delx, dely = myu.find_bin(delx, prms['trnBins']),\
											 myu.find_bin(dely, prms['trnBins'])
				delr       = myu.find_bin(delr, prms['rotBins'])
				assert delx >=0 and dely>=0 and delr>=0	
			else:
				#For regression do the normalization otherwise not. 
				delx, dely, delr = delx / (maxDeltaTrans + 1e-6), dely / (maxDeltaTrans + 1e-6),\
												 delr / (maxDeltaRot + 1e-6)
		

			#Transform the image
			imBatch[count,0,:,:]  = transform_im(im, x1, y1, r1)
			imBatch[count,1,:,:]  = transform_im(im, x2, y2, r2)
			lbBatch[count,:,:,:]  = np.asarray([delx, dely, delr]).reshape((3,1,1)) 
			count += 1
			#Save to db
			if count==batchSz or i == len(idxs)-1:
				print 'Processed %d files' % (i+1)
				imBatch = imBatch[0:count]
				lbBatch = lbBatch[0:count]
				db.add_batch((imBatch, lbBatch), imAsFloat=(False, True))
				count = 0
				imBatch = np.zeros((batchSz, 2, nr, nc)).astype(np.uint8)
				lbBatch = np.zeros((batchSz, 3, 1, 1)).astype(np.float)
		np.random.set_state(oldState)
		
		## Verify that all examples have been stored.
		#im-db
		dbRead = mpio.DbReader(imFile)
		saveCount = dbRead.get_count()
		assert saveCount == numEx, 'All examples have not been stored'
		dbRead.close()
		#Label-db
		dbRead = mpio.DbReader(lbFile)
		saveCount = dbRead.get_count()
		assert saveCount == numEx, 'All examples have not been stored'
		dbRead.close()




def make_transform_label_db(setName='train',numLabels=1000, 
														maxDeltaRot=5, maxDeltaTrans=3, maxRot=5, 
														numEx=None, baseLineMode=False, repNum=None):
	'''
		setName: Train or Test
		numLabels: number of examples for which the ground truth labels are provided. 
		maxDeltaRot  : The maximum amount of Delta in rotation (in degrees)
		maxDeltaTrans: The maximum amoubt of Delta in translation (in pixels)
		maxRot       : The maximum baseline rotation. The Delta Rotation is applied after maxRot 	
		numEx        : Number of examples to include
		baseLineMode : If this is true then only save images that have classification labels
		repNum       : If the lmdb for repeats should be made or not. 
	'''
	
	ims, lbs  = load_images(setName=setName)
	N, nr, nc = ims.shape

	#The ignore label
	ignoreLabel = 10

	if numEx is None:
		if setName=='train':
			numEx = 2e+5
		elif setName=='test':
			numEx = N
		else:
			raise Exception("Inavlid setName")

	if setName=='test':
		assert numLabels==N, 'In Testing we use labels for all the examples'

	#Set the Seed
	if repNum is None:
		np.random.seed(3)
	else:
		np.random.seed(3 + (repNum + 1) * 2)

	#Determine the examples for which labels are provided
	perm = np.random.permutation(range(N))
	perm = perm[0:numLabels]

	#Experiment Name
	if baseLineMode:
		expName = 'mnist_baseline_transform_classify_%s_dRot%d_dTrn%d_mxRot%d_nLbl%.0e_numEx%.0e' % (setName, maxDeltaRot, maxDeltaTrans, maxRot, numLabels, numEx)
	else:
		expName = 'mnist_transform_classify_%s_dRot%d_dTrn%d_mxRot%d_nLbl%.0e_numEx%.0e' % (setName, maxDeltaRot, maxDeltaTrans, maxRot, numLabels, numEx)
	

	idxs = np.random.random_integers(0, N-1, numEx)

	#dbReader
	imFile, lbFile = get_lmdb_name(expName, setName, repNum=repNum)
	print imFile, lbFile
	db     = mpio.DoubleDbSaver(imFile, lbFile)

	#Start Writing the data
	count   = 0
	batchSz = 1000
	imBatch = np.zeros((batchSz, 2, nr, nc)).astype(np.uint8)
	lbBatch = np.zeros((batchSz, 4, 1, 1)).astype(np.float)
	for (i,idx) in enumerate(idxs):
		im = ims[idx]
		lb = lbs[idx]	

		#Use the Ignore Label if needed
		if idx not in perm:
			lb = ignoreLabel
			if baseLineMode:
				continue

		#Get the Transformation
		x1, y1, r1 = sample_transform(maxDeltaTrans), sample_transform(maxDeltaTrans), sample_transform(maxDeltaRot)

		delx, dely, delr = sample_transform(maxDeltaTrans), sample_transform(maxDeltaTrans), sample_transform(maxDeltaRot)

		x2, y2, r2 = x1 + delx, y1 + dely, r1 + delr

		delx, dely, delr = np.float32(delx), np.float32(dely), np.float32(delr)
		delx, dely, delr = delx / maxDeltaTrans, dely / maxDeltaTrans, delr / maxDeltaRot
	
		#Transform the image
		imBatch[count,0,:,:]  = transform_im(im, x1, y1, r1)
		imBatch[count,1,:,:]  = transform_im(im, x2, y2, r2)
		lbBatch[count,:,:,:]  = np.asarray([np.float(lb), delx, dely, delr]).reshape((4,1,1)) 
		count += 1

		if count==batchSz or i == len(idxs)-1:
			print 'Processed %d files' % i
			imBatch = imBatch[0:count]
			lbBatch = lbBatch[0:count]
			db.add_batch((imBatch, lbBatch), imAsFloat=(False, True))
			count = 0
			imBatch = np.zeros((batchSz, 2, nr, nc)).astype(np.uint8)
			lbBatch = np.zeros((batchSz, 4, 1, 1)).astype(np.float)


##
# Make lmdbs for repeats. 
def make_rep_dbs():
	numLabels   = [1e+02, 1e+03, 1e+04]
	maxDeltaRot = 10
	maxRot      = 10
	numEx       = 1e+06
	#Test
	for r in range(5):
		make_transform_label_db(numLabels=1e+04, numEx=1e+04, maxDeltaRot=10, 
																maxRot=10, repNum=r, setName='test') 
	#Train
	for r in range(5):
		for nl in numLabels:
				make_transform_label_db(numLabels=nl, numEx=numEx, maxDeltaRot=10, 
																maxRot=10, repNum=r, setName='train') 


def run_transform_experiment_repeats(numLabels=1000, deviceId=0,
														maxDeltaRot=5, maxDeltaTrans=3, maxRot=5, 
														numEx=None, baseLineMode=False):
	maxReps = 5	
	if numEx is None:
		numEx = 1e+06
	expDir = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/classify_rot'
	expStr = 'dRot%d_dTrn%d_mxRot%d_nLbl%.0e_numEx%.0e' % \
							(maxDeltaRot, maxDeltaTrans, maxRot, numLabels, numEx)
	modelDir     = os.path.join(expDir, expStr)
	solverPrefix = 'mnist_siamese_solver'
	defPrefix    = 'mnist_siamese_train_test'
	
	if baseLineMode:
		suffix = 'baseline'
		rootDefFile = defPrefix + '_%s.prototxt' % suffix
	else:
		suffix = None
		rootDefFile = defPrefix + '.prototxt'  
	trainExpName = 'mnist_transform_classify_%s_dRot%d_dTrn%d_mxRot%d_nLbl%.0e_numEx%.0e' \
							 % ('train', maxDeltaRot, maxDeltaTrans, maxRot, numLabels, numEx)
	testExpName = 'mnist_transform_classify_%s_dRot%d_dTrn%d_mxRot%d_nLbl%.0e_numEx%.0e' \
						 % ('test', maxDeltaRot, maxDeltaTrans, maxRot, 1e+04, 1e+04)

	for rep in range(maxReps):
		#Get the name of lmdbs
		trainIm, trainLb = get_lmdb_name(trainExpName, 'train', repNum=rep)
		testIm, testLb   = get_lmdb_name(testExpName, 'test', repNum=rep)
		#Get the definition file data
		defData          = mpu.ProtoDef(os.path.join(modelDir, rootDefFile))
		#Edit the train lmdb
		defData.set_layer_property('pair_data', ['data_param','source'], '"%s"' % trainIm, phase='TRAIN')
		defData.set_layer_property('pair_labels', ['data_param','source'],' "%s"' % trainLb, phase='TRAIN')
		#Edit the test lmdb	
		defData.set_layer_property('pair_data', ['data_param','source'], '"%s"' % testIm, phase='TEST')
		defData.set_layer_property('pair_labels', ['data_param','source'],' "%s"' % testLb, phase='TEST')

		trainExp, testExp = mpu.make_experiment_repeats(modelDir, defPrefix, solverPrefix=solverPrefix,
														repNum=rep, deviceId=deviceId, suffix=suffix, defData=defData,
														testIterations=100, modelIterations=50000)
		trainExp.run()
		testExp.run()


def get_lmdb(setName='test', maxDeltaRot=5, maxDeltaTrans=2,
						 maxRot=5, numLabels=1000, numEx=2e+5, expName=None):
	if expName is None:
		expName = 'mnist_transform_classify_%s_dRot%d_dTrn%d_mxRot%d_nLbl%.0e_numEx%.0e' % (setName, maxDeltaRot, maxDeltaTrans, maxRot, numLabels, numEx)

	if 'same_transform' in expName:
		expName = 'same_transform_trn%d_mxRot%d' % (maxDeltaTrans, maxRot)
	
	imFile, lbFile = get_lmdb_name(expName, setName)
	if expName in ['normal', 'null_transform'] or 'same_transform' in expName:
		db     = mpio.DbReader(imFile)
	else:
		db     = mpio.DoubleDbReader((imFile, lbFile))
	return db


def vis_lmdb(prms, fig=None, isClassLbl=False, 
						 setName='test', maxVis=100):
	imDb    = prms['paths']['lmdb'][setName]['im']
	lbDb    = prms['paths']['lmdb'][setName]['lb']
	visFile = os.path.join(prms['paths']['visDir'], 'im%d_%d.png') 
	db   = mpio.DoubleDbReader((imDb, lbDb))
	#Plot
	plt.ion()
	plt.set_cmap('gray')
	if fig is None:
		fig = plt.figure()
	else:
		plt.figure(fig.number)

	for i in range(maxVis):
		data, lb = db.read_next()
		ch,h,w = data.shape
		im1      = data[0,:,:]
		im2      = data[1,:,:]
		
		if isinstance(lb, np.ndarray):
			lb       = np.squeeze(lb)
			if isClassLbl:
				lbStr = 'Class: %f, delx: %f, dely: %f, delr: %f' % (lb[0], lb[1], lb[2], lb[3])
			else:
				lbStr = 'delx: %f, dely: %f, delr: %f' % (lb[0], lb[1], lb[2])
		else:
			lbStr = 'Label: %d' % lb		
		#Plot
		plt.subplot(2,1,1)
		plt.imshow(im1)
		plt.subplot(2,1,2)
		plt.imshow(im2)
		plt.suptitle(lbStr)
		#Input command
		cmd = raw_input()
		if cmd == 'save':
			outFile1 = visFile % (i,1)
			outFile2 = visFile % (i,2)
			scm.imsave(outFile1, im1)
			scm.imsave(outFile2, im2)  			
	return fig	


def save_rotations_h5(im, N, outFile, numBins=20):
    bins = np.linspace(-180,180,numBins,endpoint=False)
    fid  = h5py.File(outFile,'w')
    imSet1 = fid.create_dataset("/images1", (N*28*28,), dtype='u1')
    imSet2 = fid.create_dataset("/images2", (N*28*28,), dtype='u1')
    labels = fid.create_dataset("/labels",  (N,),       dtype='u1')
    theta1 = fid.create_dataset("/theta1",  (N,),       dtype='f')
    theta2 = fid.create_dataset("/theta2",  (N,),       dtype='f')
    
    numIm =  len(im)
    print "Number of images: %d" % numIm 
    for i in range(0,N):
        idx = random.randint(0,numIm-1)
        imC  = im[idx]
        ang1 = random.uniform(0,90)
        ang2 = random.uniform(0,90)
        sgn1 = random.random()
        sgn2 = random.random()
        if sgn1 > 0.5:
            sgn1 = -1
        else:
            sgn1 = 1
        if sgn2 > 0.5:
            sgn2 = -1
        else:
            sgn2 = 1

        ang1 = ang1 * sgn1
        ang2 = ang2 * sgn2
        im1  = scm.imrotate(imC, ang1)
        im2  = scm.imrotate(imC, ang2)
        theta = ang2 - ang1
        theta = np.where((bins >= theta) == True)[0]
        if len(theta)==0:
            theta = numBins - 1
        else:
            theta = theta[0]
        st = i*28*28
        en = st + 28*28
        imSet1[st:en] = im1.flatten()
        imSet2[st:en] = im2.flatten()
        labels[i] = theta
        theta1[i] = ang1
        theta2[i] = ang2

    fid.close()


def check_hdf5(fileName):
    f = h5py.File(fileName,'r')
    im1 = f['images1']
    im2 = f['images2']
    lbl = f['labels']
    theta1 = f['theta1']
    theta2 = f['theta2']

    nr = 28
    nc = 28

    numIm = int(im1.shape[0]/784)
    for i in range(0,10):
        idx = random.randint(0,numIm-1)
        st  = idx*nr*nc
        en  = st + nr*nc
        imA = im1[st:en]
        imB = im2[st:en]
        label = lbl[idx]

        plt.figure()
        plt.subplot(211)
        plt.title("Theta1: %f, Theta2: %f, Label: %d" % (theta1[idx], theta2[idx],label))
        plt.imshow(imA.reshape(nr,nc))
        plt.subplot(212)
        plt.imshow(imB.reshape(nr,nc))
        plt.savefig('tmp/%d.png' % i, bbox_inches='tight')

    f.close()

if __name__ == "__main__":
		trainDigits = [2,4,6,7,8,9]
		valDigits   = [0,1,3,5]
		#trainDigits = [0,1,2,3,4,5,6,7,8,9]
		#valDigits = [0,1,2,3,4,5,6,7,8,9]
		numTrain    = int(1e+5)
		numVal      = int(1e+4)
		if len(sys.argv) > 1:
			dirName = sys.argv[1]
			if not os.path.exists(dirName):
				os.makedirs(dirName)
		else:
			dirName = '/data1/pulkitag/mnist_rotation/'

		trainStr = ''
		valStr = ''
		for t in trainDigits:
			trainStr = trainStr + '%d_' % t
		for v in valDigits:
			valStr = valStr + '%d_' % v
	
		trainFile = 'mnist_train_%s%dK.hdf5' % (trainStr, int(numTrain/1000))
		valFile   = 'mnist_val_%s%dK.hdf5' % (valStr, int(numVal/1000))
		trainFile = dirName + trainFile
		valFile   = dirName + valFile

		isCreate = True
		if isCreate:
				#Get the data
				im,label    = load_images()
				trainIm = [im[i] for i in range(len(label)) if label[i] in trainDigits]
				valIm   = [im[i] for i in range(len(label)) if label[i] in valDigits]
				save_rotations_h5(trainIm, numTrain, trainFile)
				save_rotations_h5(valIm, numVal, valFile)
		else: 
				check_hdf5(valFile)

    
   