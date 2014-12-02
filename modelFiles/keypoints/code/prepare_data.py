import scipy.io as sio
from scipy import linalg as linalg
import scipy
import sys, os
import h5py
import numpy as np
import pdb
import subprocess

TOOL_DIR = '/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools/'

rawDir = '/data1/pulkitag/keypoints/raw/imSz%d/';
h5Dir  = '/data1/pulkitag/keypoints/h5/';
dbDir  = '/data1/pulkitag/keypoints/leveldb_store/';
classNames = ['aeroplane','bicycle','bird','boat','bottle',\
							'bus','car','cat','chair','cow', \
							'diningtable','dog','horse','motorbike','person', \
							'potted-plant','sheep','sofa','train','tvmonitor']

rigidClass = ['aeroplane', 'bicycle', 'boat', 'bottle', \
							'bus', 'car', 'chair', 'diningtable', 'motorbike', \
							'potted-plant', 'sofa', 'train', 'tvmonitor']
							
def normalize_counts(clCount, numSamples):
	countSum = sum(clCount)
	clCount  = [int((numSamples * c / countSum) + 1) for c in clCount]	
	return clCount


def get_indexes(clIdx, numSamples):
	#Find number of valid samples
	clCount = [len(c) for c in clIdx]
	clCount = normalize_counts(clCount, numSamples)

	#For randomizing the order of classes
	np.random.seed(3)
	idxs = np.random.permutation(sum(clCount)) 

	stIdx = 0
	outIdx = []
	for (i,c) in enumerate(clIdx):
		cl1Sample = np.random.random_integers(0, len(c)-1, clCount[i])
		cl2Sample = np.random.random_integers(0, len(c)-1, clCount[i])
		print "For class %d, obtaining %d examples" % (i, clCount[i]) 	
	
		enIdx    = stIdx + clCount[i] 	
		cl1Idx = [c[k] for k in cl1Sample]
		cl2Idx = [c[k] for k in cl2Sample]
		tupleIdx = [(j,k1,k2) for (j,k1,k2) in zip(idxs[stIdx:enIdx],cl1Idx,cl2Idx)]
		outIdx.append(tupleIdx)

		stIdx = stIdx + clCount[i]
	return outIdx		


def get_experiment_details(expName, imSz=256):
	
	numTrain = 1e+6
	numVal   = 5e+4
	global rawDir
	rawDir = rawDir % imSz
	ims  = []
	view = []
	clCount = []

	#Load the raw data
	for cl in classNames:
		fileName = rawDir + cl + '.mat'
		data     = h5py.File(fileName)
		ims.append(np.transpose(data['ims'], axes=(3,2,1,0)))
		view.append(np.transpose(data['view'], axes=(2,1,0)))
		clCount.append(int(np.array(data['clCount'])[0][0]))

	#Get the class paritioning
	if expName=='generalize':
		trainClass = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',\
								'car','cat','chair','cow','diningtable','dog', \
								'person','sheep','sofa','train']
		valClass = [cl for cl in classNames if cl not in trainClass]
		trainImIdx, valImIdx = [],[]
		for (i,cl) in enumerate(classNames):
			if cl in trainClass:
				trainImIdx.append(range(clCount[i]))
			else:
				valImIdx.append(range(clCount[i])) 

	elif expName=='rigid':
		trainClass = rigidClass
		valClass   = trainClass
		#Select training and val images for each class
		trainPercent = .80 
		valPercent   = .20
		trainImIdx, valImIdx   = [],[]
		np.random.seed(7)
		for cl in rigidClass:
			cc = clCount[classNames.index(cl)]
			classIdx = np.random.permutation(cc)
			N        = int(trainPercent * cc)
			trainImIdx.append(classIdx[0:N])
			valImIdx.append(classIdx[N:-1])

	#Get the indexes in the form: (idx, pair1, pair2)
	trainIdx   = get_indexes(trainImIdx, numTrain)
	valIdx     = get_indexes(valImIdx, numVal)

	#Image and view data
	trainIms = [ims[classNames.index(cl)] for cl in classNames if cl in trainClass] 
	valIms   = [ims[classNames.index(cl)] for cl in classNames if cl in valClass] 
	trainView = [view[classNames.index(cl)] for cl in classNames if cl in trainClass] 
	valView   = [view[classNames.index(cl)] for cl in classNames if cl in valClass] 

	return (trainIdx, trainIms, trainView), (valIdx, valIms, valView)


def create_data_images(h5DataFile, ims, idxs, imSz):	
	h,w,ch = imSz,imSz,3
	imSz   = h * w * ch
	numSamples = [len(i) for i in idxs]
	numSamples = sum(numSamples)

	fid   = h5py.File(h5DataFile, 'w')
	images1 = fid.create_dataset("/images1", (numSamples * imSz,), dtype='u1')
	images2 = fid.create_dataset("/images2", (numSamples * imSz,), dtype='u1')

	for cl in range(len(idxs)):
		outIdx, im  = idxs[cl], ims[cl]
		for k in range(len(outIdx)):
			idx,clIdx1,clIdx2 = outIdx[k] 
			st  = idx * imSz 
			en  = st + imSz
			try:
				images1[st:en] = im[clIdx1].transpose((2,0,1)).flatten()
				images2[st:en] = im[clIdx2].transpose((2,0,1)).flatten()
			except ValueError:
				print "Error Encountered"
				pdb.set_trace()	
		print "Saved: %d examples for class %d" % (len(outIdx),cl)	
	fid.close()


def get_rot_angle(view1, view2):
	viewDiff = linalg.logm(np.dot(view2, np.transpose(view1)))
	viewDiff = linalg.norm(viewDiff, ord='fro')
	angle    = viewDiff/np.sqrt(2)
	return angle


def get_angle_counts(fileName, thetaThresh):
	''' Finds the number of angles below and above the thetaThresh ''' 

	fidl = h5py.File(fileName, 'r')
	angles = fidl['labels']
	count  = angles.shape[0]
	thetaThresh = (np.pi/180.0) * thetaThresh
	threshCount = np.sum(np.array(angles) <= thetaThresh)
	pdb.set_trace()
	print "%d number of examples below thresh, %d number above" % (threshCount, count - threshCount)		


def create_data_labels(h5LabelFile, views, idxs, labelType):
	numSamples = [len(i) for i in idxs]
	numSamples = sum(numSamples)

	fidl  = h5py.File(h5LabelFile, 'w')
	if labelType == '9DRot':
		labelSz = 9
	elif labelType == 'angle':
		labelSz = 1
	elif labelType == 'uniform20':
		labelSz = 1
		angRange = np.linspace(0,np.pi,20,endpoint=False)
	else:
		print "Unrecognized labelType "
		raise("Label Type not found exception")

	labels  = fidl.create_dataset("/labels",(numSamples * labelSz,), dtype='f')
	for cl in range(len(idxs)):
		outIdx, view = idxs[cl], views[cl]
		for k in range(len(outIdx)):
			idx,clIdx1,clIdx2 = outIdx[k] 
			lSt = idx * labelSz
			lEn = lSt + labelSz
			view1 = view[clIdx1]
			view2 = view[clIdx2]
			if labelType == '9DRot':
				viewDiff = np.dot(view2, np.linalg.inv(view1))
			elif labelType == 'angle': 
				viewDiff = get_rot_angle(view1, view2)								
			elif labelType == 'uniform20':
				angle    = get_rot_angle(view1, view2)			
				try:
					viewDiff = np.where(angle >= angRange)[0][-1] 
				except:
					print "Error encountered"
					pdb.set_trace()			
	
			labels[lSt:lEn] = viewDiff.flatten()
	fidl.close()	


def get_imH5Name(setName, exp, imSz):
	h5Name = '%s_images_exp%s_imSz%d.hdf5'%(setName, exp, imSz)
	h5Name = h5Dir + h5Name
	return h5Name


def get_lblH5Name(setName, exp, imSz, lblType):
	h5Name = '%s_labels_exp%s_lbl%s_imSz%d.hdf5'%(setName, exp, lblType, imSz)
	h5Name = h5Dir + h5Name
	return h5Name


def get_imDbName(setName, exp, imSz):
	dbName = '%s_images_exp%s_imSz%d-leveldb'%(setName, exp, imSz)
	dbName = dbDir + dbName
	return dbName


def get_lblDbName(setName, exp, imSz, lblType):
	dbName = '%s_labels_exp%s_lbl%s_imSz%d-leveldb'%(setName, exp, lblType, imSz)
	dbName = dbDir + dbName
	return dbName


def h52db(exp, labelType, imSz, lblOnly=False):
	imToolName = TOOL_DIR + 'hdf52leveldb_siamese_nolabels.bin'
	lbToolName = TOOL_DIR + 'hdf52leveldb_float_labels.bin'
	splits = ['val','train']
	if labelType=='9DRot':
		labelSz = 9
	elif labelType=='angle':
		labelSz = 1
	elif labelType=='uniform20':
		labelSz = 1
	
	for s in splits:
		if not lblOnly:
			h5ImName = get_imH5Name(s, exp, imSz)
			dbImName = get_imDbName(s, exp, imSz) 
			subprocess.check_call(['%s %s %s %d' % (imToolName, h5ImName, dbImName, imSz)],shell=True)
		h5LbName = get_lblH5Name(s, exp, imSz, labelType)
		dbLbName = get_lblDbName(s, exp, imSz, labelType)	 
		subprocess.check_call(['%s %s %s %d' % (lbToolName, h5LbName, dbLbName, labelSz)],shell=True)


if __name__ == "__main__":
	imSz      = 128
	exp       = 'rigid'
	labelType = 'uniform20' 
	
	trainDetails, valDetails = get_experiment_details(exp, imSz)
	
	print "Making training Data .."
	trainDataH5  = get_imH5Name('train', exp, imSz)
	trainLabelH5 = get_lblH5Name('train', exp, imSz, labelType)
	trainIdxs, trainIms, trainViews = trainDetails
	#create_data_images(trainDataH5, trainIms, trainIdxs, imSz)
	create_data_labels(trainLabelH5, trainViews, trainIdxs, labelType)

	print "Making Validation Data.."
	valDataH5   = get_imH5Name('val', exp, imSz)
	valLabelH5  = get_lblH5Name('val', exp, imSz, labelType)
	valIdxs, valIms, valViews = valDetails
	#create_data_images(valDataH5, valIms, valIdxs, imSz)
	create_data_labels(valLabelH5, valViews, valIdxs, labelType)

'''
OLD Code

def create_data(h5DataFile, h5LabelFile,  numSamples, ims, views, clCount, imSz):	
	h,w,ch = imSz,imSz,3
	imSz   = h * w * ch
	
	countSum = sum(clCount)
	clCount  = numSamples*(clCount/countSum) + 1
	clCount  = [int(c) for c in clCount]	
	numSamples = sum(clCount)
	print clCount
	
	fid   = h5py.File(h5DataFile, 'w')
	fidl  = h5py.File(h5LabelFile, 'w')
	images1 = fid.create_dataset("/images1", (numSamples * imSz,), dtype='u1')
	images2 = fid.create_dataset("/images2", (numSamples * imSz,), dtype='u1')
	labels  = fidl.create_dataset("/labels",(numSamples * 9,), dtype='f')

	np.random.seed(3)
	#For Randmizing the class order
	idxs = np.random.permutation(sum(clCount)) 

	count = 0
	for (i,num) in enumerate(clCount):
		#num - number of the images to be sampled from the class
		im    = ims[i]
		print im.shape
		view  = views[i]
		num   = int(num)
		clIdx1 = np.random.random_integers(0,im.shape[0]-1,num)
		clIdx2 = np.random.random_integers(0,im.shape[0]-1,num)
		print "For class %d, obtaining %d examples" % (i, num) 	
		#pdb.set_trace()
		for k in range(num):
			st  = idxs[count] * imSz 
			en  = st + imSz
			lSt = idxs[count] * 9
			lEn = lSt + 9
			try:
				images1[st:en] = im[clIdx1[k]].transpose((2,0,1)).flatten()
				images2[st:en] = im[clIdx2[k]].transpose((2,0,1)).flatten()
			except ValueError:
				print "Error Encountered"
				pdb.set_trace()		
			view1 = view[clIdx1[k]]
			view2 = view[clIdx2[k]]
			viewDiff = np.dot(view2, np.linalg.inv(view1)) 								
			labels[lSt:lEn] = viewDiff.flatten()
			count += 1
	fid.close()
	fidl.close()	


if __name__ == "__main__":
	trainClass = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',\
								'car','cat','chair','cow','diningtable','dog', \
								'person','sheep','sofa','train']
	valClass =  [cl for cl in classNames if cl not in trainClass]
	print "Number of train,val classes: (%d,%d)" % (len(trainClass), len(valClass))
	numTrain = 1e+6
	numVal   = 5e+4
	imSz     = 128
	rawDir   = rawDir % imSz

	print "Loading Data ..."
	trainIm = []
	valIm   = []
	trainCount, valCount = [],[]
	trainView = []
	valView   = []
	for cl in classNames:
		fileName = rawDir + cl + '.mat'
		data     = h5py.File(fileName)
		ims      = np.transpose(data['ims'], axes=(3,2,1,0))
		view     = np.transpose(data['view'], axes=(2,1,0))
		clCount  = np.array(data['clCount'])
		if cl in trainClass:
			trainIm.append(ims)
			trainView.append(view)
			trainCount.append(clCount)
		else:
			valIm.append(ims)
			valView.append(view)
			valCount.append(clCount)

	print "Making training Data .."
	trainDataH5  = h5Dir + 'train_images_imSz%d.hdf5' % imSz
	trainLabelH5 = h5Dir + 'train_labels_imSz%d.hdf5' % imSz
	create_data(trainDataH5, trainLabelH5, numTrain, trainIm, trainView, trainCount, imSz)

	print "Making Validation Data.."
	valDataH5   = h5Dir + 'val_images_imSz%d.hdf5' % imSz
	valLabelH5  = h5Dir + 'val_labels_imSz%d.hdf5' % imSz
	create_data(valDataH5, valLabelH5,  numVal, valIm, valView, valCount, imSz)

'''	
