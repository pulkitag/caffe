import scipy.io as sio
from scipy import linalg as linalg
import scipy
import sys, os
import h5py
import numpy as np
import pdb
import subprocess
import rot_utils as ru

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
	'''
	The data file will store the examples sequentially. However, we want to present the examples in a 
	randomized manner to the ConvNet. Therefore we define a tuple:
	(pos,cl1Idx,cl2Idx) - which says that example cl1Idx and cl2Idx from class cl were used to generate
	the data for example at position pos. 
	'''
	

	#Find number of valid samples
	clCount = [len(c) for c in clIdx]
	clCount = normalize_counts(clCount, numSamples)

	#For randomizing the order of examples across the classes. 
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
	numTest  = 1e+5
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
		#Hold out some classes to test generalization on those classes.
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
	testIdx    = get_indexes(valImIdx, numTest)

	#Image data
	trainIms = [ims[classNames.index(cl)] for cl in classNames if cl in trainClass] 
	valIms   = [ims[classNames.index(cl)] for cl in classNames if cl in valClass] 
	testIms  = [ims[classNames.index(cl)] for cl in classNames if cl in valClass]
	
	#view data
	trainView = [view[classNames.index(cl)] for cl in classNames if cl in trainClass] 
	valView   = [view[classNames.index(cl)] for cl in classNames if cl in valClass] 
	testView  = [view[classNames.index(cl)] for cl in classNames if cl in valClass] 
	
	return (trainIdx, trainIms, trainView), (valIdx, valIms, valView), (testIdx, testIms, testView)


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


def find_view_centers(views, idxs, numCenters, maxAngle=30):
	numSample  = 50000
	nSPerClass = int(numSample/len(idxs))
	viewDiff   = np.zeros((numSample,3,3))
	maxAngle   = (np.pi*maxAngle)/180.

	totalCount = 0
	for cl in range(len(idxs)):
		#Get views for a class
		clViews = views[cl]
		N       = len(clViews)
		cl1Sample = np.random.random_integers(0, N-1, 20*nSPerClass)
		cl2Sample = np.random.random_integers(0, N-1, 20*nSPerClass)
		count     = 0
		for i in range(20*nSPerClass):
			view1,view2 = clViews[cl1Sample[i]], clViews[cl2Sample[i]]
			angle      = get_rot_angle(view1, view2)	
			if angle < maxAngle:
				viewDiff[totalCount,:,:] = np.dot(view2, np.transpose(view1)) 
				count             += 1
				totalCount        += 1
				if count >= nSPerClass:
					break

	#Find the K-Means centers
	print "Finding Centers of Rotation Matrices"
	assgn,centers = ru.cluster_rotmats(viewDiff[0:totalCount],numCenters)

	return centers
	


def create_data_labels(h5LabelFile, views, idxs, labelType, viewCenters=[]):
	numSamples = [len(i) for i in idxs]
	numSamples = sum(numSamples)

	#Open the h5 file for writing labels
	fidl  = h5py.File(h5LabelFile, 'w')

	#Select the type of label
	if labelType == '9DRot':
		labelSz = 9
	elif labelType == 'angle':
		labelSz = 1
	elif labelType == 'uniform20':
		labelSz = 1
		angRange = np.linspace(0,np.pi,20,endpoint=False)
	elif labelType == 'limited30_3':
		#Divide zero to 30 in 3 bins and the fourth bin is for the rest
		labelSz = 1
		angRange = np.zeros((5,1))
		angRange[0:4,0] =(np.linspace(0,30,4)).flatten()
		angRange[4] = 180.
		angRange    = np.pi*(angRange/180.)
		print angRange
	elif labelType =='kmedoids30_20':
		#If views are within 30 degree rotations then assign to one of the rotation centers,
		#otherwise regard them as outsider. 
		labelSz   = 1
		numLabels = 21
		angRange = np.zeros((2,1))
		angRange[0] = 30
		angRange[1] = 180
		angRange    = np.pi*(angRange/180.)
		if len(viewCenters)==0:
			#Need to find the medoids/centers.
			viewCenters = find_view_centers(views, idxs, numLabels-1)
	else:
		print "Unrecognized labelType "
		raise("Label Type not found exception")

	#Create two datasets in the labels file:
	#labels:  store the actual label information. 
	#indices: for each label store the idx --> the position where the example is mapped to in the h5File(idx), 
	#         the object class from which the image comes (cl),
	#         the index within the examples of the class (clIdx1, clIdx2) - the rotation
	#         between which has been encoded by the label. 
	labels   = fidl.create_dataset("/labels",(numSamples * labelSz,), dtype='f')
	indices  = fidl.create_dataset("/indices",(numSamples * 4,), dtype='f')
	count    = 0
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
			elif labelType in ['uniform20', 'limited30_3']:
				angle    = get_rot_angle(view1, view2)			
				try:
					viewDiff = np.where(angle >= angRange)[0][-1] 
				except:
					print "Error encountered"
					pdb.set_trace()			
			elif labelType =='kmedoids30_20':
				viewDiff = np.dot(view2, np.linalg.inv(view1))
				viewDiff = ru.get_cluster_assignments(viewDiff, viewCenters)	
	
			labels[lSt:lEn] = viewDiff.flatten()
			indices[count:count+4] = idx,cl,clIdx1,clIdx2
			count = count + 4
	fidl.close()	
	return viewCenters


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


def get_view_centers_name(exp, lblType):
	vName = 'view_centers_exp%s_lbl%s' % (exp, lblType)
	vName = h5Dir + vName
	return vName


def save_view_centers(dat, vFile):
	N,nr,nc = dat.shape
	assert nr==3 and nc==3
	sz = nr * nc
	
	fid  = h5py.File(vFile, 'w')
	vcs   = fid.create_dataset("/vcs",(N * sz,), dtype='f')

	#Save the data
	for i in range(N):
		st = i*sz
		en = st + sz
		vcs[st:en] = dat[i].flatten()
	fid.close()


def read_view_centers(vFile):
	fid  = h5py.File(vFile, 'r')
	vcs  = fid['vcs']
	N    = vcs.shape[0]
	assert np.mod(N,9)==0
	N    = N/9

	dat  = np.zeros((N,3,3))
	for i in range(N):
		centers = vcs[i*9:(i+1)*9]
		dat[i]  = centers.reshape((3,3))

	fid.close()
	return dat


def h52db(exp, labelType, imSz, lblOnly=False):
	imToolName = TOOL_DIR + 'hdf52leveldb_siamese_nolabels.bin'
	lbToolName = TOOL_DIR + 'hdf52leveldb_float_labels.bin'
	splits = ['val','train']
	if labelType=='9DRot':
		labelSz = 9
	elif labelType in ['angle','uniform20','limited30_3','kmedoids30_20']:
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
	labelType = 'kmedoids30_20' 
	
	trainDetails, valDetails, testDetails = get_experiment_details(exp, imSz)

	#Stuff for viewCenters
	viewCenters = []
	if labelType in ['kmedoids30_20']:
		vcFileName = get_view_centers_name(exp, labelType)
		if os.path.exists(vcFileName):
			print "Loading view centers from a saved file"
			viewCenters = read_view_centers(vcFileName)

	isTrain = True
	isVal   = False
	isTest  = True

	if isTrain:	
		print "Making training Data .."
		trainDataH5  = get_imH5Name('train', exp, imSz)
		trainLabelH5 = get_lblH5Name('train', exp, imSz, labelType)
		trainIdxs, trainIms, trainViews = trainDetails
		#create_data_images(trainDataH5, trainIms, trainIdxs, imSz)
		viewCenters = create_data_labels(trainLabelH5, trainViews, trainIdxs, labelType, viewCenters)
		if labelType in ['kmedoids30_20']:
			save_view_centers(viewCenters, vcFileName)

	if isVal:
		print "Making Validation Data.."
		valDataH5   = get_imH5Name('val', exp, imSz)
		valLabelH5  = get_lblH5Name('val', exp, imSz, labelType)
		valIdxs, valIms, valViews = valDetails
		create_data_images(valDataH5, valIms, valIdxs, imSz)
		create_data_labels(valLabelH5, valViews, valIdxs, labelType, viewCenters)

	if isTest:
		print "Making Test Data .."
		testDataH5  = get_imH5Name('test', exp, imSz)
		testLabelH5  = get_lblH5Name('test', exp, imSz, labelType)
		testIdxs, testIms, testViews = testDetails
		create_data_images(testDataH5, testIms, testIdxs, imSz)
		create_data_labels(testLabelH5, testViews, testIdxs, labelType, viewCenters)

