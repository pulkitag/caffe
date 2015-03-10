import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import read_picking as rep
import scipy.misc as scm
import my_pycaffe as mp
import my_pycaffe_io as mpio
import scipy.linalg as scl
import pdb

rootDir = '/data1/pulkitag/projRotate/'
h5Dir   = '/data1/pulkitag/keypoints/h5/'
ldbDir  = '/data1/pulkitag/keypoints/leveldb_store/'

def get_train_test_class(trainPercent=0.7):
	cls = rep.get_classNames()
	np.random.seed(7)
	classIdx = np.random.permutation(len(cls))
	N        = int(trainPercent * len(cls))
	trainIdx = classIdx[0:N]
	testIdx  = classIdx[N:]
	return trainIdx, testIdx	


def get_paths(sqMask, imSz=256, noBg=False):
	if noBg:
		if sqMask:
			suffix = 'imNoBgSq%d' % imSz
		else:
			suffix = 'imNoBg%d' % imSz 
	else:
		if sqMask:
			suffix = 'imSq%d' % imSz
		else:
			suffix = 'im%d' % imSz 
	#Get the naming convention for storing these files.
	cls      = rep.get_classNames() 
	obj      = rep.PickDat(cls[0])
	baseName = os.path.basename(obj.colFile)
	baseName = baseName[0:-3] #baseName has two variables to be substituted (camNum, rot)
	prms = {}
	#Need to substitute - className, camNum, rot
	prms['maskFile'] = os.path.join(rootDir, 'object_masks', suffix, '%s', baseName + 'jpg')
	#Need to substitute - netName, className, camNum, layerName
	prms['featFile'] = os.path.join(rootDir, 'features', '%s', suffix, '%s', 'cam%d', '%s_new.h5') 
	#H5 File
	prms['imH5'] = os.path.join(h5Dir, '%s_images_exp%s_tp%.2f_imSz%d.h5')
	prms['lbH5'] = os.path.join(h5Dir, '%s_labels_exp%s_tp%.2f_imSz%d.h5')
	#LevelDB File
	prms['imldb'] = os.path.join(ldbDir, '%s_images_exp%s_tp%.2f_imSz%d-lmdb')
	prms['lbldb'] = os.path.join(ldbDir, '%s_labels_exp%s_tp%.2f_imSz%d-lmdb')
	return prms	


def save_masks(camNum, sqMask=False, imSz=256, noBg=False):
	prms     = get_paths(sqMask, imSz=imSz, noBg=noBg)
	maskFile = prms['maskFile']
	svDir    = os.path.dirname(maskFile) 
	cls      = rep.get_classNames()
	for cl in cls:
		print 'Processing Class: %s' % cl 
		obj   = rep.PickDat(cl)
		clDir = os.path.join(svDir, cl) 
		if not os.path.isdir(clDir):
			os.makedirs(clDir)	
		for rot in range(0,360,3):
			#Read and segment the image
			obj.read(camNum, rot)
			colDat = obj.colSeg(sqCrop=sqMask, noBg=noBg)
			colDat = scm.imresize(colDat, (imSz, imSz))
			#Save the outout
			baseName = os.path.basename(obj.colFile % (camNum, rot))
			svName   = os.path.join(clDir, baseName)
			plt.imsave(svName, colDat)
			

def get_polar_data(isTrain=True):
	'''
		Provides pairs of images such that the pair of images is only differentiated by the polar angle
		of the camera. We have 120 images per polar angle and there are a total of 5 polar angles. 
	'''
	clNames         = rep.get_classNames()
	tp              = 0.7
	trainCl, testCl = get_train_test_class(trainPercent=tp)
	expName         = 'picking_polar'
	if isTrain:
		clNames = [clNames[i] for i in trainCl]
		setStr  = 'train'
	else:
		clNames = [clNames[i] for i in testCl]  
		setStr  = 'test'

	numSamples = 5 * 5 * 120
	h, w, ch   = 256, 256, 3 
	N          = h * w * ch
	prms       = get_paths(sqMask=True, imSz=h)	
	imFile     = prms['maskFile'] 

	#Initialize the dataspaces
	dbName = prms['imldb'] % (setStr, expName, tp, h)
	db     = mpio.dbSaver(dbName)   
	
	#Permutation for saving the images
	perm    = np.random.permutation(numSamples * len(clNames))
	
	count  = 0
	for cls in clNames:
		for cm1 in range(1,6):
			for cm2 in range(1,6):
				for rot in range(0,360,3):
					imF1 = imFile % (cls, cm1, rot)
					imF2 = imFile % (cls, cm2, rot)
					im1  = plt.imread(imF1)
					im2  = plt.imread(imF2)
					idx  = perm[count]
					im1  = im1[:,:,[2,1,0]].transpose((2,0,1))	
					im2  = im2[:,:,[2,1,0]].transpose((2,0,1))
					imC  = (np.concatenate((im1, im2))).reshape(1, 2*ch, h, w)
					lb   = int(cm2 - cm1 + 5) #to make the range 0 to number of images				
					db.add_batch(imC, labels=np.array([lb]), svIdx=[idx])	
					count = count + 1
	db.close()


def get_polar_azimuth_data(isTrain=True, azimuthStep=12):
	'''
		Get the examples for polar and azimuth data.
		polar data can be from any of the -4,4 since we have 5 camera. 
		rotation data can be from -45, 45 angles. 
	'''
	clNames         = rep.get_classNames()
	tp              = 0.7
	trainCl, testCl = get_train_test_class(trainPercent=tp)
	expName         = 'picking_polar_azimuth%d' % azimuthStep
	if isTrain:
		clNames = [clNames[i] for i in trainCl]
		setStr  = 'train'
		nSPerClass = 10000
	else:
		clNames = [clNames[i] for i in testCl]  
		setStr  = 'test'
		nSPerClass = 1000

	h, w, ch   = 256, 256, 3 
	N          = h * w * ch
	prms       = get_paths(sqMask=True, imSz=h)	
	imFile     = prms['maskFile'] 

	rotRange = np.arange(-180/3, 180/3)
	numRots  = len(rotRange)
	mnRot    = -48/3
	mxRot    = 48/3
	mnIdx, mxIdx = np.where(rotRange==mnRot)[0][0], np.where(rotRange==mxRot)[0][0]
	
	if isTrain:
		#I want to sample in a manner that number of rotations chosen out of rotation range are equal
		#to any single elemnt in the rotation range. 
		probSample = np.zeros((numRots,))
		probSample[mnIdx:mxIdx+1] = 1.
		probSum    = np.sum(probSample)
		numOthers  = numRots - (mxIdx - mnIdx + 1)
		probOthers = 1.0/numOthers
		probSample[0:mnIdx]  = probOthers
		probSample[mxIdx+1:] = probOthers
	else:
		probSample = np.ones((numRots,))
	probSample   = probSample/sum(probSample)

	#Permutation for saving the images
	perm    = np.random.permutation(nSPerClass * len(clNames))

	#The db in which to store
	imDbName = prms['imldb'] % (setStr, expName, tp, h)
	lbDbName = prms['lbldb'] % (setStr, expName, tp, h)
	print imDbName, lbDbName
	db = mpio.DoubleDbSaver(imDbName, lbDbName)
	
	count = 0
	for cl in clNames:
		print cl
		for i in range(nSPerClass):	
			#The Polar Angle
			cm1     = np.floor(5 * min(0.99,np.random.random())) + 1 
			cm2     = np.floor(5 * min(0.99,np.random.random())) + 1 
			camLbl  = cm2 - cm1 + 4
			#The Azimuth
			rot1    = np.floor(numRots * min(0.9999, np.random.random()))
			rotDiff = np.random.choice(rotRange, p=probSample)
			rot2    = np.mod(rot1 + rotDiff, 120) 
			rotLbl  = int((rotDiff + 60) * (3.0/azimuthStep))  
			assert rotLbl >=0 and camLbl >=0, "rot:%d, cam:%d" % (rotLbl, camLbl) 

			#Get the image files and save to dbs
			imF1 = imFile % (cl, cm1, rot1*3)
			imF2 = imFile % (cl, cm2, rot2*3)
			im1  = plt.imread(imF1)
			im2  = plt.imread(imF2)
			idx  = perm[count]
			im1  = im1[:,:,[2,1,0]].transpose((2,0,1))	
			im2  = im2[:,:,[2,1,0]].transpose((2,0,1))
			imC  = (np.concatenate((im1, im2))).reshape(1, 2*ch, h, w)
			lb   = np.array([camLbl, rotLbl]).reshape(1,2,1,1)				
			db.add_batch((imC,lb), svIdx=([idx],[idx]))	
			count = count + 1
	db.close()


def test_network(expStr='polar',svImg=False, deviceId=2):
	defFile = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/keypoints/exp/picking_%s_finetune_pool5/caffenet_siamese_deploy.prototxt' % expStr
	netFile = '/data1/pulkitag/projRotate/snapshots/picking/exp%s/caffenet_pool5_finetune_siamese_iter_40000.caffemodel' % expStr
	mnFile      = '/data1/pulkitag/keypoints/leveldb_store/picking_polar_im256_mean.binaryproto'
	svImFileStr = '/data1/pulkitag/projRotate/vis/exp%s/%s/im%s.png' % (expStr, '%s', '%d') 
	tp       = 0.7
	imSz     = 256
	expName  = 'picking_%s' % expStr
	prms     = get_paths(sqMask=True, imSz=imSz)	
	#Get the LMDB
	dbName = prms['imldb'] % ('test', expName, tp, imSz)
	db     = mpio.DbReader(dbName)   
	confMat = mp.test_network_siamese_h5(netFile=netFile, defFile=defFile, meanFile=mnFile, imSz=imSz, ipLayerName='pair_data', outFeatSz=10, db=db, svImg=svImg, svImFileStr=svImFileStr, deviceId=deviceId) 		

	return confMat


def compute_distance(x1, x2, distType='l2'):
	if distType=='l2':
		return scl.norm(x1-x2)
	elif distType=='l2norm':
		n1   = scl.norm(x1)
		n2   = scl.norm(x2)
		dist = scl.norm(x1 - x2)
		return (dist/(np.sqrt(n1*n2))) 
	else:
		assert False, 'DistType not found'


def compute_feat_diff(netName='vgg', layerName='pool4', sqMask=True, camNum=5, imSz=256):
	'''
		Assumes that features have been precomputed.
		Based on these features compute the differences. 
	'''
	#The number of neighbors to consider for computing the distance
	numNgbhr = 10
	rots     = range(0,360,3)
	N        = len(rots)

	#Get the paths
	prms = get_paths(sqMask, imSz=imSz)
	#ClassNames
	cls     = rep.get_classNames()
	rotDiff = np.zeros((len(cls),N,2*numNgbhr-1)) 
	for (i,cl) in enumerate(cls):
		print cl
		#Output Feature File Name 
		featFile = prms['featFile'] % (cl, netName, camNum, layerName) 
		featFid  = h5.File(featFile, 'r')
		for (r,rot) in enumerate(rots):
			gtFeat = (featFid['rot%d' % rot][:]).flatten()
			count  = 0
			for j in range(-numNgbhr + 1, numNgbhr):
				r2   = np.mod(r + j, N)
				feat = (featFid['rot%d' % rots[r2]][:]).flatten()
				dist = compute_distance(gtFeat, feat, 'l2norm') 	
				rotDiff[i,r,count] = dist
				count += 1

	return rotDiff


def compute_feat_diff_outplane(netName='vgg', rotNum=0, layerName='pool4', sqMask=True, imSz=256):
	camNum = range(1,5)
	#Get the paths
	prms = get_paths(sqMask, imSz=imSz)
	#ClassNames
	cls     = rep.get_classNames()
	
	rotDiff = np.zeros((len(cls),len(camNum))) 
	for (i,cl) in enumerate(cls):
		print cl
		#Output Feature File Name
		for c in camNum: 
			#Load the features
			featFile1 = prms['featFile'] % (cl, netName, 1, layerName) 
			featFile2 = prms['featFile'] % (cl, netName, c + 1, layerName) 
			print featFile1, featFile2
			featFid1  = h5.File(featFile1, 'r')
			featFid2  = h5.File(featFile2, 'r')
			#Compute the Diff
			gtFeat = (featFid1['rot%d' % rotNum][:]).flatten()
			feat   = (featFid2['rot%d' % rotNum][:]).flatten()
			dist = compute_distance(gtFeat, feat, 'l2norm') 	
			rotDiff[i,c-1] = dist

	return rotDiff


def compute_features(netName='vgg', layerName='pool4', sqMask=True, camNum=5):
	#Get the Path Prms
	prms = get_paths(sqMask, imSz=256)

	#Setup the n/w
	modelFile, meanFile = mp.get_model_mean_file(netName)
	defFile             = mp.get_layer_def_files(netName, layerName=layerName)
	meanDat             = mp.read_mean(meanFile)
	myNet               = mp.MyNet(defFile, modelFile)
	ipShape             = mp.get_input_blob_shape(defFile) 
	myNet.set_preprocess(ipName='data', chSwap=(2,1,0), meanDat=meanDat, imageDims=(256, 256, 3))  	

	#Get the size of features per image
	_,ch,h,w = myNet.get_blob_shape(layerName)
	
	#Compute the features
	cls = rep.get_classNames()
	for cl in cls:
		#Output Feature File Name 
		featFile = prms['featFile'] % (cl, netName, camNum, layerName) 
		print featFile
		dirName  = os.path.dirname(featFile)
		if not os.path.exists(dirName):
			os.makedirs(dirName)
		if os.path.exists(featFile):
			os.remove(featFile)
		featFid  = h5.File(featFile, 'w')
	
		print 'Computing for class: %s' % cl
		batchSz = ipShape[0] 
		rotNum  = range(0,360,3)
		count   = 0
		compFlag = True
		while compFlag:
			#Get the batch start and end
			st = count
			en = min(len(rotNum), count + batchSz)
			
			#Get the batch images
			ipDat = []
			for rot in rotNum[st:en]:
				ipFile = prms['maskFile'] % (cl, camNum, rot)
				im     = plt.imread(ipFile)
				im     = np.reshape(im, (1,im.shape[0],im.shape[1],im.shape[2]))
				ipDat.append(im)
			ipDat = np.concatenate(ipDat[:], axis=0)

			#Process the batch images
			ims   = myNet.preprocess_batch(ipDat, ipName='data')
			feats = myNet.net.forward_all(blobs=[layerName], data=ims)
			feats = feats[layerName]

			#Save The Features
			for (i,rot) in enumerate(rotNum[st:en]):
				datName = 'rot%d' % rot
				dset    =  featFid.create_dataset(datName, (ch,h,w))
				dset[:,:,:] = feats[i]  
		
			#Termination Condition
			if st + batchSz >= len(rotNum):
				compFlag = False
			count += batchSz
		featFid.close() 
