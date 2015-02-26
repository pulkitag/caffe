import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import read_picking as rep
import scipy.misc as scm
import my_pycaffe as mp
import scipy.linalg as scl

rootDir = '/data1/pulkitag/projRotate/'


def get_paths(sqMask, imSz=256):
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
	prms['featFile'] = os.path.join(rootDir, 'features', '%s', suffix, '%s', 'cam%d', '%s.h5') 
	return prms	


def save_masks(camNum, sqMask=False):
	imSz  = 256
	if sqMask:
		svDir = os.path.join(rootDir, 'object_masks', 'imSq%d' % imSz)
	else:
		svDir = os.path.join(rootDir, 'object_masks', 'im%d' % imSz)
	cls   = rep.get_classNames()
	for cl in cls:
		print 'Processing Class: %s' % cl 
		obj   = rep.PickDat(cl)
		clDir = os.path.join(svDir, cl) 
		if not os.path.isdir(clDir):
			os.makedirs(clDir)	
		for rot in range(0,360,3):
			#Read and segment the image
			obj.read(camNum, rot)
			colDat = obj.colSeg(sqCrop=sqMask)
			colDat = scm.imresize(colDat, (imSz, imSz))
			#Save the outout
			baseName = os.path.basename(obj.colFile % (camNum, rot))
			svName   = os.path.join(clDir, baseName)
			plt.imsave(svName, colDat)
			


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
	myNet.set_preprocess(ipName='data', chSwap=(2,1,0), meanDat=meanDat, imageDims=(256, 256,3))  	

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
