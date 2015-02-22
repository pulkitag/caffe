import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import read_picking as rep
import scipy.misc as scm
import my_pycaffe as mp

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
			

def compute_features(netName='vgg', layerName='pool4', sqMask=True, camNum=5):
	#Get the Path Prms
	prms = get_paths(sqMask, imSz=256)

	#Setup the n/w
	modelFile, meanFile = mp.get_model_mean_file(netName)
	defFile             = mp.get_layer_def_files(netName, layerName=layerName)
	meanDat             = mp.read_mean(meanFile)
	net                 = mp.init_network(defFile, modelFile)
	ipShape             = mp.get_input_blob_shape(defFile) 
	mp.net_preprocess_init(net, layerName='data', meanDat=meanDat)  	

	#Get the size of features per image
	_,ch,h,w = mp.get_blob_shape(net, layerName)
	
	#Compute the features
	cls = rep.get_classNames()
	for cl in cls:
		#Output Feature File Name 
		featFile = prms['featFile'] % (cl, netName, camNum, layerName) 
		print featFile
		dirName  = os.path.dirname(featFile)
		if not os.path.exists(dirName):
			os.makedirs(dirName)	
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
			print ipDat.shape

			#Process the batch images
			ims   = mp.preprocess_batch(net, ipDat, dataLayerName='data')
			feats = net.forward_all(blobs=[layerName], data=ims)
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
