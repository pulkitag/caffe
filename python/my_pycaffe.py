import numpy as np
import h5py
import caffe

def init_network(modelFile, defFile, isGPU=True):
	net = caffe.Net(modelFile, defFile)
	net.set_phase_test()
	if isGPU:
		net.set_mode_gpu()
	else:
		net.set_mode_cpu()
	return net	


def get_batchsz(net):
	return net.blobs['data'].num


def prepare_image(im, cropSz=[], imMean=[]):
	#Only take the central crop
	shp = im.shape
	w,h = shp[-1],shp[-2]
	
	if np.ndim(im) == 3:
		im = im.reshape(shp[0],1,h,w)
	else:
		print "Incorrect image dimensions"
		raise

	if not cropSz==[]:
		assert cropSz <= w and cropSz <= h
		wSt = int((w - cropSz)/2.0)
		wEn = wSt + cropSz
		hSt = int((h - cropSz)/2.0)
		hEn = hSt + cropSz
		im  = im[:,:,hSt:hEn,wSt:wEn]

	if not imMean==[]:
		assert imMean.ndim ==4
		assert imMean.shape[0] == 1
		im = im - imMean


def get_features(net, im, layerName=None):
	dataBlob = net.blobs['data']
	batchSz  = dataBlob.num
	assert im.ndim == 4
	N,nc,h,w = im.shape
	assert h == dataBlob.height and w==dataBlob.width

	if not layerName==None:
		assert layerName in net.blobs.keys()
		layerName = [layerName]
		outName   = layerName[0]	
	else:
		outName   = net.outputs[0]

	print layerName
	imBatch = np.zeros((batchSz,nc,h,w))
	outFeats = {}
	outBlob  = net.blobs[outName]
	outFeats = np.zeros((N, outBlob.channels, outBlob.height, outBlob.width))
	for i in range(0,N,batchSz):
		st = i
		en = min(N, st + batchSz)
		l  = en - st
		imBatch[0:l,:,:,:] = np.copy(im[st:en])
		feats = net.forward(blobs=layerName, start=None, end=None, data=imBatch)		 
		outFeats[st:en] = feats[outName][0:l]

	return outFeats


def compute_error(gtLabels, prLabels, errType='classify'):
	N, lblSz = gtLabels.shape
	res = []
	assert prLabels.shape[0] == N and prLabels.shape[1] == lblSz
	if errType == 'classify':
		assert lblSz == 1
		cls     = np.unique(gtLabels)
		cls     = np.sort(cls)
		nCl     = cls.shape[0]
		confMat = np.zeros((nCl, nCl)) 
		for i in range(nCl):
			for j in range(nCl):
				confMat[i,j] = float(np.sum(np.bitwise_and((gtLabels == cls[i]),(prLabels == cls[j]))))/(np.sum(gtLabels == cls[i]))
		res = confMat
	else:
		print "Error type not recognized"
		raise
	return res	


def test_network_siamese_h5(imH5File, lbH5File, netFile, defFile, imSz=128, cropSz=112, nCh=3,  meanFile=[]):
	print imH5File, lbH5File
	imFid = h5py.File(imH5File,'r')
	lbFid = h5py.File(lbH5File,'r')
	ims1 = imFid['images1/']
	ims2 = imFid['images2/']
	lbls = lbFid['labels/']

	#Initialize network
	net  = init_network(netFile, defFile)

	#Get Sizes
	imSzSq = imSz * imSz
	assert(ims1.shape[0] % imSzSq == 0 and ims2.shape[0] % imSzSq ==0)
	N     = ims1.shape[0]/(imSzSq * nCh)
	assert(lbls.shape[0] % N == 0)
	lblSz = lbls.shape[0] / N

	#Initialize variables
	batchSz  = get_batchsz(net)
	ims      = np.zeros((batchSz, 2 * nCh, imSz, imSz))
	labels   = np.zeros((N, lblSz))
	gtLabels = np.zeros((N, lblSz)) 
	count   = 0

	#Loop through the images
	for i in np.arange(0,N,batchSz):
		st = i * nCh * imSzSq 
		en = min(N, i + batchSz) * nCh * imSzSq
		numIm = min(N, i + batchSz) - i
		ims[0:batchSz] = 0
		ims[0:numIm,0:nCh,:,:] = ims1[st:en].reshape((numIm,nCh,imSz,imSz))
		ims[0:numIm,nCh:2*nCh,:,:] = ims2[st:en].reshape((numIm,nCh,imSz,imSz))
		ims = prepare_image(ims, cropSz, imMean)  
		predFeat = get_features(net, ims)
		labels[i : i + numIm, :]    = predFeat.reshape((numIm,lblSz))
		gtLabels[i : i + numIm, : ] = lbls[i * lblSz : (i+numIm) * lblSz] 
	
	confMat = compute_error(gtLabels, labels, 'classify')
	return confMat	
