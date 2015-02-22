import numpy as np
import h5py
import caffe
import pdb


class layerSz:
	def __init__(self, stride, filterSz):
		self.imSzPrev = [] #We will assume square images for now
		self.stride   = stride #Stride with which filters are applied
		self.filterSz = filterSz #Size of filters. 
		self.stridePixPrev = [] #Stride in image pixels of the filters in the previous layers.
		self.pixelSzPrev   = [] #Size of the filters in the previous layers in the image space
		#To be computed
		self.pixelSz   = [] #the receptive field size of the filter in the original image.
		self.stridePix = [] #Stride of units in the image pixel domain.

	def prev_prms(self, prevLayer):
		self.set_prev_prms(prevLayer.stridePix, prevLayer.pixelSz)

	def set_prev_prms(self, stridePixPrev, pixelSzPrev):
		self.stridePixPrev = stridePixPrev
		self.pixelSzPrev   = pixelSzPrev

	def compute(self):
		self.pixelSz   = self.pixelSzPrev + (self.filterSz-1)*self.stridePixPrev	  
		self.stridePix = self.stride * self.stridePixPrev


def calculate_size():
	'''
		Calculate the receptive field size and the stride of the Alex-Net
	'''
	conv1 = layerSz(4,11)
	conv1.set_prev_prms(1,1)
	conv1.compute()
	pool1 = layerSz(2,3)
	pool1.prev_prms(conv1)
	pool1.compute()

	conv2 = layerSz(1,5)
	conv2.prev_prms(pool1)
	conv2.compute()
	pool2 = layerSz(2,3)
	pool2.prev_prms(conv2)
	pool2.compute()

	conv3 = layerSz(1,3)
	conv3.prev_prms(pool2)
	conv3.compute()

	conv4 = layerSz(1,3)
	conv4.prev_prms(conv3)
	conv4.compute()

	conv5 = layerSz(1,3)
	conv5.prev_prms(conv4)
	conv5.compute()
	pool5 = layerSz(2,3)
	pool5.prev_prms(conv5)
	pool5.compute()

	print 'Pool1: Receptive: %d, Stride: %d ' % (pool1.pixelSz, pool1.stridePix)	
	print 'Pool2: Receptive: %d, Stride: %d ' % (pool2.pixelSz, pool2.stridePix)	
	print 'Conv3: Receptive: %d, Stride: %d ' % (conv3.pixelSz, conv3.stridePix)	
	print 'Conv4: Receptive: %d, Stride: %d ' % (conv4.pixelSz, conv4.stridePix)	
	print 'Pool5: Receptive: %d, Stride: %d ' % (pool5.pixelSz, pool5.stridePix)	
	

def get_model_mean_file(netName='vgg'):
	'''
		Returns 
			the model file - the .caffemodel with the weights
			the mean file of the imagenet data
	'''
	if netName   == 'alex':
		modelFile    = '/data1/pulkitag/caffe_models/caffe_imagenet_train_iter_310000'
	elif netName == 'vgg':
		modelFile    = '/data1/pulkitag/caffe_models/VGG_ILSVRC_19_layers.caffemodel'
	else:
		print 'netName not recognized'
		return

	imMeanFile = '/data1/pulkitag/caffe_models/ilsvrc2012_mean.binaryproto'
	return modelFile, imMeanFile
	

def get_layer_def_files(netName='vgg', layerName='pool4'):
	'''
		Returns
			the architecture definition file of the network uptil layer layerName
	'''
	if netName=='vgg':
		defFile = '/data1/pulkitag/caffe_models/layer_def_files/vgg_19_%s.prototxt' % layerName
	else:
		print 'Cannont get files for networks other than VGG'
	return defFile	


def get_input_blob_shape(defFile):
	'''
		Get the shape of input blob from the defFile
	'''
	blobShape = []
	with open(defFile,'r') as f:
		lines  = f.readlines()
		ipMode = False
		for l in lines:
			if 'input:' in l:
				ipMode = True
			if ipMode and 'input_dim:' in l:
				ips = l.split()
				blobShape.append(int(ips[1]))
	return blobShape
				

def init_network(defFile, modelFile, isGPU=True, testMode=True):
	'''
		Initialize the network
	'''
	net = caffe.Net(defFile, modelFile)
	if testMode:
		caffe.set_phase_test()
	else:
		caffe.set_phase_train()
	#Set GPU usage
	if isGPU:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()
	return net	


def net_preprocess_init(net, layerName='data',RGBMode=True, meanDat = []):
	'''
		Sets-up preprocessing in layer, layerName. 
		The n/ws are assumed to take inputs in BGR format. 
	'''
	if RGBMode:
		print 'Enabling input in RGB, the Net works in BGR format'
		net.set_channel_swap(layerName,(2,1,0))

	if not meanDat==[]:
		print "Setting Mean.."
		net.set_mean(layerName, meanDat)
		

def preprocess_batch(net, ims, dataLayerName='data'):
	'''
		Inputs are assumed to be numIm*h*w*numChannels
		Preprocess the images
	'''
	assert ims.ndim==4, "Images should be 4D"
	N = ims.shape[0]
	imOut = []
	for i in range(N):
		imOut.append(np.asarray([net.preprocess(dataLayerName, ims[i])]))

	imOut = np.squeeze(np.asarray(imOut))
	if N ==1:
		imOut = np.reshape(imOut, (1,imOut.shape[0], imOut.shape[1], imOut.shape[2]))
	
	return imOut


def deprocess_batch(net, ims, dataLayerName='data'):
	'''
		Retrieve the image back
	'''
	assert ims.ndim==4, "Images should be 4D"
	N,ch,h,w = ims.shape
	imOut = []
	for i in range(N):
		imOut.append(net.deprocess(dataLayerName, np.reshape(ims[i],(1,ch,h,w))))

	imOut = np.squeeze(np.asarray(imOut))
	return imOut


def get_batchsz(net):
	return net.blobs['data'].num


def get_blob_shape(net, blobName):
	assert blobName in net.blobs.keys(), 'Blob Name is not present in the net'
	blob = net.blobs[blobName]
	return blob.num, blob.channels, blob.height, blob.width


def setup_prototypical_network(netName='vgg', layerName='pool4'):
	'''
		Sets up a network in a configuration in which I commonly use it. 
	'''
	modelFile, meanFile = get_model_mean_file(netName)
	defFile             = get_layer_def_files(netName, layerName=layerName)
	meanDat             = read_mean(meanFile)
	net                 = init_network(defFile, modelFile)
	net_preprocess_init(net, layerName='data', meanDat=meanDat)
	return net	


def prepare_image(im, cropSz=[], imMean=[]):
	#Only take the central crop
	shp = im.shape
	w,h = shp[-1],shp[-2]
	
	if np.ndim(im) == 3:
		im = im.reshape(shp[0],1,h,w)
	elif not np.ndim(im) == 4:
		pdb.set_trace()
		print "Incorrect image dimensions"
		raise

	if not cropSz==[]:
		assert cropSz <= w and cropSz <= h
		wSt = int((w - cropSz)/2.0)
		wEn = wSt + cropSz
		hSt = int((h - cropSz)/2.0)
		hEn = hSt + cropSz
		im  = im[:,:,hSt:hEn,wSt:wEn]
	else:
		wSt,wEn = 0,w
		hSt,hEn = 0,h

	if not imMean==[]:
		assert imMean.ndim ==4
		assert imMean.shape[0] == 1
		imMean = imMean[:,:,wSt:wEn,hSt:hEn]
		im = im - imMean

	return im


def read_mean(protoFileName):
	with open(protoFileName,'r') as fid:
		ss = fid.read()
		vec = caffe.io.caffe_pb2.BlobProto()
		vec.ParseFromString(ss)
		mn = caffe.io.blobproto_to_array(vec)

	mn = np.squeeze(mn)
	return mn


def get_features(net, im, layerName=None, ipLayerName='data'):
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
		layerName = []
	
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
		dataLayer = {ipLayerName:imBatch}
		feats = net.forward(blobs=layerName, start=None, end=None, **dataLayer)		 
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


def feats_2_labels(feats, lblType, maskLastLabel=False):
	#feats are assumed to be numEx * featDims
	labels = []
	if lblType in ['uniform20', 'kmedoids30_20']:
		r,c = feats.shape
		if maskLastLabel:
			feats = feats[0:r,0:c-1]
		labels = np.argmax(feats, axis=1)
		labels = labels.reshape((r,1))
	else:
		print "UNrecognized lblType"
		raise
	return labels


def test_network_siamese_h5(imH5File, lbH5File, netFile, defFile, imSz=128, cropSz=112, nCh=3, outLblSz=1, meanFile=[], ipLayerName='data', lblType='uniform20',outFeatSz=20, maskLastLabel=False):
	'''
		maskLastLabel: In some cases it is we may need to compute the error bt ignoring the last label
									 for example in det - where the last class might be the backgroud class
	'''
	print imH5File, lbH5File
	imFid = h5py.File(imH5File,'r')
	lbFid = h5py.File(lbH5File,'r')
	ims1 = imFid['images1/']
	ims2 = imFid['images2/']
	lbls = lbFid['labels/']

	#Initialize network
	net  = init_network(netFile, defFile)
	
	#Get the mean
	imMean = []
	if not meanFile == []:
		imMean = read_mean_txt(meanFile)	
		imMean = imMean.reshape((1,2 * nCh,imSz,imSz))

	#Get Sizes
	imSzSq = imSz * imSz
	assert(ims1.shape[0] % imSzSq == 0 and ims2.shape[0] % imSzSq ==0)
	N     = ims1.shape[0]/(imSzSq * nCh)
	assert(lbls.shape[0] % N == 0)
	lblSz = outLblSz

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
		imsPrep   = prepare_image(ims, cropSz, imMean)  
		predFeat  = get_features(net, imsPrep, ipLayerName=ipLayerName)
		predFeat  = predFeat[0:numIm]
		print numIm
		try:
			labels[i : i + numIm, :]    = feats_2_labels(predFeat.reshape((numIm,outFeatSz)), lblType, maskLastLabel=maskLastLabel)[0:numIm]
			gtLabels[i : i + numIm, : ] = (lbls[i * lblSz : (i+numIm) * lblSz]).reshape(numIm, lblSz) 
		except ValueError:
			print "Value Error found"
			pdb.set_trace()
	
	confMat = compute_error(gtLabels, labels, 'classify')
	return confMat, labels, gtLabels	


def read_mean_txt(fileName):
	with open(fileName,'r') as f:
		l = f.readlines()
		mn = [float(i) for i in l]
		mn = np.array(mn)
	return mn
