import sys
import os
import numpy as np
import pickle
import caffe
import scipy.misc as scm
import random

def load_test_images():
	dataPath  = '/data1/pulkitag/mnist/raw/'
	dataFile  = dataPath + 'testImages.pkl' 
	with open(dataFile,'r') as f:
		data = pickle.load(f)
		im    = data['im']
		label = data['label']
	
	im = im /256.0
	return im, label


def rotate_images(im, randomSample=False, N=0):
	'''
		Rotate the images
		randomSample: if True, randomly sample (with repetition to form images) 
	'''
	
	numIm,h,w = im.shape 
	if randomSample:
		idxs = np.random.choice(im.shape[0], N)
	else:
		idxs = range(im.shape[0])
	
	imOut = np.zeros((len(idxs),h,w))
	for (i,idx) in enumerate(idxs):
		imC  = im[idx]
		ang = random.uniform(0,90)
		sgn = random.random()
		if sgn > 0.5:
				sgn = -1
		else:
				sgn = 1
		ang = ang * sgn
		imOut[i]  = scm.imrotate(imC, ang)

	return imOut


def feats2labels(feats):
	[N,ch,h,w] = feats.shape
	assert h==1 and w == 1
	
	lbl = np.zeros((N,1))
	for i in range(N):
		f = feats[i]
		lbl[i] = f.argmax()

	return lbl
	

def get_accuracy(net, im, label):
	feats = get_features(net, im)
	predLabels = feats2labels(feats)
	hits = [label[i]==predLabels[i] for i  in range(im.shape[0])]
	acc  = float(sum(hits))/len(hits)
	return acc


def init_network(modelFile, defFile, isGPU=True):
	net = caffe.Net(modelFile, defFile)
	net.set_phase_test()
	if isGPU:
		net.set_mode_gpu()
	else:
		net.set_mode_cpu()
	
	assert len(net.inputs)==1
	net.set_raw_scale(net.inputs[0], 1.0/256)

	return net	


def get_features(net, im, layerName=None):
	dataBlob = net.blobs['data']
	batchSz  = dataBlob.num
	N,nr,nc  = im.shape
	assert nr == dataBlob.height and nc==dataBlob.width

	if not layerName==None:
		assert layerName in net.blobs.keys()
		layerName = [layerName]
		outName   = layerName[0]	
	else:
		outName   = net.outputs[0]

	print layerName
	imBatch = np.zeros((batchSz,1,nr,nc))
	outFeats = {}
	outBlob  = net.blobs[outName]
	outFeats = np.zeros((N, outBlob.channels, outBlob.height, outBlob.width))
	for i in range(0,N,batchSz):
		st = i
		en = min(N, st + batchSz)
		l  = en - st
		tempIm = im[st:en]
		imBatch[0:l,:,:,:] = tempIm.reshape((l,1,nr,nc))
		feats = net.forward(blobs=layerName, start=None, end=None, data=imBatch)		 
		outFeats[st:en] = feats[outName][0:l]

	return outFeats

	

def test_classify(modelFile, defFile):
	#From the n/w
	net = init_network(modelFile, defFile)
	
	#Get test images and labels
	im,label = load_test_images()

	#Sanity-Check- Verify accuracy
	acc = get_accuracy(net, im, label)	
	print "Accuracy is: %f" % acc


def test_rots_space(modelFile, defFile):
	'''
	Generates rotated versions of images to test if:
	digit_x + rot1(digit_y) - rot2(digit_y) == rot(digit_x)
	where rot is a composition of rot1 and rot2
	'''
	#From the n/w
	net = init_network(modelFile, defFile)
	
	#Get test images and labels
	im,label = load_test_images()

	#TestLayer
	layerName = 'ip1'

	imRef   = im[0:100]
	imTr    = im[100:200]
	imOther = im[200:-1]

	print "Getting Rotations"
	imTrRot1   = rotate_images(imTr)
	imTrRot2   = rotate_images(imTr)
	imOtherRot = rotate_images(imOther,randomSample=True,N=100000)

	print "Computing Features"
	featRef       = get_features(net, imRef,layerName)
	featTrRot1    = get_features(net, imTrRot1, layerName)
	featTrRot2    = get_features(net, imTrRot2, layerName)
	featOtherRot  = get_features(net, imOtherRot, layerName)

	featEmbed = featRef + featTrRot1 - featTrRot2
	#idxs      = find_matches(featEmbed, featOtherRot)	
