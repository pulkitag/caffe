import numpy as np
import other_utils as ou
import matplotlib.pyplot as plt
import pdb
import caffe
import os
import scipy.misc as scm
import copy

def test_normal():
	net = caffe.Net('euclidean.prototxt', caffe.TEST)
	numRound = 100
	for r in range(numRound):
		pred = np.random.rand(10,31,1,1).astype(np.float32) 
		lb1  = np.random.rand(10,31,1,1).astype(np.float32)
		lb2  = np.ones((10,32,1,1)).astype(np.float32)
		lb2[:,0:31,:,:] = copy.copy(lb1)
		data  = net.forward(blobs=['loss','loss-ig'],**{'pred': pred, 'label1': lb1, 'label2': lb2})
		loss1 = data['loss']
		loss2 = data['loss-ig']
		diff  = np.abs(loss1 - loss2)
		assert(float(diff)/min(loss1, loss2) < 1e-6)
	print "NORMAL RUN TEST PASSED"

def test_ignore():
	net = caffe.Net('euclidean.prototxt', caffe.TEST)
	numRound = 100
	for r in range(numRound):
		pred = np.random.rand(10,31,1,1).astype(np.float32) 
		lb1  = np.random.rand(10,31,1,1).astype(np.float32)
		lb2  = np.ones((10,32,1,1)).astype(np.float32)
		lb2[:,0:31,:,:] = copy.copy(lb1)
		#Randomly chose examples to ignore
		randIdx = np.random.rand(10)
		rCount  = 0
		for rr in range(10):
			if randIdx[rr] < 0.5:
				rCount = rCount + 1
				lb2[rr,31,0,0] = 0
		data  = net.forward(blobs=['loss-ig'],**{'pred': pred, 'label1': lb1, 'label2': lb2})
		loss2 = data['loss-ig']
		#Calculate the same by zeroing out the pred and lb	
		for rr in range(10):
			if randIdx[rr] < 0.5:
				lb1[rr,:,:,:] = 0
				pred[rr,:,:,:] = 0
		#print rCount
		data  = net.forward(blobs=['loss'],**{'pred': pred, 'label1': lb1, 'label2': lb2})
		loss1 = data['loss']
		diff  = np.abs(loss1 - loss2)
		#print (diff)
		assert(float(diff)/min(loss1, loss2) < 1e-6)
	print "IGNORE RUN TEST PASSED"


