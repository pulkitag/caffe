import make_rotations as mr
import mnist_exp  as me
import numpy as np
import my_pycaffe_utils as mpu
import pdb
import collections as co

def get_noisy_network(adaptiveSigma=0.1):
	nw = []
	nw.append([('Convolution',  {'num_output': 96,  'kernel_size': 3, 'stride': 1}), 
					('RandomNoise', {'adaptive_sigma': 'true', 'adaptive_factor': adaptiveSigma}), ('ReLU',{}),
					('Pooling', {'kernel_size': 3, 'stride': 2}),
					('Convolution',  {'num_output': 256, 'kernel_size': 3, 'stride': 1}),
					('RandomNoise', {'adaptive_sigma': 'true', 'adaptive_factor': adaptiveSigma}), ('ReLU',{}),
					('Pooling', {'kernel_size': 3, 'stride': 2}),
					('InnerProduct', {'num_output': 500}), 
					('RandomNoise', {'adaptive_sigma': 'true', 'adaptive_factor': adaptiveSigma}), ('ReLU',{}),
					('Dropout', {'dropout_ratio': 0.5}),
					('InnerProduct', {'num_output': 10}), 
					('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
					('Accuracy', {'bottom2': 'label'})] )			 

	return nw


def train_network(lrAbove=None, max_iter=10000, stepsize=5000, runType='train'):
	deviceId=2
	nw = []
	aSigma = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
	for sig in aSigma:
		nw = nw + get_noisy_network(sig)
	numEx = [1000, 10000, 60000]

	acc = co.OrderedDict()
	for ex in numEx:
		exKey = 'n%d' % ex	
		acc[exKey] = co.OrderedDict()
		for nn in nw:
			name = me.nw2name(nn)
			prms = mr.get_prms(transform='normal', numTrainEx=ex)
			cPrms = me.get_caffe_prms(nn, isSiamese=False, lrAbove=lrAbove, max_iter=max_iter, stepsize=stepsize)
			if runType=='train': 
				me.run_experiment(prms, cPrms, deviceId=deviceId)
			elif runType=='test':
				me.run_test(prms, cPrms)
			elif runType == 'accuracy':
				acc[exKey][name] = me.read_accuracy(prms, cPrms)

	return acc


