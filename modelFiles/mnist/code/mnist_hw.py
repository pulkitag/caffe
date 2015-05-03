import make_rotations as mr
import mnist_exp  as me
import numpy as np
import my_pycaffe_utils as mpu
import pdb
import collections as co

def get_networks():
	nw = []
	nw.append( [('InnerProduct', {'num_output': 20}),  ('ReLU',{}),
						 	('InnerProduct', {'num_output': 500}), ('ReLU',{}),
						 	('InnerProduct', {'num_output': 10}),
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	nw.append( [('InnerProduct', {'num_output': 20}),  ('ReLU',{}),
						 	('InnerProduct', {'num_output': 50}), ('ReLU',{}),
						 	('InnerProduct', {'num_output': 500}), ('ReLU',{}),
						 	('InnerProduct', {'num_output': 10}),
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	#Caffe LeNet architecture. 
	nw.append( [('Convolution',{'num_output': 20, 'kernel_size': 5, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
							('Convolution',{'num_output': 50, 'kernel_size': 5, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
						  ('InnerProduct',{'num_output': 500}),('ReLU',{}), 
						 	('InnerProduct', {'num_output': 10}),
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	#Modified Le-Net
	nw.append( [('Convolution',{'num_output': 20, 'kernel_size': 5, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
							('Convolution',{'num_output': 50, 'kernel_size': 5, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
						  ('InnerProduct',{'num_output': 32}),('ReLU',{}), 
						 	('InnerProduct', {'num_output': 10}),
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	nw.append( [('Convolution',{'num_output': 20, 'kernel_size': 3, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
							('Convolution',{'num_output': 50, 'kernel_size': 3, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
						  ('InnerProduct',{'num_output': 32}),('ReLU',{}), 
						 	('InnerProduct', {'num_output': 10}),
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	nw.append( [('Convolution',{'num_output': 20, 'kernel_size': 7, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
							('Convolution',{'num_output': 50, 'kernel_size': 7, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
						  ('InnerProduct',{'num_output': 32}),('ReLU',{}), 
						 	('InnerProduct', {'num_output': 10}),
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	return nw

def train_networks(lrAbove=None, max_iter=20000, stepsize=5000, runType='run'):
	deviceId=2
	nw = get_networks()
	numEx = [10000, 60000]
	acc = co.OrderedDict()
	for ex in numEx:	
		exKey = 'n%d' % ex
		acc[exKey] = co.OrderedDict()
		for nn in nw:
			name  = me.nw2name_small(nn)
			prms  = mr.get_prms(transform='normal', numTrainEx=ex)
			cPrms = me.get_caffe_prms(nn, isSiamese=False, lrAbove=lrAbove, max_iter=max_iter, stepsize=stepsize)
		
			isExist = me.find_experiment(prms, cPrms, max_iter)
			print name
			if runType == 'run':
				print 'EXPERIMENT EXISTS - SKIPPING'
				if not isExist:
					me.run_experiment(prms, cPrms, deviceId=deviceId)
			elif runType == 'test':
				if isExist:
					me.run_test(prms, cPrms)
			elif runType == 'accuracy':
				if isExist:
					acc[exKey][name] = me.read_accuracy(prms, cPrms)
			else:
				raise Exception('Unrecognized run type %s' % runType)
	return acc	

