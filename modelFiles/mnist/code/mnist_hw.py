import make_rotations as mr
import mnist_exp  as me
import numpy as np
import my_pycaffe_utils as mpu
import pdb
import collections as co
import my_pycaffe as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
	
	nw.append( [('InnerProduct', {'num_output': 20}),  ('Sigmoid',{}),
						 	('InnerProduct', {'num_output': 500}), ('Sigmoid',{}),
						 	('InnerProduct', {'num_output': 10}),
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	nw.append( [('InnerProduct', {'num_output': 20}),  ('Sigmoid',{}),
						 	('InnerProduct', {'num_output': 50}), ('Sigmoid',{}),
						 	('InnerProduct', {'num_output': 500}), ('Sigmoid',{}),
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


def compile_results():
	numEx    = [10000, 60000]
	exKey    = ['n%d' % ex for ex in numEx]
	nws      = get_networks()
	acc      = train_networks(runType='accuracy')
	lines = []
	for nn in nws:
		key, latKey = me.nw2name_small(nn, True)
		l = key
		for ex in exKey:
			l = l + ' & ' + '%.2f' % (100 * acc[ex][key])
		l = l + '\\\ \n'
		lines.append(l)

	resFile = '/data1/pulkitag/others/gsi/mnist_results.tex'
	fid = open(resFile, 'w')
	for l in lines:
		fid.write(l)
	fid.close()
	
def vis_lenet():
	nn =        [('Convolution',{'num_output': 20, 'kernel_size': 5, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
							('Convolution',{'num_output': 50, 'kernel_size': 5, 'stride': 1}),
							('Pooling',{'kernel_size': 2, 'stride': 2}),
						  ('InnerProduct',{'num_output': 500}),('ReLU',{}), 
						 	('InnerProduct', {'num_output': 10}),
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})]

	prms  = mr.get_prms(transform='normal', numTrainEx=60000)
	cPrms = me.get_caffe_prms(nn, isSiamese=False, lrAbove=None, max_iter=20000, stepsize=5000)
	exp   = me.setup_experiment(prms, cPrms)
	modelFile = exp.get_snapshot_name(numIter=20000)
	defFile   = exp.files_['netdef']
	netUnsup    = mp.MyNet(defFile, modelFile)
	fig = plt.figure()
	ax = plt.subplot(1,1,1)
	plt.set_cmap(plt.cm.gray)
	plt.ion()
	netUnsup.vis_weights('conv1', ax=ax, isFc=False, h=4, w=5)
	plt.show()
	with PdfPages('/data1/pulkitag/others/gsi/lenet_weights.pdf') as pdf:
		pdf.savefig()

