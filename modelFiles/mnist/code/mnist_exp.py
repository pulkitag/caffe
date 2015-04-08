import scipy.misc as scm
import pickle
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
import my_pycaffe_io as mpio
import my_pycaffe_utils as mpu
import pdb
import make_rotations as mr


def make_def_proto(nw, isSiamese=True, baseFileStr='split_im.prototxt'):
	'''
		If is siamese then wait for the Concat layers - and make all layers until then siamese.
	'''
	baseDir  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/base_files/'
	baseFile = os.path.join(baseDir, baseFileStr)
	protoDef = mpu.ProtoDef(baseFile)

	if baseFileStr in ['split_im.prototxt', 'normal.prototxt']:
		lastTop  = 'data'

	siameseFlag = isSiamese
	stream1, stream2 = [], []
	mainStream = []

	nameGen     = mpu.LayerNameGenerator()
	for l in nw:
		lType, lParam = l
		lName         = nameGen.next_name(lType)
		#To account for layers that should not copied while finetuning
		# Such layers need to named differently.
		if lParam.has_key('nameDiff'):
			lName = lName + '-%s' % lParam['nameDiff']
		if lType == 'Concat':
			siameseFlag = False
			if not lParam.has_key('bottom2'):
				lParam['bottom2'] = lastTop + '_p'
	
		if siameseFlag:
			lDef, lsDef = mpu.get_siamese_layerdef_for_proto(lType, lName, lastTop, **lParam)
			stream1.append(lDef)
			stream2.append(lsDef)
		else:
			lDef = mpu.get_layerdef_for_proto(lType, lName, lastTop, **lParam)
			mainStream.append(lDef)

		if lParam.has_key('shareBottomWithNext'):
			assert lParam['shareBottomWithNext']
			pass
		else:
			lastTop = lName

	#Add layers
	mainStream = stream1 + stream2 + mainStream
	for l in mainStream:
		protoDef.add_layer(l['name'][1:-1], l)	

	return protoDef

##
# Generates a string to represent the n/w name

def nw2name(nw):
	nameGen     = mpu.LayerNameGenerator()
	nwName = []
	for l in nw:
		lType, lParam = l
		lName = nameGen.next_name(lType)
		if lType in ['InnerProduct', 'Convolution']:
			lName = lName + '-%d' % lParam['num_output']
			if lType == 'Convolution':
				lName = lName + 'sz%d-st%d' % (lParam['kernel_size'], lParam['stride'])
			nwName.append(lName)
		elif lType in ['Concat', 'Dropout']:
			nwName.append(lName)
		else:
			pass
	nwName = ''.join(s + '_' for s in nwName)
	nwName = nwName[:-1]
	return nwName	


def get_caffe_prms(nw, isSiamese=True, batchSize=128, isTest=False,
						isFineTune=False, fineExp=None, fineModelIter=None,
						max_iter=40000, stepsize=10000, snapshot=5000, gamma=0.5, base_lr=0.01,
						test_iter=100, test_interval=500, lr_policy='"step"'):
	'''
		isFineTune: If the weights of an auxiliary experiment are to be used to start finetuning
		fineExp   : Instance of CaffeExperiment from which finetuning needs to begin. 	
		fineModelIterations: Used for getting model needed for finetuning. 
	'''
	cPrms  = {}
	nwName = nw2name(nw)
	cPrms['nw']     = nw
	cPrms['nwName'] = nwName
	cPrms['isSiamese'] = isSiamese
	cPrms['batchSize'] = batchSize
	cPrms['isTest']     = isTest
	cPrms['isFineTune'] = isFineTune 
	cPrms['fineExp']    = fineExp
	cPrms['fineModelIter'] = fineModelIter

	expStr = []
	if isFineTune:
		assert fineExp is not None
		assert fineModelIter is not None
		expStr.append('FROM-%s-TO' % fineExp.dataExpName_)
		cPrms['initModelFile'] = fineExp.get_snapshot_name(fineModelIter)

	if isSiamese:
		expStr.append('siam')
	expStr.append('bLr%.0e' % base_lr)
	expStr.append('stp%.0e' % stepsize)
	expStr.append('mIt%.0e'  % max_iter) 
	expStr = ''.join(s + '_' for s in expStr)
	cPrms['expName'] = expStr[0:-1]

	#Setup the solver
	solArgs = {'test_iter': test_iter, 'test_interval': test_interval, 'max_iter': max_iter,
						 'stepsize': stepsize, 'gamma': gamma, 'base_lr': base_lr, 'lr_policy':lr_policy}
	cPrms['solver'] = mpu.make_solver(**solArgs)  
	return cPrms
	

def get_experiment_object(prms, cPrms, deviceId=1):
	targetExpDir = prms['paths']['expDir']
	expName = prms['expName'] + '_' + cPrms['nwName']
	caffeExpName = cPrms['expName']
	caffeExp = mpu.CaffeExperiment(expName, caffeExpName,
								targetExpDir, prms['paths']['snapDir'],
								deviceId=deviceId, isTest=cPrms['isTest'])
	return caffeExp


def setup_experiment(prms, cPrms, deviceId=1):
	caffeExp = get_experiment_object(prms, cPrms, deviceId=deviceId)
	
	#make the def file
	if prms['transform'] == 'rotTrans':
		baseFile  = 'split_im.prototxt'
		labelName = 'label'
	elif prms['transform'] == 'normal':
		baseFile  = 'normal.prototxt'
		labelName = 'label' 
	else:
		raise Exception('Transform type not recognized')

	netdef = make_def_proto(cPrms['nw'], cPrms['isSiamese'], baseFileStr=baseFile) 
	caffeExp.init_from_external(cPrms['solver'], netdef)	

	#Set the lmdbs
	trainImDb = prms['paths']['lmdb']['train']['im']
	testImDb  = prms['paths']['lmdb']['test']['im']
	trainLbDb = prms['paths']['lmdb']['train']['lb']
	testLbDb  = prms['paths']['lmdb']['test']['lb']
	if prms['transform'] == 'rotTrans':
		caffeExp.set_layer_property('pair_data', ['data_param','source'],
																 '"%s"' % trainImDb, phase='TRAIN')
		caffeExp.set_layer_property('pair_data', ['data_param','batch_size'],
																 cPrms['batchSize'], phase='TRAIN')
		caffeExp.set_layer_property('pair_data', ['data_param','source'], '"%s"' % testImDb,  phase='TEST')
		caffeExp.set_layer_property(labelName, ['data_param','source'], '"%s"' % trainLbDb, phase='TRAIN')
		caffeExp.set_layer_property(labelName, ['data_param','batch_size'], cPrms['batchSize'], phase='TRAIN')
		caffeExp.set_layer_property(labelName, ['data_param','source'], '"%s"' % testLbDb,  phase='TEST')
	elif prms['transform'] == 'normal':
		caffeExp.set_layer_property('data', ['data_param','source'],
																 '"%s"' % trainImDb, phase='TRAIN')
		caffeExp.set_layer_property('data', ['data_param','batch_size'],
																 cPrms['batchSize'], phase='TRAIN')
		caffeExp.set_layer_property('data', ['data_param','source'], '"%s"' % testImDb,  phase='TEST')
	else:
		raise Exception('Not recognized')
	return caffeExp

##
def make_experiment(prms, cPrms, deviceId=1):
	caffeExp = setup_experiment(prms, cPrms, deviceId=deviceId)
	if cPrms['isFineTune']:
		caffeExp.make(modelFile=cPrms['initModelFile'], writeTest=cPrms['isTest'])
	else:
		caffeExp.make()
	return caffeExp

def run_experiment(prms, cPrms, deviceId=1):
	caffeExp = make_experiment(prms, cPrms, deviceId)
	caffeExp.run()

##
# Run InnerProduct networks
def run_networks():
	deviceId = 2
	nw = []
	
	nw.append( [('InnerProduct',{'num_output': 200}),('ReLU',{}),('Concat',{'concat_dim':1}),
						 ('InnerProduct',{'num_output': 100}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5}), 
						 ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )
	
	nw.append( [('InnerProduct',{'num_output': 64}),('ReLU',{}),('Concat',{'concat_dim':1}),
						 ('InnerProduct',{'num_output': 100}),('ReLU',{}),('Dropout', {'dropout_ratio': 0.5}),
						 ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )

	nw.append( [('InnerProduct',{'num_output': 500}),('ReLU',{}),('Concat',{'concat_dim':1}),
						 ('InnerProduct',{'num_output': 100}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5}), 
						 ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )
	
	prms  = mr.get_prms(maxRot=10, maxDeltaRot=30)
	for nn in nw:
		cPrms = get_caffe_prms(nn, isSiamese=True)
		run_experiment(prms, cPrms, deviceId=deviceId)
	#caffeExp = make_experiment(prms, cPrms, deviceId=deviceId)
	#return caffeExp

##
# Run Convolutional Filters
def run_networks_conv():
	deviceId = 2
	nw = []
	
	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 11, 'stride': 5}),
						  ('ReLU',{}),('Concat',{'concat_dim':1}),
						 ('InnerProduct',{'num_output': 100}),('ReLU',{}), 
						 ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )
	
	prms  = mr.get_prms(maxRot=10, maxDeltaRot=30)
	for nn in nw:
		cPrms = get_caffe_prms(nn, isSiamese=True)
		run_experiment(prms, cPrms, deviceId=deviceId)



def run_finetune():
	deviceId = 2
	sourceNw = []	
	targetNw = []
 
	sourceNw.append( [('InnerProduct',{'num_output': 200}),('ReLU',{}),('Concat',{'concat_dim':1}),
						 ('InnerProduct',{'num_output': 100}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5}), 
						 ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )
	
	targetNw.append( [('InnerProduct',{'num_output':200}), ('ReLU',{}),
							('InnerProduct', {'num_output': 100, 'nameDiff': 'ft'}), ('ReLU',{}),
						  ('InnerProduct', {'num_output': 10, 'nameDiff': 'ft'}), 
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	srcPrms = mr.get_prms(maxRot=10, maxDeltaRot=30)
	tgtPrms = mr.get_prms(transform='normal', numTrainEx=10000)
	for snn,tnn in zip(sourceNw, targetNw):	
		#Source Experiment
		srcCaffePrms = get_caffe_prms(snn, isSiamese=True)
		srcExp = setup_experiment(srcPrms, srcCaffePrms)
		#Target Experiment
		tgtCaffePrms = get_caffe_prms(tnn, isSiamese=False, isFineTune=True, fineExp=srcExp,
												fineModelIter=40000)
		run_experiment(tgtPrms, tgtCaffePrms, deviceId=deviceId)


def run_scratch():
	deviceId = 2
	nw = []
	nw.append( [('InnerProduct',{'num_output':200}), ('ReLU',{}),
							('InnerProduct', {'num_output': 100, 'nameDiff': 'ft'}), ('ReLU',{}),
						  ('InnerProduct', {'num_output': 10, 'nameDiff': 'ft'}), 
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	prms = mr.get_prms(transform='normal', numTrainEx=10000)
	for nn in nw:	
		cPrms = get_caffe_prms(nn, isSiamese=False)
		run_experiment(prms, cPrms, deviceId=deviceId)


