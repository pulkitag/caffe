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
import collections as co
import h5py as h5

BASE_DIR = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/base_files/'

def make_def_proto(nw, isSiamese=True, baseFileStr='split_im.prototxt'):
	'''
		If is siamese then wait for the Concat layers - and make all layers until then siamese.
	'''
	baseFile = os.path.join(BASE_DIR, baseFileStr)
	protoDef = mpu.ProtoDef(baseFile)

	#if baseFileStr in ['split_im.prototxt', 'normal.prototxt']:
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
	#Solver prms
	cPrms['max_iter'] = max_iter

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
		if prms['lossType'] == 'classify':
			baseFile    = 'mnist_cls_top.prototxt'
			botBaseFile = 'mnist_cls_bot.prototxt'  
			labelName = 'label'
		elif prms['lossType'] == 'regress':
			baseFile  = 'split_im.prototxt'
			labelName = 'label'
		else:
			raise Exception('loss type not recognized')
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
		if prms['lossType'] == 'classify':
			#Connect the loss layers appropriately
			lastTop = caffeExp.get_last_top_name()
			botDef = mpu.ProtoDef(os.path.join(BASE_DIR, botBaseFile))
			for _,l in botDef.layers_['TRAIN'].iteritems():
				caffeExp.add_layer(l['name'][1:-1], l, phase='TRAIN')	
			#Make appropriate modifications to the fc layers
			fcNames = ['delx_fc', 'dely_fc', 'rot_fc']	
			fcSz    = [prms['trnLblSz'], prms['trnLblSz'], prms['rotLblSz']]
			for name,sz in zip(fcNames, fcSz):
				caffeExp.set_layer_property(name, ['bottom'], '"%s"' % lastTop)
				caffeExp.set_layer_property(name, ['inner_product_param', 'num_output'], sz)

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

	'''	
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

	'''
	
	nw.append( [('InnerProduct',{'num_output': 200}),('ReLU',{}),('Concat',{'concat_dim':1}),
						 ('InnerProduct',{'num_output': 100}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5})])
	
	prms  = mr.get_prms(lossType='classify',maxRot=10, maxDeltaRot=30)
	for nn in nw:
		cPrms = get_caffe_prms(nn, isSiamese=True)
		run_experiment(prms, cPrms, deviceId=deviceId)
		#caffeExp = make_experiment(prms, cPrms, deviceId=deviceId)
		#return caffeExp
	#return caffeExp

##
# Run Convolutional Filters
def run_networks_conv():
	deviceId = 0
	nw = []

	'''	
	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 200, 'kernel_size': 5, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 400}),('ReLU',{}), 
						  ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )

	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 500, 'kernel_size': 5, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 500}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5}), 
						  ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )
	
	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 250, 'kernel_size': 5, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 1000}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5}), 
						  ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )
	
	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 250, 'kernel_size': 3, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 1000}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5}), 
						  ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )
	
	nw.append( [('Convolution',{'num_output': 200, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 200, 'kernel_size': 3, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 1000}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5}), 
						  ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})] )

	prms  = mr.get_prms(maxRot=10, maxDeltaRot=30)
	'''

	'''	
	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 200, 'kernel_size': 5, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 400}),('ReLU',{})] )

	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 500, 'kernel_size': 5, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 500}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5})])
	
	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 250, 'kernel_size': 5, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 1000}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5})]) 
	
	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 250, 'kernel_size': 3, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 1000}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5})] )
	
	nw.append( [('Convolution',{'num_output': 200, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
							('Convolution',{'num_output': 200, 'kernel_size': 3, 'stride': 2}), ('ReLU',{}),
							('Concat',{'concat_dim':1}),
						  ('InnerProduct',{'num_output': 1000}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5})]) 
	'''	
	nw.append( [('Convolution', {'num_output': 50, 'kernel_size': 5, 'stride': 2})])			 

	prms  = mr.get_prms(maxRot=10, maxDeltaRot=30, lossType='classify', numTrainEx=1e+07)

	for nn in nw:
		cPrms = get_caffe_prms(nn, isSiamese=True)
		run_experiment(prms, cPrms, deviceId=deviceId)


	
def run_finetune(max_iter=5000, stepsize=1000):
	deviceId = 2
	sourceNw = []	
	targetNw = []

	''' 
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
	'''
	sourceNw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
              ('Convolution',{'num_output': 500, 'kernel_size': 5, 'stride': 2}), ('ReLU',{}),
              ('Concat',{'concat_dim':1}),
              ('InnerProduct',{'num_output': 500}),('ReLU',{}), ('Dropout', {'dropout_ratio': 0.5})])	

	targetNw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
              ('Convolution',{'num_output': 500, 'kernel_size': 5, 'stride': 2}), ('ReLU',{}),
              ('InnerProduct',{'num_output': 200, 'nameDiff': 'ft'}),('ReLU',{}),
              ('InnerProduct',{'num_output': 10, 'nameDiff': 'ft'}),
							('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
							('Accuracy', {'bottom2': 'label'})] )

	srcPrms = mr.get_prms(maxRot=10, maxDeltaRot=30,
					 lossType='classify', numTrainEx=1e+7)
	for numEx in  [100, 1000, 10000]:
		for snn,tnn in zip(sourceNw, targetNw):	
			#Source Experiment
			srcCaffePrms = get_caffe_prms(snn, isSiamese=True)
			srcExp = setup_experiment(srcPrms, srcCaffePrms)
			#Target Experiment
			tgtPrms = mr.get_prms(transform='normal', numTrainEx=numEx)
			tgtCaffePrms = get_caffe_prms(tnn, isSiamese=False, isFineTune=True, fineExp=srcExp,
													fineModelIter=40000, max_iter=max_iter, stepsize=stepsize)
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

##
# Find the adversary network starting from scratch. 
def find_adversary_scratch(runType='run', max_iter=10000, stepsize=10000):
	'''
		runType: run  - just run the experiment
		         test - test the results.
						 accuracy - just read the accuracy.  
	'''	
	numEx = [100, 1000, 10000]
	nw = []
	acc = co.OrderedDict()

	#LeNet
	nw.append([('Convolution',{'num_output': 20, 'kernel_size': 5, 'stride': 1}), ('ReLU', {}),
				('Pooling', {'kernel_size': 2, 'stride': 2}),
				('Convolution', {'num_output': 50, 'kernel_size': 5, 'stride': 1}), ('ReLU', {}),
				('Pooling', {'kernel_size': 2, 'stride': 2}),
				('InnerProduct', {'num_output': 500}), ('ReLU', {}), ('InnerProduct', {'num_output': 10}), 
				('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
				('Accuracy', {'bottom2': 'label'})])

	#LeNet with one layer. 
	nw.append([('Convolution',{'num_output': 20, 'kernel_size': 5, 'stride': 1}), ('ReLU', {}),
				('Pooling', {'kernel_size': 3, 'stride': 2}),
				('InnerProduct', {'num_output': 500}), ('ReLU', {}), ('InnerProduct', {'num_output': 10}), 
				('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
				('Accuracy', {'bottom2': 'label'})])

	#LeNet with 96 conv-1
	nw.append([('Convolution',{'num_output': 96, 'kernel_size': 5, 'stride': 1}), ('ReLU', {}),
				('Pooling', {'kernel_size': 2, 'stride': 2}),
				('Convolution', {'num_output': 50, 'kernel_size': 5, 'stride': 1}), ('ReLU', {}),
				('Pooling', {'kernel_size': 2, 'stride': 2}),
				('InnerProduct', {'num_output': 500}), ('ReLU', {}), ('InnerProduct', {'num_output': 10}), 
				('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
				('Accuracy', {'bottom2': 'label'})])

	#1-layered N/w
	nw.append([('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 2}), ('ReLU', {}),
				('Pooling', {'kernel_size': 3, 'stride': 2}),
				('InnerProduct', {'num_output': 500}), ('ReLU', {}), ('InnerProduct', {'num_output': 10}), 
				('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
				('Accuracy', {'bottom2': 'label'})])

	#1-layered n/w with different kernel size. 
	nw.append([('Convolution',{'num_output': 96, 'kernel_size': 5, 'stride': 1}), ('ReLU', {}),
				('Pooling', {'kernel_size': 3, 'stride': 2}),
				('InnerProduct', {'num_output': 500}), ('ReLU', {}), ('InnerProduct', {'num_output': 10}), 
				('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
				('Accuracy', {'bottom2': 'label'})])

	#Same architecture as one of my networks. 
 	nw.append( [('Convolution',{'num_output': 96, 'kernel_size': 7, 'stride': 3}), ('ReLU',{}),
              ('Convolution',{'num_output': 500, 'kernel_size': 5, 'stride': 2}), ('ReLU',{}),
              ('InnerProduct',{'num_output': 200, 'nameDiff': 'ft'}),('ReLU',{}),
              ('InnerProduct',{'num_output': 10, 'nameDiff': 'ft'}),
              ('SoftmaxWithLoss', {'bottom2': 'label', 'shareBottomWithNext': True}),
              ('Accuracy', {'bottom2': 'label'})] )

	acc['numExamples'] = numEx
	for nn in nw:
		name = nw2name(nn)
		acc[name] = []

	for ex in numEx:
		for nn in nw:
			prms = mr.get_prms(transform='normal', numTrainEx=ex)
			#I have tried stepsize of 2000 and stepsize of 10000
			cPrms = get_caffe_prms(nn, isSiamese=False, max_iter=max_iter, stepsize=stepsize)
			if runType == 'run':
				run_experiment(prms, cPrms, deviceId=0)
			elif runType == 'test':
				run_test(prms, cPrms)
			elif runType == 'accuracy':
				caffeExp = get_experiment_object(prms, cPrms)
				name = nw2name(nn)
				resFile  = prms['paths']['resFile'] % (caffeExp.dataExpName_ + '_' + cPrms['expName'])
				#print resFile
				res     = h5.File(resFile, 'r')
				acc[name].append(res['acc'][:][0])
				res.close()
			else:
				raise Exception('Unrecognized run type %s' % runType)
	if runType == 'accuracy':
		return acc, numEx	
	
## 
def run_test(prms, cPrms, cropH=28, cropW=28, imH=28, imW=28):
	caffeExp  = setup_experiment(prms, cPrms)
	caffeTest = mpu.CaffeTest.from_caffe_exp_lmdb(caffeExp, prms['paths']['lmdb']['test']['im'])
	
	delLayers = []
	delLayers = delLayers + caffeExp.get_layernames_from_type('Accuracy')
	delLayers = delLayers + caffeExp.get_layernames_from_type('SoftmaxWithLoss')

	#The last inner product layer is the classification layer.
	opNames   = caffeExp.get_layernames_from_type('InnerProduct')
	numOp     = caffeExp.get_layer_property(opNames[-1], 'num_output')
	assert numOp==10, 'Last FC layer has wrong number of outputs, %d instead of 10' % numOp	
	caffeTest.setup_network([opNames[-1]], imH=imH, imW=imW,
								 cropH=cropH, cropW=cropW, channels=1,
								 modelIterations=cPrms['max_iter'], delLayers=delLayers)
	caffeTest.run_test()
	resFile  = prms['paths']['resFile'] % (caffeExp.dataExpName_ + '_' + cPrms['expName'])
	dirName  = os.path.dirname(resFile)
	if not os.path.exists(dirName):
		os.makedirs(dirName)
	caffeTest.save_performance(['acc', 'accClassMean'], resFile)

