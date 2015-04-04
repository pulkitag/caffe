## @package my_pycaffe_utils
#  Miscellaneous Util Functions
#

import my_pycaffe as mp
import my_pycaffe_io as mpio
import numpy as np
import pdb
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import subprocess
import collections as co
import other_utils as ou
import shutil

def zf_saliency(net, imBatch, numOutputs, opName, ipName='data', stride=2, patchSz=11):
	'''
		Takes as input a network and set of images imBatch
		net: Instance of MyNet
		imBatch: the images for which saliency needs to be computed. (expects: N * ch * h * w)
		numOutputs: Number of output units in the blob named opName
		Produces the saliency map of im
		net is of type MyNet
	'''

	assert(np.mod(patchSz,2)==1), 'patchSz needs to an odd Num'
	p = int(np.floor(patchSz/2.0))
	N = imBatch.shape[0]

	if isinstance(opName, basestring):
		opName = [opName]
		numOutputs = [numOutputs]
	
	#Transform the image
	imT = net.preprocess_batch(imBatch)

	#Find the Original Scores
	dataLayer = {}
	dataLayer[ipName] = imT
	batchSz,ch,nr,nc = imT.shape
	dType     = imT.dtype
	nrNew     = len(range(p, nr-p-1, stride))
	ncNew     = len(range(p, nc-p-1, stride)) 

	origScore = np.copy(net.net.forward_all(**dataLayer))
	for op in opName:
		assert op in origScore.keys(), "Some outputs not found"

	imSalient = {}
	for (op, num) in zip(opName, numOutputs):
		imSalient[op] = np.zeros((N, num, nrNew, ncNew))
 
	for (imCount,im) in enumerate(imT[0:N]):
		count   = 0
		imIdx   = []
		ims     = np.zeros(imT.shape).astype(dType)
		for (ir,r) in enumerate(range(p, nr-p-1, stride)):
			for (ic,c) in enumerate(range(p, nc-p-1, stride)):
				imPatched = np.copy(im)
				#Make an image patch 0
				imPatched[:, r-p:r+p+1, c-p:c+p+1] = 0	
				ims[count,:,:,:] = imPatched
				imIdx.append((ir,ic))
				count += 1
				#If count is batch size compute the features
				if count==batchSz or (ir == nrNew-1 and ic == ncNew-1):
					dataLayer = {}
					dataLayer[ipName] = net.preprocess_batch(ims)
					allScores = net.net.forward(**dataLayer)
					for (op,num) in zip(opName, numOutputs):
						scores = origScore[op][imCount] - allScores[op][0:count]
						scores = scores.reshape((count, num))
						for idx,coords in enumerate(imIdx):
							y, x = coords
							imSalient[op][imCount, :, y, x] = scores[idx,:].reshape(num,)
					count = 0
					imIdx = []	
	
	return imSalient, origScore


def mapILSVRC12_labels_wnids(metaFile):
	dat    = sio.loadmat(metaFile, struct_as_record=False, squeeze_me=True)
	dat    = dat['synsets']
	labels, wnid = [],[]
	for dd in dat:
		labels.append(dd.ILSVRC2012_ID - 1)
		wnid.append(dd.WNID)
	labels = labels[0:1000]
	wnid   = wnid[0:1000]	
	return labels, wnid

##
# Read ILSVRC Data
class ILSVRC12Reader:
	def __init__(self, caffeDir='/work4/pulkitag-code/pkgs/caffe-v2-2'):
		labelFile  = '/data1/pulkitag/ILSVRC-2012-raw/devkit-1.0/data/ILSVRC2012_validation_ground_truth.txt'
		metaFile      = '/data1/pulkitag/ILSVRC-2012-raw/devkit-1.0/data/meta.mat'
		self.imFile_     = '/data1/pulkitag/ILSVRC-2012-raw/256/val/ILSVRC2012_val_%08d.JPEG'
		self.count_      = 0

		#Load the groundtruth labels from Imagenet
		fid  = open(labelFile)
		data = fid.readlines()
		valLabels = [int(i)-1 for i in data] #-1 for python formatting 
		
		#Load the Synsets
		synFile = os.path.join(caffeDir, 'data/ilsvrc12/synsets.txt')
		fid     = open(synFile, 'r')
		self.synsets_ = [s.strip() for s in fid.readlines()]
		fid.close()

		#Align the Imagenet Labels to the Synset order on which the BVLC reference model was trained.
		modLabels, modWnid = mapILSVRC12_labels_wnids(metaFile)
		self.labels_ = np.zeros((len(valLabels),)).astype(np.int)
		for (i,lb) in enumerate(valLabels):
			lIdx = modLabels.index(lb)
			syn  = modWnid[lIdx]
			sIdx = self.synsets_.index(syn)
			self.labels_[i] = sIdx

		#Load the synset words
		synWordFile = os.path.join(caffeDir, 'data/ilsvrc12/synset_words.txt')
		fid         = open(synWordFile, 'r')
		data        = fid.readlines()
		self.words_ = {}
		for l in data:
			synNames = l.split()
			syn      = synNames[0]
			words    = [w for w in synNames[1:]]
			self.words_[syn] = words
		fid.close()


	def reset(self):
		self.count_ = 0

	def set_count(self, count):
		self.count_ = count

	def read(self):
		imFile = self.imFile_ % (self.count_ + 1)
		im     = mp.caffe.io.load_image(imFile)
		lb     = self.labels_[self.count_]	
		syn    = self.synsets_[lb]
		words  = self.words_[syn]
		self.count_ += 1
		return im, lb, syn, words

	def word_label(self, lb):
		return self.words_[self.synsets_[lb]]	


def read_layerdefs_from_proto(fName):
	'''
		Reads the definitions of layers from a protoFile
	'''
	fid = open(fName,'r')
	lines = fid.readlines()
	fid.close()

	layerNames, topNames  = [], []
	layerDef   = []
	stFlag     = True
	layerFlag  = False
	tmpDef = []
	for (idx,l) in enumerate(lines):
		isLayerSt = 'layer' in l
		if isLayerSt:
			if stFlag:
				stFlag = False
				layerNames.append('init')
				topNames.append('')
			else:
				layerNames.append(layerName)
				topNames.append(topName)
			layerDef.append(tmpDef)
			tmpDef    = []
			layerName, topName = mp.find_layer_name(lines[idx:])

		tmpDef.append(l)
		
	return layerNames, topNames, layerDef

##
# Generate String for writing down a protofile for a layer. 
def get_layerdef_for_proto(layerType, layerName, bottom, numOutput=1, **kwargs):
	layerDef = co.OrderedDict()
	layerDef['name']  = '"%s"' % layerName
	layerDef['type']  = '"%s"' % layerType
	layerDef['bottom'] =	'"%s"' % bottom
	if layerType == 'InnerProduct':
		layerDef['top']    = '"%s"' % layerName
		layerDef['param']  = co.OrderedDict()
		layerDef['param']['lr_mult']    = '1'
		layerDef['param']['decay_mult'] = '1'
		paramDup = make_key('param', layerDef.keys())
		layerDef[paramDup] = co.OrderedDict()
		layerDef[paramDup]['lr_mult']    = '2'
		layerDef[paramDup]['decay_mult'] = '0'
		ipKey = 'inner_product_param'
		layerDef[ipKey]  = co.OrderedDict()
		layerDef[ipKey]['num_output'] = str(numOutput)
		layerDef[ipKey]['weight_filler'] = {}
		layerDef[ipKey]['weight_filler']['type'] = '"gaussian"'
		layerDef[ipKey]['weight_filler']['std']  = str(0.005)
		layerDef[ipKey]['bias_filler'] = {}
		layerDef[ipKey]['bias_filler']['type'] = '"constant"'
		layerDef[ipKey]['bias_filler']['value']  = str(0.)
	elif layerType=='Silence':
		#Nothing to be done
		a = True
	elif layerType=='Dropout':
		layerDef['top']    = '"%s"' % kwargs['top']
		layerDef['dropout_param'] = co.OrderedDict()
		layerDef['dropout_param']['dropout_ratio'] = str(kwargs['dropout_ratio'])	
	else:
		raise Exception('%s layer type not found' % layerType)
	'''	
	defStr = []
	defStr.append('layer { \n')
	defStr.append('  name: "%s" \n' % layerName)
	if layerType == 'InnerProduct':
		defStr.append('  type: "InnerProduct" \n')
		defStr.append('	 bottom: "%s" \n' % bottom)
		defStr.append('  top: "%s" \n' % layerName)
		defStr.append('  param { \n')
		defStr.append('    lr_mult: 1 \n')
		defStr.append('    decay_mult: 1 \n')
		defStr.append('  } \n')
		defStr.append('  param { \n')
		defStr.append('    lr_mult: 2 \n')
		defStr.append('    decay_mult: 0 \n')
		defStr.append('  } \n')
		defStr.append('  inner_product_param { \n')
		defStr.append('    num_output: %d \n' % numOutput)
		defStr.append('    weight_filler { \n')
		defStr.append('      type: "gaussian" \n')
		defStr.append('      std:  0.005 \n')
		defStr.append('    } \n')
		defStr.append('    bias_filler { \n')
		defStr.append('      type: "constant" \n')
		defStr.append('      value:  0. \n')
		defStr.append('    } \n')
		defStr.append('  } \n')
	if layerType == 'Silence'
		defStr.append('  type: "Silence" \n')
		defStr.append('	 bottom: "%s" \n' % bottom)
	else:
		raise Exception('%s layer type not found' % layerType)
	defStr.append('} \n')
	return defStr	
	'''
	return layerDef


def find_in_line(l, findType):
	'''
		Helper fucntion for process_debug_log - helps find specific things in the line
	'''
	if findType == 'iterNum':
		assert 'Iteration' in l, 'Iteration word must be present in the line'
		ss  = l.split()
		idx = ss.index('Iteration')
		return int(ss[idx+1][0:-1])
	elif findType == 'layerName':
		assert 'Layer' in l, 'Layer word must be present in the line'
		ss = l.split()
		idx = ss.index('Layer')
		return ss[idx+1][:-1]

	else:
		raise Exception('Unrecognized findType')


def process_debug_log(logFile, setName='train'):
	'''
		A debug log contains information about the Forward, Backward and Update Stages
		The debug file contains infromation in the following format
		Forward Pass
		Backward Pass
		#The loss of the the network (Since, backward pass has not been used to update,
		 the loss corresponds to previous iteration. 
		Update log. 
	'''

	assert setName in ['train','test'], 'Unrecognized set-name'

	fid = open(logFile, 'r')
	lines = fid.readlines()	

	#Find the layerNames	
	layerNames = []
	for (count, l) in enumerate(lines):
		if 'Setting up' in l:
			name = l.split()[-1]
			if name not in layerNames:
				layerNames.append(name)
		if 'Solver scaffolding' in l:
			break
	lines = lines[count:] 

	#Start collecting the required stats. 
	iters  = []
	
	#Variable Intialization
	allBlobOp, allBlobParam, allBlobDiff = {}, {}, {}
	blobOp, blobParam, blobDiff = {}, {}, {}
	for name in layerNames:
		allBlobOp[name], allBlobParam[name], allBlobDiff[name] = [], [], []
		blobOp[name], blobParam[name], blobDiff[name] = [], [], []

	#Only when seeFlag is set to True, we will collect the required data stats.  
	seeFlag    = False
	updateFlag = False
	appendFlag = False
	for (count, l) in enumerate(lines):
		#Find the start of a new and relevant iteration  
		if 'Iteration' in l:
			if setName=='train':
				if not ('lr' in l):
					seeFlag    = False
					continue
			else:
				if 'Testing' not in l:
					seeFlag = False
					continue
			
			seeFlag = True
			iterNum = find_in_line(l, 'iterNum')
			iters.append(iterNum)	
			if len(iters) > 1:
				#print len(iters), appendFlag
				if appendFlag:
					for name in layerNames:
						allBlobOp[name].append(blobOp[name])
						allBlobParam[name].append(blobParam[name])
						allBlobDiff[name].append(blobDiff[name])
						blobOp[name], blobParam[name], blobDiff[name] = [], [], []
					appendFlag = False
	
		#Find the data stored by the blobs
		if seeFlag and 'Forward' in l:
			appendFlag = True
			lName = find_in_line(l, 'layerName')	
			if 'top blob' in l:	
				blobOp[lName].append(float(l.split()[-1]))
			if 'param blob' in l:
				blobParam[lName].append(float(l.split()[-1]))

		#Find the back propogated diffs
		if seeFlag and ('Backward' in l) and ('Layer' in l):
			appendFlag = True
			lName = find_in_line(l, 'layerName')
			if 'param blob' in l:
				blobDiff[lName].append(float(l.split()[-1]))

	fid.close()
	for name in layerNames:
		data = [np.array(a).reshape(1,len(a)) for a in allBlobOp[name]]
		allBlobOp[name] = np.concatenate(data, axis=0)
		data = [np.array(a).reshape(1,len(a)) for a in allBlobParam[name]]
		allBlobParam[name] = np.concatenate(data, axis=0)
		data = [np.array(a).reshape(1,len(a)) for a in allBlobDiff[name]]
		allBlobDiff[name] = np.concatenate(data, axis=0)

	return allBlobOp, allBlobParam, allBlobDiff, layerNames, iters


def plot_debug_subplots_(data, subPlotNum, label, col, fig, 
					numPlots=3):
	'''
		Helper function for plot_debug_log
		data: N * K where N is the number of points and K 
					is the number of outputs
		subPlotNum: The subplot Number
		label     : The label to put
		col       : The color of the line
		fig       : The figure to use for plotting
		numPlots  : number of plots
	'''
	plt.figure(fig.number)
	if data.ndim > 0:
		data = np.abs(data)
		'''
		#print data.shape
		#if data.ndim > 1:
			data = np.sum(data, axis=tuple(range(1,data.ndim)))
		#print label, data.shape
		'''
		for i in range(data.shape[1]):
			plt.subplot(numPlots,1,subPlotNum + i)
			print label + '_blob%d' % i
			plt.plot(range(data.shape[0]), data[:,i], 
								label=label +  '_blob%d' % i, color=col)
			plt.legend()

##
# Plots the weight and features values from a debug enabled log file. 
def plot_debug_log(logFile, setName='train', plotNames=None):
	blobOp, blobParam, blobDiff, layerNames, iterNum = process_debug_log(logFile, setName=setName)
	if plotNames is not None:
		for p in plotNames:
			assert p in layerNames, '%s layer not found' % p
	else:
		plotNames = [l for l in layerNames]

	colIdx = np.linspace(0,255,len(plotNames)).astype(int)
	plt.ion()
	print colIdx
	print layerNames
	for count,name in enumerate(plotNames):
		fig = plt.figure()
		col = plt.cm.jet(colIdx[count])
		numPlots = blobOp[name].shape[1] + blobParam[name].shape[1] + blobDiff[name].shape[1]
		plot_debug_subplots_(blobOp[name], 1, name + '_data', col,
												 fig, numPlots=numPlots)
		plot_debug_subplots_(blobParam[name], 1 + blobOp[name].shape[1],
										name + '_weights', col, fig, numPlots=numPlots)
		plot_debug_subplots_(blobDiff[name],
							 1 + blobOp[name].shape[1] + blobParam[name].shape[1],
							 name + '_diff', col,  fig, numPlots=numPlots)
	plt.show()	


##
# Get useful caffe paths
#
# Set the paths over here for using the utils code. 
def get_caffe_paths():
	paths  = {}
	paths['caffeMain']   = '/work4/pulkitag-code/pkgs/caffe-v2-2' 
	paths['tools']       = os.path.join(paths['caffeMain'], 'build', 'tools')
	paths['pythonTest']  = os.path.join(paths['caffeMain'], 'python', 'test')
	return paths


def give_run_permissions_(fileName):
	args = ['chmod u+x %s' % fileName]
	subprocess.check_call(args,shell=True)


##
#Make a new key if there are duplicates.
def make_key(key, keyNames, dupStr='_$dup$'):
	'''
		The new string is made by concatenating the dupStr,
		until a unique name is found. 
	'''
	if key not in keyNames:
		return key
	else:
		key = key + dupStr
		return make_key(key, keyNames, dupStr)

##
# If the key has dupStr, strip it off. 
def strip_key(key, dupStr='_$dup$'):
	if dupStr in key:
		idx = key.find(dupStr)
		key = key[:idx]	
	return key 


##
# Extracts message parameters of a .prototxt from a list of strings. 
def get_proto_param(lines):
	#The String to use in case of duplicate names
	dupStr    = '_$dup$'
	data      = co.OrderedDict()
	braCount  = 0
	readFlag  = True
	i         = 0
	while readFlag:
		if i>=len(lines):
			break
		l = lines[i]
		#print l.strip()
		if l in ['', '\n']:
			#Continue if empty line encountered.
			print 'Skipping empty line: %s ' % l.strip()
		elif '#' in l:
			#In case of comments
			print 'Ignoring line: %s' % l.strip()
		elif '{' in l and '}' in l:
			raise Exception('Reformat the file, both "{" and "}" cannot be present on the same line %s' % l)
		elif '{' in l:
			name       = l.split()[0]
			if '{' in name:
				assert name[-1] == '{'
				name = name[:-1]
			name       = make_key(name, data.keys(), dupStr=dupStr)
			data[name], skipI = get_proto_param(lines[i+1:]) 
			braCount += 1
			i        += skipI
		elif '}' in l:
			braCount -= 1
		else:
			#print l
			splitVals = l.split()
			if len(splitVals) > 2:
				raise Exception('Improper format for: %s, l.split() should only produce 2 values' % l)
			name, val  = l.split()
			name       = name[:-1]
			name       = make_key(name, data.keys(), dupStr=dupStr)
			data[name] = val
		if braCount == -1:
			break
		i += 1
	return data, i

##
# Write the proto information into a file. 
def write_proto_param(fid, protoData, numTabs=0):
	'''
		fid :    The file handle to which data needs to be written.
		data:    The data to be written. 
		numTabs: Is for the proper alignment. 
	'''
	tabStr = '\t ' * numTabs 
	for (key, data) in protoData.iteritems():
		key = strip_key(key)
		if isinstance(data, dict):
			line = '%s %s { \n' % (tabStr,key)
			fid.write(line)
			write_proto_param(fid, data, numTabs=numTabs + 1)
			line = '%s } \n' % (tabStr)
			fid.write(line)
		else:
			line = '%s %s: %s \n' % (tabStr, key, data)
			fid.write(line)


class ProtoDef():
	'''
		Reads the architecture definition file and converts it into a nice, programmable format. 		
	'''
	ProtoPhases = ['TRAIN', 'TEST'] 
	def __init__(self, defFile):
		self.layers_ = {}
		self.layers_['TRAIN'] = co.OrderedDict()	
		self.layers_['TEST']  = co.OrderedDict()
		self.siameseConvert_  = False  #If the def has been convered to siamese or not. 
		fid   = open(defFile, 'r')
		lines = fid.readlines()
		i     = 0
		layerInit = False
		#Lines that are there before the layers start. 
		self.initData_ = []
		while True:
			l = lines[i]
			if not layerInit:
				self.initData_.append(l)
			if ('layers' in l) or ('layer' in l):
				layerInit = True
				layerName,_ = mp.find_layer_name(lines[i:])
				layerData, skipI  = get_proto_param(lines[i+1:])
				if layerData.has_key('include'):
					phase = layerData['include']['phase']
					assert phase in ['TRAIN', 'TEST'], '%s phase not recognized' % phase
					assert layerName not in self.layers_[phase].keys(), 'Duplicate LayerName Found'
					self.layers_[phase][layerName] = layerData
				else:
					#Default Phase is Train
					assert layerName not in self.layers_['TRAIN'].keys(),\
																 'Duplicate LayerName: %s found' % layerName
					self.layers_['TRAIN'][layerName] = layerData
				i += skipI
			i += 1
			if i >= len(lines):
				break
		#The last line of iniData_ is going to be "layer {", so remove it. 
		self.initData_ = self.initData_[:-1]

	##
	# Convert a network into a siamese network. 
	def make_siamese():
		print "WARNING: Only Convolution and InnerProduct layers are assumed to have params"
		print "To be completed"	

	##
	# Write the prototxt architecture file
	def write(self, outFile):
		with open(outFile, 'w') as fid:
			#Write Init Data
			for l in self.initData_:
				fid.write(l)
			#Write TRAIN/TEST Layers
			for (key, data) in self.layers_['TRAIN'].iteritems():
				fid.write('layer { \n')
				write_proto_param(fid, data, numTabs=0)
				fid.write('} \n')
				#Write the test layer if it is present.  
				if self.layers_['TEST'].has_key(key):
					fid.write('layer { \n')
					write_proto_param(fid, self.layers_['TEST'][key], numTabs=0)
					fid.write('} \n')
			#Write the layers in TEST which were not their in TRAIN
			testKeys = self.layers_['TEST'].keys()
			for key in testKeys:
				if key in self.layers_['TRAIN'].keys():
					continue
				else:
					fid.write('layer { \n')
					write_proto_param(fid, self.layers_['TEST'][key], numTabs=0)
					fid.write('} \n')

	##					
	def set_layer_property(self, layerName, propName, value, phase='TRAIN',  propNum=0): 
		'''
			layerName: Name of the layer in which the property is present.
			propName : Name of the property.
								 If there is a recursive property like, layer['data_param']['source'],
								 then provide a list.   
			value    : The value of the property. 
			phase    : The phase in which to make edits. 
			propNum  : Some properties like top can duplicated mutliple times in a layer, so which one.
		'''
		assert phase in ProtoDef.ProtoPhases, 'phase name not recognized'
		assert layerName in self.layers_[phase].keys(), '%s layer not found' % layerName
		if not isinstance(propName, list):
			#Modify the propName to account for duplicates
			propName = propName + '_$dup$' * propNum
			propName = [propName]
		else:
			if isinstance(propNum, list):
				assert len(propNum)==len(propName), 'Lengths mismatch'
				propName = [p + i * '_$dup$' for (p,i) in zip(propName, propNum)]
			else:
				assert propNum==0,'propNum is not appropriately specified'
		#Set the value
		ou.set_recursive_key(self.layers_[phase][layerName], propName, value)

	##
	def add_layer(self, layerName, layer, phase='TRAIN'):
		assert layerName not in self.layers_[phase].keys(), 'Layer already exists'
		self.layers_[phase][layerName] = layer	

	##
	def del_layer(self, layerName):
		for phase in self.layers_.keys():
			if not isinstance(layerName, list):
				layerName = [layerName]
			for l in layerName:
				if self.layers_[phase].has_key(l):
					del self.layers_[phase][l]


##
# Class for making the solver_prototxt
class SolverDef:
	def __init__(self):
		self.data_ = {}

	@classmethod
	def from_file(cls, inFile):
		self = cls()
		with open(inFile,'r') as f:
			lines = f.readlines()
			self.data_,_ = get_proto_param(lines)
		return self

	##
	# Add a property if not there, modifies if already exists
	def add_property(self, propName, value):
		if propName in self.data_.keys():
			self.set_property(propName, value)
		else:
			self.data_[propName] = value

	##
	# Delete Property
	def del_property(self, propName):
		assert propName in self.data_.keys(), '%s not found' % propName
		del self[propName]

	##
	# Get property
	def get_property(self, propName):
		assert propName in self.data_.keys(), '%s not found' % propName
		return self.data_[propName]

	##
	# Set property
	def set_property(self, propName, value): 
		assert propName in self.data_.keys()
		if not isinstance(propName, list):
			propName = [propName]
		ou.set_recursive_key(self.data_, propName, value)

	##
  # Write the solver file
	def write(self, outFile):
		with open(outFile, 'w') as fid:
			fid.write('# Autotmatically generated solver prototxt \n')
			write_proto_param(fid, self.data_, numTabs=0)

	
class ExperimentFiles:
	'''
		Used for writing experiment files in an easy manner. 
	'''
	def __init__(self, modelDir, defFile='caffenet.prototxt', 
							 solverFile='solver.prototxt', logFile='log.txt', 
							 runFile='run.sh', deviceId=0, repNum=None):
		'''
			modelDir   : The directory where model will be stored. 
			defFile    : The relative (to modelDir) path of architecture file.
			solverFile : The relative path of solver file
			logFile    : Relative path of log file 
			deviceId   : The GPU ID to be used.
			repNum     : If none - then no repeats, otherwise use repeats. 
		'''
		self.modelDir_ = modelDir
		if not os.path.exists(self.modelDir_):
			os.makedirs(self.modelDir_)
		self.solver_   = os.path.join(self.modelDir_, solverFile)
		self.log_      = os.path.join(self.modelDir_, logFile)
		self.def_      = os.path.join(self.modelDir_, defFile)
		self.run_      = os.path.join(self.modelDir_, runFile)
		self.paths_    = get_caffe_paths()
		self.deviceId_ = deviceId
		self.repNum_   = repNum
		#To Prevent the results from getting lost I will copy over the log files
		#into a result folder. 
		self.resultDir_ = os.path.join(self.modelDir_,'result_store')
		if not os.path.exists(self.resultDir_):
			os.makedirs(self.resultDir_)
		self.resultLog_ = os.path.join(self.resultDir_, logFile)

	##
	# Write script for training.  
	def write_run_train(self):
		with open(self.run_,'w') as f:
			f.write('#!/usr/bin/env sh \n \n')
			f.write('TOOLS=%s \n \n' % self.paths_['tools'])
			f.write('GLOG_logtostderr=1 $TOOLS/caffe train')
			f.write('\t --solver=%s' % self.solver_)
			f.write('\t -gpu %d' % self.deviceId_)
			f.write('\t 2>&1 | tee %s \n' % self.log_)
		give_run_permissions_(self.run_)

	##
	# Write test script
	def write_run_test(self, modelIterations, testIterations):
		'''
			modelIterations: Number of iterations of the modelFile. 
											 The modelFile Name is automatically extracted from the solver file.
			testIterations:  Number of iterations to use for testing.   
		'''
		snapshot = self.extract_snapshot_name() % modelIterations
		with open(self.run_,'w') as f:
			f.write('#!/usr/bin/env sh \n \n')
			f.write('TOOLS=%s \n \n' % self.paths_['tools'])
			f.write('GLOG_logtostderr=1 $TOOLS/caffe test')
			f.write('\t --weights=%s' % snapshot)
			f.write('\t --model=%s ' % self.def_)
			f.write('\t --iterations=%d' % testIterations)
			f.write('\t -gpu %d' % self.deviceId_)
			f.write('\t 2>&1 | tee %s \n' % self.log_)
		give_run_permissions_(self.run_)

	##
	# Initialiaze a solver from the file/SolverDef instance. 
	def init_solver_from_external(self, inFile):
		if isinstance(inFile, SolverDef):
			self.solDef_ = inFile
		else:
			self.solDef_ = SolverDef.from_file(inFile)
		self.solDef_.add_property('device_id', self.deviceId_)
		#Modify the name of the net
		self.solDef_.set_property('net', '"%s"' % self.def_)		

	##
	# Intialize the net definition from the file/ProtoDef Instance.
	def init_netdef_from_external(self, inFile):
		if isinstance(inFile, ProtoDef):
			self.netDef_ = inFile
		else: 
			self.netDef_ = ProtoDef(inFile)

	##
	# Write a solver file for making reps.
	def write_solver(self):
		'''
			Modifies the inFile to make it appropriate for running repeats
		'''
		if self.repNum_ is not None:
			#Modify snapshot name
			snapName   = self.solDef_.get_property('snapshot_prefix')
			snapName   = snapName[:-1] + '_rep%d"' % repNum
			self.solDef_.set_property('snapshot_prefix', snapName)
		
		self.solDef_.write(self.solver_)	

	##		
	def extract_snapshot_name(self):
		'''
			Find the name with which models are being stored. 
		'''
		snapshot   = self.solDef_.get_property('snapshot_prefix')
		#_iter_%d.caffemodel is added by caffe while snapshotting. 
		snapshot = snapshot[1:-1] + '_iter_%d.caffemodel'
		return snapshot

	##
	def write_netdef(self):
		self.netDef_.write(self.def_)

	##
	# Run the Experiment
	def run(self):
		cwd = os.getcwd()
		subprocess.check_call([('cd %s && ' % self.modelDir_) + self.run_] ,shell=True)
		os.chdir(cwd)
		shutil.copyfile(self.log_, self.resultLog_)		


class CaffeExperiment:
	def __init__(self, dataExpName, caffeExpName, expDirPrefix, snapDirPrefix,
							 defPrefix = 'caffenet', solverPrefix = 'solver',
							 logPrefix = 'log', runPrefix = 'run', deviceId = 0,
							 repNum = None):
		'''
			experiment directory: expDirPrefix  + dataExpName
			snapshot   directory: snapDirPrefix + dataExpName
			solver     file     : expDir + solverPrefix + caffeExpName
			net-def    file     : expDir + defPrefix    + caffeExpName
			log        file     : expDir + logPrefix    + caffeExpName
			run        file     : expDir + runPrefix    + caffeExpName 
		'''
		#Relevant directories. 
		self.dirs_  = {}
		self.dirs_['exp']  = os.path.join(expDirPrefix,  dataExpName)
		self.dirs_['snap'] = os.path.join(snapDirPrefix, dataExpName)  

		#Relevant files. 
		self.files_ = {}
		solverFile  = solverPrefix + '_' + caffeExpName + '.prototxt'
		logFile     = logPrefix    + '_' + caffeExpName + '.txt'
		runFile     = runPrefix    + '_' + caffeExpName + '.sh'
		defFile     = defPrefix    + '_' + caffeExpName + '.prototxt'
		self.files_['solver'] = os.path.join(self.dirs_['exp'], solverFile) 
		self.files_['netdef'] = os.path.join(self.dirs_['exp'], defFile) 
		self.files_['log']    = os.path.join(self.dirs_['exp'], logFile)
		self.files_['run']    = os.path.join(self.dirs_['exp'], runFile)

		#snapshot
		snapPrefix = defPrefix + '_' + caffeExpName 
		self.files_['snap'] = os.path.join(snapDirPrefix, dataExpName,
													snapPrefix + '_iter_%d.caffemodel')  
		self.snapPrefix_    = '"%s"' % os.path.join(snapDirPrefix, dataExpName, snapPrefix)		

		#Setup the experiment files.
		self.expFile_ = ExperimentFiles(self.dirs_['exp'], defFile = defFile,
											solverFile = solverFile, logFile = logFile, 
											runFile = runFile, deviceId = deviceId,
											repNum = repNum)

	##
	#initalize from solver file/SolverDef and netdef file/ProtoDef
	def init_from_external(self, solverFile, netDefFile):
		self.expFile_.init_solver_from_external(solverFile)
		self.expFile_.init_netdef_from_external(netDefFile)	
		#Set the correct snapshot prefix. 
		self.expFile_.solDef_.set_property('snapshot_prefix', self.snapPrefix_)

	##
	# init from self
	def init_from_self(self):
		self.expFile_.init_solver_from_external(self.files_['solver'])
		self.expFile_.init_netdef_from_external(self.files_['netdef'])	

	##
	def del_layer(self, layerName):
		self.expFile_.netDef_.del_layer(layerName) 

	## Set the property. 	
	def set_layer_property(self, layerName, propName, value, **kwargs):
		self.expFile_.netDef_.set_layer_property(layerName, propName, value, **kwargs)

	##
	def add_layer(self, layerName, layer, phase):
		self.expFile_.netDef_.add_layer(layerName, layer, phase)

	##
	def get_snapshot_name(self, numIter=10000):
		snapName = self.expFile_.extract_snapshot_name() % numIter
		return snapName

	# Make the experiment. 
	def make(self):
		if not os.path.exists(self.dirs_['exp']):
			os.makedirs(self.dirs_['exp'])
		if not os.path.exists(self.dirs_['snap']):
			os.makedirs(self.dirs_['snap'])
	 
		self.expFile_.write_netdef()
		self.expFile_.write_solver()
		self.expFile_.write_run_train()	

 
def make_experiment_repeats(modelDir, defPrefix,
									 solverPrefix='solver', repNum=0, deviceId=0, suffix=None, 
										defData=None, testIterations=None, modelIterations=None):
	'''
		Used to run an experiment multiple times.
		This is useful for testing different random initializations. 
		repNum       : The repeat. 
		modelDir     : The directory containing the defFile and solver file
		defPrefix    : The prefix of the architecture prototxt file. 
		solverPrefix : The prefix of the solver file. 
		deviceId     : The GPU device to use.
		suffix       : If a suffix is present it is added to all the files.
								   For eg solver.prototxt will become, solver_suffix.prototxt
		defData      : None or Instance of class ProtoDef which defined a protoFile
		testIterations: Number of test iterations to run
		modelIterations: The number of iterations of training for which model should be loaded 
	'''
	assert os.path.exists(modelDir), 'ModelDir %s not found' % modelDir

	#Make the directory for storing rep data. 
	repDir          = modelDir + '_reps'
	if not os.path.exists(repDir):
		os.makedirs(repDir)
	#Make the definition file. 
	if suffix is not None:
		defFile    = defPrefix + '_' + suffix + '.prototxt'
		solverFile = solverPrefix + '_' + suffix + '.prototxt'
		repSuffix  = '_rep%d_%s' % (repNum, suffix) 
	else:
		defFile    = defPrefix + '.prototxt'
		solverFile = solverPrefix + '.prototxt' 
		repSuffix  = '_rep%d' % repNum
 
	#Get Solver File Name
	solRoot, solExt = os.path.splitext(solverFile)

	repSol   = solRoot + repSuffix + solExt
	repDef   = defPrefix + repSuffix + '.prototxt'
	repLog   = 'log%s.txt' % (repSuffix)
	repRun   = 'run%s.sh'  % (repSuffix)
	repLogTest   = 'log_test%s.txt' % (repSuffix)
	repRunTest   = 'run_test%s.sh'  % (repSuffix)
			
	#Training Phase
	trainExp = ExperimentFiles(modelDir=repDir, defFile=repDef, solverFile=repSol,
						 logFile=repLog, runFile=repRun, deviceId=deviceId, repNum=repNum)
	trainExp.init_solver_from_external(os.path.join(modelDir, solverFile))
	if defData is None:
		trainExp.init_netdef_from_external(os.path.join(modelDir, defFile))
	else:
		trainExp.init_netdef_from_external(defData)
	trainExp.write_solver()
	trainExp.write_netdef()
	trainExp.write_run_train()
	
	#Test Phase	
	if testIterations is not None:
		assert modelIterations is not None	 
		testExp      = ExperimentFiles(modelDir=repDir, defFile=repDef, solverFile=repSol,
									 logFile=repLogTest, runFile=repRunTest, deviceId=deviceId, repNum=repNum)
		testExp.write_run_test(modelIterations, testIterations)

	return trainExp, testExp
