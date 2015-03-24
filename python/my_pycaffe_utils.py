import my_pycaffe as mp
import my_pycaffe_io as mpio
import numpy as np
import pdb
import os
import scipy.io as sio
import matplotlib.pyplot as plt

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


def get_layerdef_for_proto(layerType, layerName, bottom, numOutput=1):
	'''
		Gives the text for writing a prot-def file. 
	'''
	defStr = []
	if layerType == 'InnerProduct':
		defStr = []
		defStr.append('layer { \n')
		defStr.append('  name: "%s" \n' % layerName)
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
		defStr.append('} \n')
	else:
		print '%s layer type not found' % layerType

	return defStr	


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
	
