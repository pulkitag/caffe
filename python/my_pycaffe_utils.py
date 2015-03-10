import my_pycaffe as mp
import my_pycaffe_io as mpio
import numpy as np
import pdb
import os
import scipy.io as sio

def zf_saliency(net, imBatch, numOutputs, opName, ipName='data', stride=2, patchSz=11):
	'''
		Takes as input a network and some image im
		Produces the saliency map of im
		net is of type MyNet
	'''

	assert(np.mod(patchSz,2)==1), 'patchSz needs to an odd Num'
	p = int(np.floor(patchSz/2.0))

	#Transform the image
	imT = net.preprocess_batch(imBatch)

	#Find the Original Scores
	dataLayer = {}
	dataLayer[ipName] = imT
	origScore = np.copy(net.net.forward(**dataLayer)[opName])
	
	N,ch,nr,nc = imT.shape
	dType      = imT.dtype
	nrNew     = len(range(p, nr-p-1, stride))
	ncNew     = len(range(p, nc-p-1, stride)) 
	imSalient = np.zeros((N, numOutputs, nrNew, ncNew))
 
	for (imCount,im) in enumerate(imT):
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
				if count==N or (ir == nrNew-1 and ic == ncNew-1):
					dataLayer = {}
					dataLayer[ipName] = net.preprocess_batch(ims)
					scores = net.net.forward(**dataLayer)[opName]
					scores = origScore[imCount] - scores[0:N]
					scores = scores.reshape((N, numOutputs))
					for idx,coords in enumerate(imIdx):
						y, x = coords
						imSalient[imCount, :, y, x] = scores[idx,:].reshape(numOutputs,)
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

			
