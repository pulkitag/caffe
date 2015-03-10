import my_pycaffe as mp
import my_pycaffe_io as mpio
import numpy as np
import pdb

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


	
