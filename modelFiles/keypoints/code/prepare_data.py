import scipy.io as sio
import sys, os
import h5py
import numpy as np
import pdb
import subprocess

TOOL_DIR = '/work4/pulkitag-code/pkgs/caffe-v2-2/build/tools/'

rawDir = '/data1/pulkitag/keypoints/raw/';
h5Dir  = '/data1/pulkitag/keypoints/h5/';
dbDir  = '/data1/pulkitag/keypoints/leveldb_store/';
classNames = ['aeroplane','bicycle','bird','boat','bottle',\
							'bus','car','cat','chair','cow', \
							'diningtable','dog','horse','motorbike','person', \
							'potted-plant','sheep','sofa','train','tvmonitor'];


def create_data(h5DataFile, h5LabelFile,  numSamples, ims, views, clCount):	
	h,w,ch = 256,256,3
	imSz   = h * w * ch

	countSum = sum(clCount)
	clCount  = numSamples*(clCount/countSum) + 1
	clCount  = [int(c) for c in clCount]	
	numSamples = sum(clCount)
	print clCount
	
	fid   = h5py.File(h5DataFile, 'w')
	fidl  = h5py.File(h5LabelFile, 'w')
	images1 = fid.create_dataset("/images1", (numSamples * imSz,), dtype='u1')
	images2 = fid.create_dataset("/images2", (numSamples * imSz,), dtype='u1')
	labels  = fidl.create_dataset("/labels",(numSamples * 9,), dtype='f')

	np.random.seed(3)
	#For Randmizing the class order
	idxs = np.random.permutation(sum(clCount)) 

	count = 0
	for (i,num) in enumerate(clCount):
		#num - number of the images to be sampled from the class
		im    = ims[i]
		print im.shape
		view  = views[i]
		num   = int(num)
		clIdx1 = np.random.random_integers(0,im.shape[0]-1,num)
		clIdx2 = np.random.random_integers(0,im.shape[0]-1,num)
		print "For class %d, obtaining %d examples" % (i, num) 	
		#pdb.set_trace()
		for k in range(num):
			st  = idxs[count] * imSz 
			en  = st + imSz
			lSt = idxs[count] * 9
			lEn = lSt + 9
			try:
				images1[st:en] = im[clIdx1[k]].transpose((2,0,1)).flatten()
				images2[st:en] = im[clIdx2[k]].transpose((2,0,1)).flatten()
			except ValueError:
				print "Error Encountered"
				pdb.set_trace()		
			view1 = view[clIdx1[k]]
			view2 = view[clIdx2[k]]
			viewDiff = np.dot(view2, np.linalg.inv(view1)) 								
			labels[lSt:lEn] = viewDiff.flatten()
			count += 1
	fid.close()
	fidl.close()	



def h52db():
	imToolName = TOOL_DIR + 'hdf52leveldb_siamese_nolabels.bin'
	lbToolName = TOOL_DIR + 'hdf52leveldb_float_labels.bin'
	splits = ['val','train']

	for s in splits:
		h5ImName = h5Dir + '%s_images.hdf5' % s
		dbImName = dbDir + '%s_images_leveldb' % s	 
		subprocess.check_call(['%s %s %s' % (imToolName, h5ImName, dbImName)],shell=True)
		h5LbName = h5Dir + '%s_labels.hdf5' % s
		dbLbName = dbDir + '%s_labels_leveldb' % s	 
		subprocess.check_call(['%s %s %s' % (lbToolName, h5LbName, dbLbName)],shell=True)


if __name__ == "__main__":
	trainClass = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',\
								'car','cat','chair','cow','diningtable','dog', \
								'person','sheep','sofa','train']
	valClass =  [cl for cl in classNames if cl not in trainClass]
	print "Number of train,val classes: (%d,%d)" % (len(trainClass), len(valClass))
	numTrain = 1e+6
	numVal   = 5e+4
	print "Loading Data ..."
	trainIm = []
	valIm   = []
	trainCount, valCount = [],[]
	trainView = []
	valView   = []
	for cl in classNames:
		fileName = rawDir + cl + '.mat'
		data     = h5py.File(fileName)
		ims      = np.transpose(data['ims'], axes=(3,2,1,0))
		view     = np.transpose(data['view'], axes=(2,1,0))
		clCount  = np.array(data['clCount'])
		if cl in trainClass:
			trainIm.append(ims)
			trainView.append(view)
			trainCount.append(clCount)
		else:
			valIm.append(ims)
			valView.append(view)
			valCount.append(clCount)

	print "Making training Data .."
	trainDataH5  = h5Dir + 'train_images.hdf5'
	trainLabelH5 = h5Dir + 'train_labels.hdf5' 
	create_data(trainDataH5, trainLabelH5, numTrain, trainIm, trainView, trainCount)

	print "Making Validation Data.."
	valDataH5   = h5Dir + 'val_images.hdf5'
	valLabelH5  = h5Dir + 'val_labels.hdf5'
	#create_data(valDataH5, valLabelH5,  numVal, valIm, valView, valCount)
	
