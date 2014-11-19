import scipy.misc as scm
import pickle
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os

def load_images(setName = 'train'):
	dataPath = '/work4/pulkitag/data_sets/mnist/'
	if setName == 'train':
		dataFile = dataPath + 'trainImages.pkl'
	else:
		dataFile = dataPath + 'testImages.pkl'

	with open(dataFile,'r') as f:
		data = pickle.load(f)
		im     = data['im']
		label  = data['label']

	return im, label

def make_rotations(im, N, outFile, numBins=20):
    bins = np.linspace(-180,180,numBins,endpoint=False)
    fid  = h5py.File(outFile,'w')
    imSet1 = fid.create_dataset("/images1", (N*28*28,), dtype='i1')
    imSet2 = fid.create_dataset("/images2", (N*28*28,), dtype='i1')
    labels = fid.create_dataset("/labels",  (N,),       dtype='i1')
    theta1 = fid.create_dataset("/theta1",  (N,),       dtype='f')
    theta2 = fid.create_dataset("/theta2",  (N,),       dtype='f')
    
    
    numIm =  len(im)
    print "Number of images: %d" % numIm 
    for i in range(0,N):
        idx = random.randint(0,numIm-1)
        imC  = im[idx]
        ang1 = random.uniform(0,90)
        ang2 = random.uniform(0,90)
        sgn1 = random.random()
        sgn2 = random.random()
        if sgn1 > 0.5:
            sgn1 = -1
        else:
            sgn1 = 1
        if sgn2 > 0.5:
            sgn2 = -1
        else:
            sgn2 = 1

        ang1 = ang1 * sgn1
        ang2 = ang2 * sgn2
        im1  = scm.imrotate(imC, ang1)
        im2  = scm.imrotate(imC, ang2)
        theta = ang2 - ang1
        theta = np.where((bins >= theta) == True)[0]
        if len(theta)==0:
            theta = numBins - 1
        else:
            theta = theta[0]
        st = i*28*28
        en = st + 28*28
        imSet1[st:en] = im1.flatten()
        imSet2[st:en] = im2.flatten()
        labels[i] = theta
        theta1[i] = ang1
        theta2[i] = ang2

    fid.close()

def check_hdf5(fileName):
    f = h5py.File(fileName,'r')
    im1 = f['images1']
    im2 = f['images2']
    lbl = f['labels']
    theta1 = f['theta1']
    theta2 = f['theta2']

    nr = 28
    nc = 28

    numIm = int(im1.shape[0]/784)
    for i in range(0,10):
        idx = random.randint(0,numIm-1)
        st  = idx*nr*nc
        en  = st + nr*nc
        imA = im1[st:en]
        imB = im2[st:en]
        label = lbl[idx]

        plt.figure()
        plt.subplot(211)
        plt.title("Theta1: %f, Theta2: %f, Label: %d" % (theta1[idx], theta2[idx],label))
        plt.imshow(imA.reshape(nr,nc))
        plt.subplot(212)
        plt.imshow(imB.reshape(nr,nc))
        plt.savefig('tmp/%d.png' % i, bbox_inches='tight')

    f.close()

if __name__ == "__main__":
		#trainDigits = [2,4,6,7,8,9]
		#valDigits   = [0,1,3,5]
		trainDigits = [0,1,2,3,4,5,6,7,8,9]
		valDigits = [0,1,2,3,4,5,6,7,8,9]
		numTrain    = int(1e+5)
		numVal      = int(1e+4)
		if len(sys.argv) > 1:
			dirName = sys.argv[1]
			if not os.path.exists(dirName):
				os.makedirs(dirName)
		else:
			dirName = './'


		trainStr = ''
		valStr = ''
		for t in trainDigits:
			trainStr = trainStr + '%d_' % t
		for v in valDigits:
			valStr = valStr + '%d_' % v
	
		trainFile = 'mnist_train_%s%dK.hdf5' % (trainStr, int(numTrain/1000))
		valFile   = 'mnist_val_%s%dK.hdf5' % (valStr, int(numVal/1000))
		trainFile = dirName + trainFile
		valFile   = dirName + valFile

		isCreate = True
		if isCreate:
				#Get the data
				im,label    = load_images()
				trainIm = [im[i] for i in range(len(label)) if label[i] in trainDigits]
				valIm   = [im[i] for i in range(len(label)) if label[i] in valDigits]
				make_rotations(trainIm, numTrain, trainFile)
				make_rotations(valIm, numVal, valFile)
		else: 
				check_hdf5(valFile)

    
   
