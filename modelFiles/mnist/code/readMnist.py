import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import struct
import numpy as np
import scipy.io as io
import pickle
import h5py
import subprocess

digitsDir  = '/data1/pulkitag/mnist/raw/'
leveldbDir = '/data1/pulkitag/mnist/leveldb_store/'
isTrain = False
isTest  = True

def load_images(fileName):
    f = open(fileName,'rb')
    magicNum = struct.unpack('>i',f.read(4))
    N = struct.unpack('>i',f.read(4))[0]
    nr = struct.unpack('>i',f.read(4))[0]
    nc = struct.unpack('>i',f.read(4))[0]

    print "Num Images: %d, numRows: %d, numCols: %d" % (N,nr,nc)
    im = np.zeros((N,nr,nc),dtype=np.uint8)
    for i in range(N):
			for r in range(nr):
				for c in range(nc):
					im[i,r,c] = struct.unpack('>B',f.read(1))[0]

    f.close()
    return im


def load_labels(labelFile):
    f = open(labelFile,'rb')
    magicNum = struct.unpack('>i',f.read(4))
    N = struct.unpack('>i',f.read(4))[0]
    
    print "Number of labels found: %d" % N
    label = np.zeros((N,1),dtype=np.uint8)
    for i in range(N):
        label[i] = struct.unpack('>B',f.read(1))[0]

    f.close()
    return label


def make_hdf5_classlimit(rawFileName,outFileName, numPerClass):
	'''
		Used a pickle file to make a hdf5 file, using numPerClass images per class
	'''
	data  = pickle.load(open(rawFileName,'r'))
	im    = data['im']
	label = data['label']
	cls   = np.unique(label)
	assert all(cls==range(len(cls)))

	np.random.seed(3)
	perm  = np.random.permutation(len(label))		
	idxs  = []
	clCount = np.zeros((len(cls),1))
	for idx in perm:
		lbl = label[idx]
		if clCount[lbl] < numPerClass:
			idxs.append(idx)
		clCount[lbl] += 1

	im    = im[idxs]
	label = label[idxs]
	N     = len(label)
	with h5py.File(outFileName,'w') as fid:
		imSet = fid.create_dataset("/images",(N*28*28,), dtype='u1')
		labelSet = fid.create_dataset("/labels",(N,), dtype='u1')
	 	imSet[0:N*28*28] = im.flatten()
		labelSet[0:N]    = label.flatten()	


def hdf52leveldb(h5File, dbFile):
	cmd = '../../../build/tools/hdf52leveldb.bin %s %s'
	cmd = cmd % (h5File, dbFile)
	subprocess.check_call([cmd],shell=True)


def make_leveldb(numPerClass):
	rawFile = digitsDir + 'trainImages.pkl'
	h5File  = leveldbDir + 'mnist_train_numcl%d.hdf5' % numPerClass
	dbFile  = leveldbDir + 'mnist_train_numcl%d_leveldb' % numPerClass
	make_hdf5_classlimit(rawFile, h5File, numPerClass)
	hdf52leveldb(h5File, dbFile)


def check_leveldb(dirName='./'):
	numFiles = 10
	fig   = plt.figure()
	for i in range(numFiles):
		filename = dirName + 'count%08d.txt' % i
		outName  = dirName + 'im%08d.png' % i
		with open(filename,'r') as f:
			lines = f.readlines()
			lines = [int(l) for l in lines]
			im    = np.array(lines).reshape((28,28))
			print np.max(im.flatten())
			plt.imshow(im)
			plt.savefig(outName,bbox_inches='tight')	


def check_leveldb_siamese(dirName='./'):
	'''
		Run leveldb_2_images.bin to generate the 
		txt files which store images
	'''
	numFiles = 10
	fig   = plt.figure()
	for i in range(numFiles):
		filename = dirName + 'count%08d.txt' % i
		outName  = dirName + 'im%08d.png' % i
		with open(filename,'r') as f:
			lines = f.readlines()
			lines = [int(l) for l in lines]
			im    = np.array(lines).reshape((2,28,28))
			print np.max(im.flatten())
			plt.subplot(2,1,1)
			plt.imshow(im[0])
			plt.subplot(2,1,2)
			plt.imshow(im[1])
			plt.savefig(outName,bbox_inches='tight')	


if __name__ == "__main__":
    if isTrain:
        fileName  = digitsDir + 'train-images-idx3-ubyte'
        labelFile = digitsDir + 'train-labels-idx1-ubyte' 
        trainFileName = digitsDir + 'trainImages.mat'
        pTrainFileName = digitsDir + 'trainImages.pkl'

    if isTest:
        fileName  = digitsDir + 't10k-images-idx3-ubyte'
        labelFile = digitsDir + 't10k-labels-idx1-ubyte'
        trainFileName  = digitsDir + 'testImages.mat'
        pTrainFileName = digitsDir + 'testImages.pkl'

    im    = load_images(fileName)
    label = load_labels(labelFile)
    io.savemat(trainFileName,{'im':im,'label':label})
    pickle.dump({'im':im, 'label': label},open(pTrainFileName,'wb'))
