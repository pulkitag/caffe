import struct
import numpy as np
import scipy.io as io
import pickle

digitsDir = '/work4/pulkitag/data_sets/mnist/'
isTrain = True
isTest  = False

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
