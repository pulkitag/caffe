import h5py as h5
import numpy as np
import pdb
import os

def ims2hdf5(im, labels, batchSz, batchPath, isColor=True, batchStNum=1, isUInt8=True, scale=None):
	'''
		Converts an image dataset into hdf5
	'''
	h5SrcFile = os.path.join(batchPath, 'h5source.txt')
	strFid    = open(h5SrcFile, 'w')

	dType = im.dtype
	if isUInt8:
		assert im.dtype==np.uint8, 'Images should be in uint8'
		h5DType = 'u1'
	else:
		assert im.dtype==np.float32, 'Images can either be uint8 or float32'		
		h5DType = 'f'

	if scale is not None:
		im = im * scale

	if isColor:
		assert im.ndim ==4 
		N,ch,h,w = im.shape
		assert ch==3, 'Color images must have 3 channels'
	else:
		assert im.ndim ==3
		N,h,w    = im.shape
		im       = np.reshape(im,(N,1,h,w))
		ch       = 1

	count = batchStNum
	for i in range(0,N,batchSz):
		st      = i
		en      = min(N, st + batchSz)
		if st + batchSz > N:
			break
		h5File    = os.path.join(batchPath, 'batch%d.h5' % count)
		h5Fid     = h5.File(h5File, 'w')
		imBatch = np.zeros((N, ch, h, w), dType) 
		imH5      = h5Fid.create_dataset('/data',(batchSz, ch, h, w), dtype=h5DType) 
		lbH5 = h5Fid.create_dataset('/label', (batchSz,1,1,1), dtype='f')
		imH5[0:batchSz] = im[st:en]
		lbH5[0:batchSz] = labels[st:en].reshape((batchSz,1,1,1))
		h5Fid.close()
		strFid.write('%s \n' % h5File)
		count += 1	
	strFid.close()
