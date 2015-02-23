import h5py as h5
import numpy as np

def ims2hdf5(im, labels, batchSz, batchPath, isColor=True, batchStNum=1, isUInt8=True):
	'''
		Converts an image dataset into hdf5
	'''
	dType = im.dtype
	if isUInt8:
		assert im.dtype==np.uint8, 'Images should be in uint8'
		h5DType = 'u1'
	else
		assert im.dtype==np.float32, 'Images can either be uint8 or float32'		
		h5DType = 'f'

	if isColor:
		assert im.ndim ==4 
		N,ch,h,w = im.shape
		assert ch==3, 'Color images must have 3 channels'
	else:
		assert im.ndim ==3
		N,h,w    = im.shape

	count = batchStNum
	for i in range(0,N,batchSz):
		st      = i
		en      = min(st, st + batchSz)
		h5File    = batchPath + 'batch%d.h5' % count
		h5Fid     = h5.File(h5File, 'w')
		if isColor:
			imBatch = np.zeros((N, ch, h, w), dType) 
			imH5      = h5Fid.create_dataset('/data',(batchSz, ch, h, w), dtype=h5DType) 
		else:
			imBatch = np.zeros((N, h, w), dType)
			imH5      = h5Fid.create_dataset('/data',(batchSz, h, w), dtype=h5DType) 
		lbH5 = h5Fid.create_dataset('/labels', (batchSz,1,1,1), dtype='f')

		imH5[0:batchSz] = im[st:en]
		lbH5[0:batchSz] = labels[st:en]
		h5Fid.close()
		count += 1	

