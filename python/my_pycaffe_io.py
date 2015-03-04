import h5py as h5
import numpy as np
import caffe
import pdb
import os
import lmdb

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


class dbSaver:
	def __init__(self, dbName, isLMDB=True):
		self.db    = lmdb.open(dbName, map_size=int(1e12))
		self.count = 0

	def __del__(self):
		self.db.close()

	def add_batch(self, ims, labels=None, imAsFloat=False, svIdx=None):
		'''
			Assumes ims are numEx * ch * h * w
			svIdx: Allows one to store the images randomly. 
		'''
		self.txn   = self.db.begin(write=True)
		if labels is not None:
			assert labels.dtype == np.int or labels.dtype==np.long
		else:
			labels = np.zeros((N,)).astype(np.int)

		if svIdx is not None:
			itrtr = zip(svIdx, ims, labels)
		else:
			itrtr = zip(range(count, count + ims.shape[0]), ims, labels)

		for idx, im, lb in itrtr:
			if not imAsFloat:
				im    = im.astype(np.uint8)
			imDat = caffe.io.array_to_datum(im, label=lb)
			aa    = imDat.SerializeToString()
			print idx, lb, len(aa), imDat.channels, imDat.height, imDat.width
			self.txn.put('{:0>10d}'.format(idx), imDat.SerializeToString())
		self.txn.commit()
		self.count = self.count + ims.shape[0]

	def close(self):
		self.db.close()


class dbReader:
	def __init__(self, dbName, isLMDB=True, readahead=True):
		#For large LMDB set readahead to be False
		self.db  = lmdb.open(dbName, readonly=True, readahead=readahead)
		self.txn = self.db.begin(write=False) 
		self.cursor = self.txn.cursor()		
		self.itr    = self.cursor.iternext()

	def __del__(self):
		self.txn.commit()
		self.db.close()
		
	def read_next(self):
		key, dat = self.itr.next()
		datum  = caffe.io.caffe_pb2.Datum()
		datStr = datum.FromString(dat)
		data   = caffe.io.datum_to_array(datStr)
		return data

	def close(self):
		self.txn.commit()
		self.db.close()


def save_lmdb_images(ims, dbFileName, labels=None, asFloat=False):
	'''
		Assumes ims are numEx * ch * h * w
	'''
	N,_,_,_ = ims.shape
	if labels is not None:
		assert labels.dtype == np.int or labels.dtype==np.long
	else:
		labels = np.zeros((N,)).astype(np.int)

	db = lmdb.open(dbFileName, map_size=int(1e12))
	with db.begin(write=True) as txn:
		for (idx, im) in enumerate(ims):
			if not asFloat:
				im    = im.astype(np.uint8)
			imDat = caffe.io.array_to_datum(im, label=labels[idx])
			txn.put('{:0>10d}'.format(idx), imDat.SerializeToString())
	db.close()


