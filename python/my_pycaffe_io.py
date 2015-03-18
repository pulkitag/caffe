import h5py as h5
import numpy as np
import caffe
import pdb
import os
import lmdb

def ims2hdf5(im, labels, batchSz, batchPath, isColor=True, batchStNum=1, isUInt8=True, scale=None, newLabels=False):
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
		imH5[0:batchSz] = im[st:en]
		if newLabels:
			lbH5 = h5Fid.create_dataset('/label', (batchSz,), dtype='f')
			lbH5[0:batchSz] = labels[st:en].reshape((batchSz,))
		else: 
			lbH5 = h5Fid.create_dataset('/label', (batchSz,1,1,1), dtype='f')
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
			N      = ims.shape[0]
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
			self.txn.put('{:0>10d}'.format(idx), imDat.SerializeToString())
		self.txn.commit()
		self.count = self.count + ims.shape[0]

	def close(self):
		self.db.close()



class DoubleDbSaver:
	'''
		Useful for example when storing images and labels in two different dbs
	'''
	def __init__(self, dbName1, dbName2, isLMDB=True):
		self.dbs_ = []
		self.dbs_.append(dbSaver(dbName1, isLMDB=isLMDB))
		self.dbs_.append(dbSaver(dbName2, isLMDB=isLMDB))

	def __del__(self):
		for db in self.dbs_:
			db.__del__()

	def close(self):
		for db in self.dbs_:
			db.close()

	def add_batch(self, ims, labels=(None,None), imAsFloat=(False,False), svIdx=(None,None)):
		for (i,db) in enumerate(self.dbs_):
			im = ims[i]
			db.add_batch(ims[i], labels[i], imAsFloat=imAsFloat[i], svIdx=svIdx[i])	


class DbReader:
	def __init__(self, dbName, isLMDB=True, readahead=True):
		#For large LMDB set readahead to be False
		self.db_     = lmdb.open(dbName, readonly=True, readahead=readahead)
		self.txn_    = self.db_.begin(write=False) 
		self.cursor_ = self.txn_.cursor()		
		self.itr_    = self.cursor_.iternext()

	def __del__(self):
		self.txn_.commit()
		self.db_.close()
		
	def read_next(self):
		if self.cursor_.next():
			key, dat = self.itr_.next()
		else:
			return None, None
		datum  = caffe.io.caffe_pb2.Datum()
		datStr = datum.FromString(dat)
		data   = caffe.io.datum_to_array(datStr)
		label  = datStr.label
		return data, label

	def read_batch(self, batchSz):
		data, label = [], []
		for b in range(batchSz):
			dat, lb = self.read_next()
			if dat is None:
				break
			else:
				ch, h, w = dat.shape
				dat = np.reshape(dat,(1,ch,h,w))
				data.append(dat)
				label.append(lb)
		data  = np.concatenate(data[:])
		label = np.array(label)
		label = label.reshape((len(label),1))
		return data, label 
		 
	def get_label_stats(self, maxLabels):
		countArr  = np.zeros((maxLabels,))
		countFlag = True
		while countFlag:
			_,lb     = self.read_next()	
			if lb is not None:
				countArr[lb] += 1
			else:
				countFlag = False
		return countArr				

	def close(self):
		self.txn_.commit()
		self.db_.close()


class SiameseDbReader(DbReader):
	def get_next_pair(self, flipColor=True):
		imDat,label  = self.read_next()
		ch,h,w = imDat.shape
		assert np.mod(ch,2)==0
		ch = ch / 2
		imDat  = np.transpose(imDat,(1,2,0))
		im1    = imDat[:,:,0:ch]
		im2    = imDat[:,:,ch:2*ch]
		if flipColor:
			im1 = im1[:,:,[2,1,0]]
			im2 = im2[:,:,[2,1,0]]
		return im1, im2, label
			 

class DoubleDbReader:
	def __init__(self, dbNames, isLMDB=True, readahead=True):
		#For large LMDB set readahead to be False
		self.dbs_ = []
		for d in dbNames:
			self.dbs_.append(DbReader(d, isLMDB=isLMDB, readahead=readahead))	

	def __del__(self):
		for db in self.dbs_:
			db.__del__()
		
	def read_next(self):
		data = []
		for db in self.dbs_:
			dat,_ = db.read_next()
			data.append(dat)
		return data[0], data[1]

	def read_batch(self, batchSz):
		data = []
		for db in self.dbs_:
			dat,_ = db.read_batch(batchSz)
			data.append(dat)
		return data[0], data[1]
	
	def close(self):
		for db in self.dbs_:
			db.close()

	
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


