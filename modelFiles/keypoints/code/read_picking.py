import os
import cypbm
import h5py as h5
import matplotlib.pyplot as plt
import scipy
import scipy.misc as scm
import scipy.linalg as scl
import skimage.measure as skim
import numpy as np
import pdb

def get_classNames():
	pathName = '/data1/pulkitag/data_sets/amazon_picking_challenge/'
	dirNames = os.listdir(pathName)
	dirNames = [d for d in dirNames if os.path.isdir(pathName + d)]
	return dirNames

class PickDat:
	def __init__(self, className):
		self.classPath = '/data1/pulkitag/data_sets/amazon_picking_challenge/%s/' % className
		self.maskFile  = os.path.join(self.classPath, 'masks', 'NP%d_%d_mask.pbm')
		self.depthFile = os.path.join(self.classPath, 'NP%d_%d.h5')
		self.colFile   = os.path.join(self.classPath, 'NP%d_%d.jpg')
		self.poseFile  = os.path.join(self.classPath, 'poses','NP%d_%d_pose.h5')
		self.calibFile = os.path.join(self.classPath, 'calibration.h5')
		self.maskDat   = []
		self.colDat    = []
		self.depthDat  = []
		self.calibDat  = h5.File(self.calibFile,'r')

	def __del__(self):
		self.calibDat.close()

	def read(self, camNum, rotNum):
		self.camNum   = camNum
		self.rotNum   = rotNum
		#Get the Mask
		tmpMask       = np.transpose(cypbm.read(self.maskFile % (camNum, rotNum)),(1,0))
		tmpMask       = scipy.ndimage.filters.gaussian_filter((tmpMask).astype('float'),0.5) > 0
		cc            = skim.label(tmpMask)
		uniqComps     = np.unique(cc)
		assert len(uniqComps) >= 2, 'Something is wrong in obtaining the mask'
		compSz        = [-np.sum(cc==i) for i in uniqComps]
		sortIdx       = np.argsort(compSz)
		objMask       = np.zeros(tmpMask.shape).astype('bool')
		#Background is the biggest object, and the second largest is the object of interest
		objMask[cc==uniqComps[sortIdx[1]]] = True
		self.maskDat  = objMask
		self.maskDat  = scm.imresize(self.maskDat, (480,640), interp='nearest')
		self.maskDat  = self.maskDat.astype('bool')
		#Color Data
		self.colDat   = scm.imresize(plt.imread(self.colFile  % (camNum, rotNum)),(480,640),interp='bicubic')
		#Depth Im
		fid           = h5.File(self.depthFile % (camNum, rotNum),'r') 
		self.depthDat = fid['depth'][:]
		fid.close()
		#The rotation pose can be judged - based on the camera NP5 - as the calibration matrix is avail.
		pFid          = h5.File(self.poseFile % (5, rotNum),'r')
		self.poseDat  = pFid['H_table_from_reference_camera'][:]
		pFid.close()

	def get_crop(self):
		'''
			Returns the crop of the image
		'''
		#sum over the rows
		rowSum = np.sum(self.maskDat, axis=0)
		xnz    = np.nonzero(rowSum)
		colSum = np.sum(self.maskDat, axis=1)
		ynz    = np.nonzero(colSum)
		xmin,xmax   = min(xnz[0]), max(xnz[0])
		ymin,ymax   = min(ynz[0]), max(ynz[0])
		return xmin, xmax, ymin, ymax

	def get_square_crop(self):
		'''
			Square crop may be required when the aspect ratio needs to be preserved. 
		'''
		xmin, xmax, ymin, ymax = self.get_crop()
		X,Y = self.maskDat.shape
		if ymax - ymin < xmax - xmin:
			L      = (xmax - xmin) - (ymax - ymin)
			stride = L/2	
			ymin   = np.max([0, ymin - stride])
			ymax   = np.min([Y, ymax + stride])
		else: 
			L      = (ymax - ymin) - (xmax - xmin)
			stride = L/2	
			xmin   = np.max([0, xmin - stride])
			xmax   = np.min([X, xmax + stride])
		return xmin, xmax, ymin, ymax
		

	def colSeg(self, sqCrop=False, noBg=False):
		'''
			noBG: If true, then the background is removed. 
		'''
		if noBg:
			xmin,xmax,ymin,ymax = self.get_crop()			
			if sqCrop:
				xminSq,xmaxSq,yminSq,ymaxSq = self.get_square_crop()		
				mx = int(max(xmaxSq - xminSq, ymaxSq - yminSq))	
				colDat = (255*np.ones((mx, mx, 3))).astype(np.uint8)
				x1     = int(np.floor((xmin - xminSq)/2.0))
				x2     = int(x1 + (xmax - xmin))
				y1     = int(np.floor((ymin - yminSq)/2.0))
				y2     = int(y1 + (ymax - ymin))
				colDat[y1:y2,x1:x2,:] = self.colDat[ymin:ymax, xmin:xmax,:]
				#Pad the image in a special way by considering the last pixel row/col.
				if y1>0:
					padRow = (colDat[y1,x1:x2,:]).reshape(1,(x2-x1),3)
					colDat[0:y1,x1:x2,:] = np.tile(padRow, (y1,1,1))
				if y2 < mx:
					padRow = colDat[y2-1,x1:x2,:].reshape(1,(x2-x1),3)
					colDat[y2:mx,x1:x2,:] = np.tile(padRow, (mx-y2,1,1))
				if x1>0:
					padCol = (colDat[y1:y2,x1,:]).reshape((y2-y1),1,3)
					colDat[y1:y2,0:x1,:] = np.tile(padCol, (1,x1,1))
				if x2 < mx:
					padCol = (colDat[y1:y2,x2-1,:]).reshape((y2-y1),1,3)
					colDat[y1:y2,x2:mx,:] = np.tile(padCol, (1,mx-x2,1))
			else:
				colDat =  np.copy(self.colDat[ymin:ymax, xmin:xmax])
		else:
			if sqCrop:
				xmin,xmax,ymin,ymax = self.get_square_crop()			
			else:
				xmin,xmax,ymin,ymax = self.get_crop()			
			colDat =  np.copy(self.colDat[ymin:ymax, xmin:xmax])
		return colDat


	def depthSeg(self, sqCrop=False):
		if sqCrop:
			xmin,xmax,ymin,ymax = self.get_square_crop()			
		else:
			xmin,xmax,ymin,ymax = self.get_crop()			
		return np.copy(self.depthDat[ymin:ymax, xmin:xmax])


class objPair:
	def __init__(self, obj1, obj2):
		self.obj1 = obj1
		self.obj2 = obj2

	def transformCol(self):
		'''
			Find the transformation which will tranform object 2 to the same pose as object 1. 
		'''
		#Transform of im1 from the frame of NP5
		tMat1 = self.obj1.calibDat['H_NP%d_from_NP5' % self.obj1.camNum][:]
		#Transfrom the pose of im2 to im1 (in the frame of view of NP5)
		tMat2 = np.dot(scl.inv(self.obj2.poseDat), self.obj1.poseDat)
		#Transform im2 into the view of camera NP5
		tMat2 = np.dot(scl.inv(self.obj2.calibDat['H_NP%d_from_NP5' % self.obj2.camNum][:]), tMat2)
		self.transformColMat = np.dot(tMat1, tMat2)
