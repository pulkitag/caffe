import caffe
import numpy as np
import scipy.misc as scm
import my_pycaffe as mp
import matplotlib.pyplot as plt

##
# Test Python Layer 
class TestPythonLayer(caffe.Layer):
	'''
		Format of the text file.
		# Image_Name x1 y1 x2 y2 label
	''' 
	def setup(self, bottom, top):
		pass		
 
	def reshape(self, bottom, top):
		pass	

	def cache_data(self):
		fid   = open(self.windowFile, 'r')
		lines = fid.readlines()
		fid.close()
		self.numIm  = len(lines)
		self.imData = np.zeros((self.numIm, 256, 256, 3)).astype(np.uint8)
		for (i,l) in enumerate(lines):
			fileName = l.split()[0]
			self.imData[i] = scm.imresize(plt.imread(fileName),(256, 256))
			
	def forward(self, bottom, top):
		top[0].data[...] = self.imData[0:100]
	 
	def backward(self, bottom, top):
		pass

	 
