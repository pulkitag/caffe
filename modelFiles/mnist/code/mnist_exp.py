import scipy.misc as scm
import pickle
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
import my_pycaffe_io as mpio
import my_pycaffe_utils as mpu
import pdb
import make_rotations as mr


def make_def_file(nw, isSiamese=True, baseFileStr='split_im.prototxt'):
	'''
		If is siamese then wait for the Concat layers - and make all layers until then siamese.
	'''
	baseDir  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/base_files/'
	baseFile = os.path.join(baseDir, baseFileStr)
	protoDef = mpu.ProtoDef(baseFile)

	if baseFileStr == 'split_im.prototxt':
		lastTop  = 'data'

	siameseFlag = isSiamese
	if siameseFlag:
		stream1, stream2 = [], []
	mainStream = []

	nameGen     = mpu.LayerNameGenerator()
	for l in nw:
		lType, lParam = l
		lName         = nameGen.next_name(lType)
		if lType == 'Concat':
			siameseFlag = False
			if not lParam.has_key('bottom2'):
				lParam['bottom2'] = lastTop + '_p'
		if siameseFlag:
			lDef, lsDef = mpu.get_siamese_layerdef_for_proto(lType, lName, lastTop, **lParam)
			stream1.append(lDef)
			stream2.append(lsDef)
		else:
			lDef = mpu.get_layerdef_for_proto(lType, lName, lastTop, **lParam)
			mainStream.append(lDef)
		lastTop = lName

	#Add layers
	mainStream = stream1 + stream2 + mainStream
	for l in mainStream:
		protoDef.add_layer(l['name'][1:-1], l)	

	return protoDef

##
# Generates a string to represent the n/w name

def nw2name(nw):
	nameGen     = mpu.LayerNameGenerator()
	nwName = []
	for l in nw:
		lType, lParam = l
		lName = nameGen.next_name(lType)
		if lType in ['InnerProduct', 'Convolution']:
			lName = lName + '-%d' % lParam['num_output']
			nwName.append(lName)
		elif lType in ['Concat', 'Dropout']:
			nwName.append(lName)
		else:
			pass
	nwName = ''.join(s + '_' for s in nwName)
	nwName = nwName[:-1]
	return nwName	


def design_network():
	nw      = [('InnerProduct',{'num_output': 200}),('ReLU',{}),('Concat',{'concat_dim':1}),
						 ('InnerProduct',{'num_output': 500}),('ReLU',{}), 
						 ('InnerProduct',{'num_output': 3}), ('EuclideanLoss',{'bottom2': 'label'})]

	print nw2name(nw)
	return make_def_file(nw)
















 
