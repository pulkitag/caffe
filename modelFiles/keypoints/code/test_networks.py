import my_pycaffe as mp
import numpy as np

expDir  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/keypoints/'
snapDir = '/data1/pulkitag/snapshots/keypoints/'
h5Dir   = '/data1/pulkitag/keypoints/h5/'

def get_def_files(exp, labelType, numIter=100000, imSz=128):
	expNameDef = expDir + 'exp/%s_%d_%s/' % (exp,imSz,labelType)
	expNameSnp = expDir + 'exp/%s_imSz%d_%s/' % (exp,imSz,labelType)
	defFileSiamese = expNameDef + 'keynet_siamese_deploy.prototxt'
	snapFile = snapDir + 'exp%s_lbl%s/' % (exp, labelType)
	snapFile = snapFile + 'keypoints_siamese_iter_%d.caffemodel'
	snapFile = snapFile % numIter
	return defFileSiamese, snapFile


def get_h5_files(exp, labelType, imSz=128, setName='val'):
	h5LbFile = h5Dir + '%s_labels_exp%s_lbl%s_imSz%d.hdf5'
	h5LbFile = h5LbFile % (setName, exp, labelType, imSz)
	h5ImFile = h5Dir + '%s_images_exp%s_imSz%d.hdf5'
	h5ImFile = h5ImFile % (setName, exp, imSz)
	return h5ImFile, h5LbFile 		


def main(expName='rigid', labelType='uniform20', imSz=128):
	h5ImFile, h5LbFile = get_h5_files(expName, labelType, imSz=imSz)
	defFile, netFile   = get_def_files(expName, labelType, imSz=imSz)

	confMat = mp.test_network_siamese_h5(h5ImFile, h5LbFile, defFile, netFile)	
