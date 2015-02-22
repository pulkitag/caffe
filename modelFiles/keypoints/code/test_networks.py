import my_pycaffe as mp
import numpy as np
import scipy.io as sio
import h5py

expDir  = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/keypoints/'
snapDir = '/data1/pulkitag/snapshots/keypoints/'
h5Dir   = '/data1/pulkitag/keypoints/h5/'
dbDir   = '/data1/pulkitag/keypoints/leveldb_store/'
svDir   = '/data1/pulkitag/keypoints/outputs/'


def get_def_files(exp, labelType, numIter=100000, imSz=128):
	expNameDef = expDir + 'exp/%s_%d_%s/' % (exp,imSz,labelType)
	expNameSnp = expDir + 'exp/%s_imSz%d_%s/' % (exp,imSz,labelType)
	defFileSiamese = expNameDef + 'keynet_siamese_deploy.prototxt'
	snapFile = snapDir + 'exp%s_lbl%s/' % (exp, labelType)
	snapFile = snapFile + 'keypoints_siamese_iter_%d.caffemodel'
	snapFile = snapFile % numIter
	meanFile = dbDir + 'keypoints_im%d_mean.txt' % imSz 
	return defFileSiamese, snapFile, meanFile


def get_h5_files(exp, labelType, imSz=128, setName='val'):
	h5LbFile = h5Dir + '%s_labels_exp%s_lbl%s_imSz%d.hdf5'
	h5LbFile = h5LbFile % (setName, exp, labelType, imSz)
	h5ImFile = h5Dir + '%s_images_exp%s_imSz%d.hdf5'
	h5ImFile = h5ImFile % (setName, exp, imSz)
	return h5ImFile, h5LbFile 		


def save_features(expName='rigid', labelType='uniform20', imSz=128, setName='test'):
	h5ImFile, h5LbFile = get_h5_files(expName, labelType, imSz=imSz, setName=setName)
	defFile, netFile, meanFile = get_def_files(expName, labelType, imSz=imSz)
	confMat,predLabels,gtLabels = mp.test_network_siamese_h5(h5ImFile, h5LbFile, defFile, netFile, ipLayerName='pair_data',outLblSz=1,lblType=labelType,meanFile=meanFile)

	fid =  h5py.File(h5LbFile,'r')
	indices = fid['indices']

	imIndices = np.reshape(indices,(-1,4),order='C')
	outFile = svDir + 'exp%s_lbl%s_preds.mat'
	outFile = outFile % (expName, labelType)
	sio.savemat(outFile, {'predLabels': predLabels, 'gtLabels': gtLabels,'imIndices': imIndices})	
	fid.close()
 

def main(expName='rigid', labelType='uniform20', imSz=128):
	h5ImFile, h5LbFile = get_h5_files(expName, labelType, imSz=imSz)
	defFile, netFile, meanFile = get_def_files(expName, labelType, imSz=imSz)

	maskLastLabel = False
	if labelType=='uniform20':
		outFeatSz = 20
	elif labelType=='kmedoids30_20':
		outFeatSz = 21
		maskLastLabel = True
	else:	
		print 'Label Type Not Recognized'
		return

	confMat,predLabels,gtLabels = mp.test_network_siamese_h5(h5ImFile, h5LbFile, defFile, netFile, ipLayerName='pair_data',outLblSz=1,lblType=labelType,meanFile=meanFile, outFeatSz=outFeatSz, maskLastLabel=maskLastLabel)
	
	if maskLastLabel:
		print 'maskLastLabel is True'
	return confMat	
