import my_pycaffe as mp
import my_pycaffe_utils as mpu
import h5py
import numpy as np
import pdb

def test_zf_saliency(stride=2, patchSz=5):
	defFile   = '/work4/pulkitag-code/pkgs/caffe-v2-2/modelFiles/mnist/hdf5_test/lenet.prototxt'
	modelFile,_ = mp.get_model_mean_file('lenet')
	net       = mp.MyNet(defFile, modelFile, isGPU=False)
	N         = net.get_batchsz()
	net.set_preprocess(chSwap=None, imageDims=(28,28,1), isBlobFormat=True)	

	h5File    = '/data1/pulkitag/mnist/h5store/test/batch1.h5' 
	fid       = h5py.File(h5File,'r')
	data      = fid['data']
	data      = data[0:N]

	#Do the saliency
	imSal, score  = mpu.zf_saliency(net, data, 10, 'ip2', patchSz=patchSz, stride=stride)	
	pdLabels      = np.argmax(score.squeeze(), axis=1)
	gtLabels      = fid['label']
	return data, imSal, pdLabels, gtLabels		
