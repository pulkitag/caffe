import numpy as np
import h5py
import caffe

def get_montage():
	'''
		Creates a montage image to verify the performance of the cross convolution layer. 
		Create a 3*3 montage
	'''
	h5File = '/data1/pulkitag/mnist/h5store/test/batch1.h5'
	fid    = h5py.File(h5File, 'r')
	data   = fid['data']
	data   = data[0:9]

	data    = np.transpose(data, (0,2,3,1))
	montage = np.zeros((87,87,1))
	count = 0 
	for i in range(0,3):
		for j in range(0,3):
			rSt = i*28+1
			rEn = rSt + 28
			cSt = j*28 + 1
			cEn = cSt + 28
			montage[rSt:rEn, cSt:cEn,:] = data[count,:,:,:]
			count = count + 1

	montage = np.tile(montage,(1,1,1,2))
	montage = np.transpose(montage, (0,3,1,2))
	return montage 


def run_montage():
	caffe.set_mode_gpu()
	net = caffe.Net('cross_conv_net.prototxt', caffe.TEST)
	montage = get_montage()
	data    = {}
	data['data'] = montage
	op  = net.forward_backward_all(**data)
	return op,montage,net
	
