import numpy as np
import h5py
import caffe

def main():
	'''
	'''
	caffe.set_mode_gpu()
	net = caffe.Net('random_noise_net.prototxt', caffe.TEST)
	data    = {}
	data['data'] = np.ones((1,1,28,28))
	op  = net.forward_all(blobs=['data', 'rn1'],**data)
	print op.keys()
	return op, net

if __name__ == '__main__':
	main()	
