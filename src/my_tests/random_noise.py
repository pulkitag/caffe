import numpy as np
import h5py
import caffe
import my_pycaffe as mp

def main():
	'''
	'''
	isGPU=True
	defFile = 'proto_files/random_noise_net.prototxt'
	net = mp.MyNet(defFile, testMode=True, isGPU=isGPU)
	op   = net.forward(blobs=['conv1', 'rn1'], noInputs=True)
	print "Forward Done"
	diff = net.backward(diffs=['conv1'], noInputs=True)
	print op.keys()
	return op, diff, net

if __name__ == '__main__':
	main()	
