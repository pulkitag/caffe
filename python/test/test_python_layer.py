import my_pycaffe as mp
import numpy as np
import caffe

def test():
	solFile  = 'test_solver.prototxt'
	sol      = mp.MySolver.from_file(solFile)
	pLayer   = sol.get_layer_pointer('data')
	pLayer.windowFile = '/data1/pulkitag/data_sets/pascal_3d/my/window_file_val.txt'
	pLayer.cache_data()
	return sol



