import numpy as np
import scipy.io as sio
import os

def get_paths():
	paths = {}
	paths['data']    = '/data1/pulkitag/data_sets/pascal_3d/PASCAL3D+_release1.1/'
	paths['annData'] = os.path.join(paths['data'], 'Annotations', '%s_%s') #Fill with class_dataset

##
# Reads a file and finds the relevant data
def read_file(fileName):
	dat = sio.loadmat(fileName, squeeze_me=True, struct_as_record=False)
	dat = dat['record'] 
 
