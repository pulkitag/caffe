import numpy as np
import my_pycaffe_io as mpio
import rot_utils as ru
import vis_utils as vu
import os
import pdb
import matplotlib.pyplot as plt
import subprocess
import scipy.misc as scm

def get_paths():
	paths = {}
	dataDir = '/data1/pulkitag/data_sets/cities/SanFrancisco_dataset'
	imDir   = '/data0/pulkitag/data_sets/cities/SanFrancisco_dataset/images/'
	paths['dataDir']  = dataDir
	paths['imDir']    = imDir
	paths['imList']   = os.path.join(dataDir, 'list.txt')
	paths['pairList'] = os.path.join(dataDir, 'pairs.txt')
	paths['tarList']  = os.path.join(dataDir, 'data_tar_files.txt')  		
	return paths


def get_prms():
	prms = {}
	paths = get_paths()
	prms['paths'] = paths
	return prms

##
# Get the list of tar files for downloading the image data
def get_tar_list(prms):
	f = open(prms['paths']['tarList'], 'r')
	lines = f.readlines()
	f.close()
	tarNames = []
	for l in lines:
		dat = l.split('=')[1][1:-4].split('>')[0][:-1]
		tarNames.append(dat)
	return tarNames	

##
# Download the image data
def download_image_data(prms):
	tarNames = get_tar_list(prms)
	currDir = os.getcwd()
	os.chdir(prms['paths']['imDir'])
	for name in tarNames:
		print name
		subprocess.check_call(['wget -l 0 %s' % name] ,shell=True)	
	os.chdir(currDir)
	
##
# Read the list of images
def get_imnames(prms):
	imF     = open(prms['paths']['imList'], 'r')
	imLines = imF.readlines()
	imNames, blah, focal = [],[],[]
	for l in imLines:
		dat = l.split()
		imNames.append(os.path.join(prms['paths']['imDir'], dat[0]))
		blah.append(dat[0])
		focal.append(dat[0])

	return imNames, blah, focal


def read_pairs(prms):
	imNames,_,_ = get_imnames(prms)
	with open(prms['paths']['pairList'], 'r') as f:
		line  = f.readline()
		numIm, numPairs = int(line.split()[0]), int(line.split()[1])
		assert numIm == len(imNames), 'Lenght mismatch %d v/s %d' % (numIm, len(imNames))
		imName1, imName2 = [], []
		euler = []
		translation = []
		for count in range(numPairs):
			line = f.readline()
			lDat = line.split()
			#Image Ids
			imId1, imId2 = int(lDat[0]), int(lDat[1])
			#The rotation matrix. 
			rotMat = np.array(lDat[2:11]).astype(float).reshape((3,3))
			euls   = ru.mat2euler(rotMat)
			trans  = np.array(lDat[11:14]).astype(float)
			#Append the data
			imName1.append(imNames[imId1])
			imName2.append(imNames[imId2])
			euler.append(euls)
			translation.append(trans)

		return imName1, imName2, euler, translation


def vis_pairs(prms):
	imName1, imName2, euls, trans = read_pairs(prms)
	N = len(imName1)
	perm = np.random.permutation(N)	
	fig = plt.figure()
	plt.ion()
	imName1 = [imName1[i] for i in perm]
	imName2 = [imName2[i] for i in perm]
	euls    = [euls[i] for i in perm]
	trans   = [trans[i] for i in perm]
	titleStr = 'Trans: ' + '%.3f ' * 3 + 'Rot: ' + '%.3f ' * 3
	for (im1,im2,eu,tr) in zip(imName1, imName2, euls, trans):
		titleName = titleStr % (tuple(tr) + eu)
		im1 = scm.imread(im1)
		im2 = scm.imread(im2)
		vu.plot_pairs(im1, im2, fig, titleStr=titleName)
		cmd = raw_input()	
		if cmd == 'exit':
			return

