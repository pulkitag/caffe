from subprocess import Popen, PIPE
import subprocess
import os
import sys

def modify_source(lines, traindb, testdb):
	count = 0
	dataMode = False
	for (i,l) in enumerate(lines):
		if 'data_param' in l:
			dataMode = True
		if dataMode and 'source' in l:
			if count == 0:
				newLine = 'source: "%s"' % traindb
				lines[i] = newLine
			if count == 1:
				newLine = 'source: "%s"' % testdb
				lines[i] = newLine

		if dataMode and '}' in l:
			dataMode = False
			count += 1
			if count == 2:
				break

		if count ==1 and 'include' in l and 'phase' in l:
			assert 'TRAIN' in l

	return lines


def write_file(lines, fileName):
	with open(fileName,'w') as f:
		for l in lines:
			f.write('%s' % l)


def read_file(fileName):
	with open(fileName,'r') as f:
		lines = f.readlines()
	return lines


def modify_snapshot_prefix(lines, prefix):
	for (i,l) in enumerate(lines):
		if 'snapshot_prefix' in l:
			newLine = 'snapshot_prefix: "%s"' % prefix
			lines[i] = newLine
			break
	return lines


def modify_solver_file(lines, baseLr=None, weightDecay=None, net=None, snapPrefix=None):
	if not snapPrefix==None:
		modify_snapshot_prefix(lines, snapPrefix)

	for (i,l) in enumerate(lines):
		if not baseLr==None and 'base_lr' in l:
			lines[i] = 'base_lr: %f' % baseLr
		if not weightDecay==None and 'weight_decay' in l:
			lines[i] = 'weight_decay: %f' % weight_decay
		if not net==None and 'net:' in l:
			lines[i] = 'net: "%s"' % net
	return lines
		
	
def get_leveldb_names(trainStr, valStr):
	dbDir = '/data1/pulkitag/mnist_rotation/leveldb_store/'
	trainDb = dbDir + 'mnist_rotation_train_' + trainStr + '_leveldb'
	valDb   = dbDir + 'mnist_rotation_val_' + valStr + '_leveldb'
	return trainDb, valDb


def get_hdf5_names(trainStr, valStr):
	hdf5Dir = '/data1/pulkitag/mnist_rotation/'
	trainH5 = hdf5Dir + 'mnist_train_%s.hdf5' % trainStr
	valH5   = hdf5Dir + 'mnist_val_%s.hdf5' % valStr
	return trainH5, valH5


def get_names(numTrain, numVal, trainDigits, valDigits):
	trainStr = ''
	valStr   = ''
	for t in trainDigits:
		trainStr = trainStr + '%d_' % t
	for v in valDigits:
		valStr   = valStr   + '%d_'	% v

	trainStr = trainStr + '%dK' % int(numTrain/1000)
	valStr   = valStr   + '%dK' % int(numVal/1000)
	expStr = 'train_%s_val_%s' % (trainStr, valStr)
	
	snapDir = '/data1/pulkitag/snapshots/mnist_rotation/exp_%s/' % expStr
	if not os.path.exists(snapDir):
		os.makedirs(snapDir)
	snapPrefix = snapDir + 'mnist_'

	return trainStr, valStr, expStr, snapPrefix


def h52db_siamese(h5Name, dbName):
	print "Creating Leveldb from " + h5Name
	if not os.path.exists(dbName):
		args = ['../../../build/tools/hdf52leveldb_siamese.bin %s %s' % (h5Name, dbName)]
		subprocess.check_call(args,shell=True)
	else:
		print "Leveldb: %s already exists, skipping conversion." % dbName	


def give_run_permissions(fileName):
	args = ['chmod u+x %s' % fileName]
	subprocess.check_call(args,shell=True)


def write_run_file(runFile, solverFile, logFile):
	with open(runFile,'w') as f:
		f.write('#!/usr/bin/env sh \n \n')
		f.write('TOOLS=../../../build/tools \n \n')
		f.write('GLOG_logtostderr=1 $TOOLS/caffe train')
		f.write('\t --solver=%s' % solverFile)
		f.write('\t 2>&1 | tee %s \n' % logFile)
	give_run_permissions(runFile)


def write_run_test_file(runFile, defFile, modelFile,  logFile):
	with open(runFile,'w') as f:
		f.write('#!/usr/bin/env sh \n \n')
		f.write('TOOLS=../../../build/tools \n \n')
		f.write('GLOG_logtostderr=1 $TOOLS/caffe test')
		f.write('\t --model=%s' % defFile)
		f.write('\t --iterations=1000')
		f.write('\t --gpu=0')
		f.write('\t --weights=%s' % modelFile)
		f.write('\t 2>&1 | tee %s \n' % logFile)
	give_run_permissions(runFile)


def write_run_finetune_file(runFile, solverFile, modelFile,  logFile):
	with open(runFile,'w') as f:
		f.write('#!/usr/bin/env sh \n \n')
		f.write('TOOLS=../../../build/tools \n \n')
		f.write('GLOG_logtostderr=1 $TOOLS/caffe train ')
		f.write('\t --solver=%s'  % solverFile)
		f.write('\t --weights=%s' % modelFile)
		f.write('\t 2>&1 | tee %s \n' % logFile)
	give_run_permissions(runFile)


def make_experiment(numTrain=1e+6, numVal=1e+4, \
				trainDigits = [2, 4, 6, 7, 8, 9], valDigits = [0, 1, 3 ,5]):

	modelIter = 50000
	#Names
	trainStr, valStr, expStr, snapPrefix = get_names(numTrain, numVal, trainDigits, valDigits) 
	
	#Names of HDF5 files
	trainH5, valH5 = get_hdf5_names(trainStr, valStr)

	#Names of Leveldbs
	trainDb, valDb = get_leveldb_names(trainStr, valStr)

	#Convert HDF% into leveldb
	h52db_siamese(trainH5, trainDb)
	h52db_siamese(valH5, valDb)
	
	expDir = '../rotation_%s/' % expStr
	baseDir = '../base_files/'	
	if not os.path.exists(expDir):
		os.makedirs(expDir)
	
	defStr  = 'mnist_siamese_train_test.prototxt'
	fineStr = 'mnist_train_test_finetune.prototxt' 	

	#Rotation Training Net-Def
	defFile1 = baseDir + defStr
	defFile2 = expDir + defStr
	defLines = read_file(defFile1)
	defLines = modify_source(defLines, trainDb, valDb)
	write_file(defLines, defFile2) 

	#Finetune Net-Def
	fineFile1 = baseDir + fineStr
	fineFile2 = expDir + fineStr	
	fineLines = read_file(fineFile1)
	write_file(fineLines, fineFile2)

	#Rotation Trainin Solver File
	solvStr = 'mnist_siamese_solver.prototxt'
	solvFile1 = baseDir + solvStr
	solvFile2 = expDir  + solvStr	
	solvLines = read_file(solvFile1)
	solvLines = modify_snapshot_prefix(solvLines, snapPrefix)
	modelName = snapPrefix + '_iter_%d.caffemodel' % modelIter
	write_file(solvLines, solvFile2)

	#Finetune Solver File
	fSolvStr = 'mnist_finetune_solver.prototxt'
	fSolvFile1 = baseDir + fSolvStr
	fSolvFile2 = expDir  + fSolvStr
	snapDir    = '/data1/pulkitag/snapshots/mnist/finetune_rot/exp_%s/' % expStr
	if not os.path.exists(snapDir):
		os.makedirs(snapDir)		
	snapPrefixFine = snapDir + 'mnist_finetune_rot'
	fSolvLines = read_file(fSolvFile1)
	fSolvLines = modify_snapshot_prefix(fSolvLines, snapPrefixFine)
	write_file(fSolvLines, fSolvFile2)

	#Run file for Rotation training
	runFile = expDir + 'train_mnist_siamese.sh'
	write_run_file(runFile, solvStr, 'log.txt')

	#Run file for fine-tuning
	runFineFile = expDir + 'finetune_mnist_rots.sh'
	write_run_finetune_file(runFineFile, fSolvStr, modelName, 'log_finetune.txt')	
	
	#Test file Rotation
	runFile = expDir + 'test_rotations.sh'
	rotModel = snapPrefix + '_iter_%d.caffemodel' % modelIter
	write_run_test_file(runFile,defFile2, rotModel, 'log_test_rot.txt')

	#Test File Mnist	
	runFile = expDir + 'test_classify.sh'
	clfModel = snapPrefixFine + '_iter_%d.caffemodel' % modelIter
	write_run_test_file(runFile, fineFile2, clfModel, 'log_test.txt')


def run_experiment(numTrain=1e+6, numVal=1e+4, \
				trainDigits = [2, 4, 6, 7, 8, 9], valDigits = [0, 1, 3 ,5]):

	trainStr, valStr, expStr, snapPrefix = get_names(numTrain, numVal, trainDigits, valDigits) 
	expDir = '../rotation_%s/' % expStr

	#Run rotation learning
	runFile = expDir + 'train_mnist_siamese.sh'
	subprocess.check_call(['cd %s && ' % expDir + runFile],shell=True)

	#Test Rotations
	runFile = expDir + 'test_rotations.sh'
	subprocess.check_call(['cd %s && ' % expDir + runFile],shell=True)


	#Run fineuning
	runFineFile = expDir + 'finetune_mnist_rots.sh'
	subprocess.check_call(['cd %s && ' % expDir + runFineFile],shell=True)

	#Result finetuning
	runFineFile = expDir + 'test_classify.sh'
	subprocess.check_call(['cd %s && ' % expDir + runFineFile],shell=True)
	
	subprocess.check_call(['cd -'], shell=True)

