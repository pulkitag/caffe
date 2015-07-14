##Analyzes mnist networks. 
import make_rotations as mr
import mnist_exp as me
import other_utils as ou
import plot_utils as pu
import pdb
import pickle
##
# This is the n/w that performed best on egomotion based supervision on the ICCV-15
def get_best_nw():
	nw = []
	nw.append([('Convolution',  {'num_output': 96,  'kernel_size': 3, 'stride': 1}), ('ReLU',{}),
						('Pooling', {'kernel_size': 3, 'stride': 2}),
						('Convolution',  {'num_output': 256, 'kernel_size': 3, 'stride': 1}), ('ReLU',{}),
						('Pooling', {'kernel_size': 3, 'stride': 2}),
						('Concat', {'concat_dim': 1}),
						('InnerProduct', {'num_output': 1000}), ('ReLU',{}), 
						('Dropout', {'dropout_ratio': 0.5}),
						])			 
	return nw

##
#The effect of pretraining with different number of examples. 
def effect_num_pretrain(isPlot=False):
	nTr = [1e+3, 1e+4, 1e+5, 1e+6, 1e+7]
	nTe = [100, 300, 1000]
	nw  = get_best_nw()
	acc = me.run_final_finetune(runTypes=['accuracy'], runNum=[0],
					numTrainEx=nTr, nw=nw)
	del acc['numExamples']
	nwName = me.nw2name(nw[0])
	redFn  = lambda x: 1 - x[0]
	acc2 = ou.conditional_select(acc, [None, 'top', None, nwName], reduceFn=redFn)
	if isPlot:
		keyFn1 = lambda x: 'nTrn%.01e' % x
		keyFn2 = lambda x: 'n%d' % x
		pu.plot_nested_dict(acc2, nTr, keyFn1, nTe, keyFn2)
	return acc2

##
#The effect of pretraining with less diversity of data
def effect_diversity_pretrain(isPlot=False):
 	nTr = [1e+3, 1e+4, 1e+5, 1e+6, 1e+7]
	nTe = [100, 300, 1000]
	nw  = get_best_nw()
	acc = me.run_final_finetune(runTypes=['accuracy'], runNum=[0],
					numTrainEx=nTr, nw=nw, clsOnly=[1])
	del acc['numExamples']
	nwName = me.nw2name(nw[0])
	redFn  = lambda x: 1 - x[0]
	acc2 = ou.conditional_select(acc, [None, 'top', None, nwName], reduceFn=redFn)
	if isPlot:
		keyFn1 = lambda x: 'nTrn%.01e' % x
		keyFn2 = lambda x: 'n%d' % x
		pu.plot_nested_dict(acc2, nTr, keyFn1, nTe, keyFn2)
	return acc2

##
#Compare slow features with ego-motion
def compare_slowness_egomotion():
	egoRes  = '/data1/pulkitag/mnist/results/compiled/pretrain.pkl'
	#Get the accuracies
	accSlow = me.run_final_finetune(runTypes=['accuracy'], runNum=[0], isSlowness=True)
	accEgo  = pickle.load(open(egoRes, 'rb'))
	trKey   = 'nTrn%.01e' % 1e+7
	accSlow = accSlow[trKey]['top']
	accEgo  = accEgo['muAcc']['top']
	#Get the n/ws
	nwSlow  = me.get_final_source_networks_slowness()
	#Massage the data in the right format
	numTe   = [100, 300, 1000, 10000]
	teKey   = ['n%d' % t for t in numTe]
	acc     = {}
	lines = [] 
	for ns  in nwSlow:
		name     = me.nw2name(ns)
		key      = me.nw2name_small(ns)
		l   = '' + key
		#pdb.set_trace()
		#Slowness Results
		for te in teKey:
			l = l + ' & ' + '%.1f' % (100 - 100*accSlow[te][name][0]) 
		#EgoResults
		for te in teKey:
			l = l + ' & ' + '%.1f' % accEgo[te][key]
		l = l + '\\\ \n'
		lines.append(l)
	return lines
