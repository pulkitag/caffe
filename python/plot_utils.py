## @package plot_utils
#  Miscellaneous Plotting Functions
#
import matplotlib.pyplot as plt

DEFAULT_COLORS = ['r','b','g','black']

def plot_nested_dict(data, xVar, xKeyFn, yVar, yKeyFn, **kwargs):
	'''
		xVar  : variable in x (assumed to be the first level)
		xKeyFn: key = xKeyFn(xVar(i)) should return the key in the dict
		yVar  : variable in y (assumed to be the second level of nested dict)
		yKeyFn: refer to xKeyFn 
		**kwargs:
			colors: The colors for the lines
	'''
	if not kwargs.has_key('colors'):
		colors = DEFAULT_COLORS
	else:
		colors = kwargs['colors'] 
	lines = []
	for i,y in enumerate(yVar):
		key1 = yKeyFn(y)
		plotData  = []
		for x in xVar:
			key2 = xKeyFn(x)
			plotData.append(data[key2][key1])
		line, = plt.semilogx(xVar, plotData)
		plt.setp(line,linewidth=3, color=colors[i])
		lines.append(line)
	plt.legend(lines, [yKeyFn(y) for y in yVar]) 
 
