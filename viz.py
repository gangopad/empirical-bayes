"""
In this file we read the serialized outputs from our 
models and visualize results
"""
import numpy as np
import matplotlib.pyplot as plt

#plots the chart
def createPlot(n_groups, lda_prob, modmax_prob, nmf_prob):
	# create plot
	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.25
	opacity = 0.8
 
	rects1 = plt.bar(index, lda_prob, bar_width,
                 alpha=opacity,
                 color='b',
                 label='LDA')
 
	rects2 = plt.bar(index + bar_width, modmax_prob, bar_width,
                 alpha=opacity,
                 color='g',
                 label='ModMax')

	rects3 = plt.bar(index + bar_width * 2, nmf_prob, bar_width,
                 alpha=opacity,
                 color='y',
                 label='NMF')


 
	plt.xlabel('Dataset')
	plt.ylabel('Probability')
	plt.title('P(Model | Data) By Dataset')
	plt.xticks(index + bar_width, ('NYT', '20NewsGroup', 'NIPs'))
	plt.legend()
 
	plt.tight_layout()
	#plt.show()

	fig = plt.gcf()
	fig.set_size_inches(9.5, 5.5)
	fig.savefig('probs.png', dpi=100)


if __name__ == "__main__":
	# data to plot
	n_groups = 3
	lda_prob = (90, 55, 40)
	modmax_prob = (85, 62, 54)
	nmf_prob = (38,55,87)

	createPlot(n_groups, lda_prob, modmax_prob, nmf_prob)
 