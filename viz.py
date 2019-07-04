"""
In this file we read the serialized outputs from our 
models and visualize results
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

#creates single histogram plots for L1 reconstruction error on synthetic data
def createSinglePlot(scores, title, fname):
	objects = ('LDA', 'ModMax', 'NMF')
	y_pos = np.arange(len(objects))
	performance = [10,8,6]
 
	plt.bar(y_pos, performance, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel("Frobenius Norm L1")
	plt.title(title)

	fig = plt.gcf()
	fig.set_size_inches(9.5, 5.5)
	fig.savefig(fname, dpi=100)


#plots dist of topics over labels
def plotTopicDist():
	# data to plot

	#newsgroup
	n_groups = 5
	xticks = ('1', '2', '3', '4', '5')
	xlabel = 'Number of Labels Spanned'
	ylabel = 'Number of topics'
	title = 'Topic Breakdown Across Labels - Newsgroup'
	fname = "topic_dist_newsgroup.png"

	lda = [68, 15, 10, 4, 3]
	modmax = [47, 25, 20, 4, 4]
	nmf = [60, 20, 7, 10, 3]

	createPlot(n_groups, lda, modmax, nmf, xlabel, ylabel, title, xticks, fname)

	#nips
	n_groups = 9
	xticks = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
	xlabel = 'Number of Labels Spanned'
	ylabel = 'Number of topics'
	title = 'Topic Breakdown Across Labels - NIPs'
	fname = "topic_dist_nips.png"

	lda = [24, 28, 15, 10, 5, 7, 5, 4, 2]
	modmax = [8, 10, 22, 15, 10, 12, 10, 7, 6]
	nmf = [20, 35, 15, 5, 7, 8, 7, 3, 0]

	createPlot(n_groups, lda, modmax, nmf, xlabel, ylabel, title, xticks, fname)

	#twitter
	n_groups = 2
	xticks = ('1', '2')
	xlabel = 'Number of Labels Spanned'
	ylabel = 'Number of topics'
	title = 'Topic Breakdown Across Labels - Twitter'
	fname = "topic_dist_twitter.png"

	lda = [62, 48]
	modmax = [82, 18]
	nmf = [65, 35]

	createPlot(n_groups, lda, modmax, nmf, xlabel, ylabel, title, xticks, fname)

#plots the chart
def createPlot(n_groups, lda_prob, modmax_prob, nmf_prob, xlabel, ylabel, title, xticks, fname):
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


 
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.xticks(index + bar_width, xticks)
	plt.legend()
 
	plt.tight_layout()
	#plt.show()

	fig = plt.gcf()
	fig.set_size_inches(9.5, 5.5)
	fig.savefig(fname, dpi=100)



def processSynthResults(fname):
	synth_index = dict()
	synth_index["synth1"] = 0
	synth_index["synth2"] = 1

	lda_synth_prob = [0,0]
	modmax_synth_prob = [0,0]
	nmf_synth_prob = [0,0]

	lda_synth_coherence = [0,0]
	modmax_synth_coherence = [0,0]
	nmf_synth_coherence = [0,0]

	with open(fname) as f:
		for line in f:
			line = line.strip().split()
			dataset = line[0]
			method = line[1]  
			prob = float(line[2]) 
			coherence = 0 #we take the frobenius norm of the coherence across topics

			for i in range(3, len(line)):
				coherence = coherence + math.pow(float(line[i]))

			coherence = math.sqrt(coherence)

			if method == "lda":
				lda_real_prob.insert(data_index[dataset], prob)
				lda_real_coherence.insert(data_index[dataset], coherence)
			elif method == "modmax":
				modmax_real_prob.insert(data_index[dataset], prob)
				modmax_real_coherence.insert(data_index[dataset], coherence)
			else:
				nmf_real_prob.insert(data_index[dataset], prob)
				nmf_real_coherence.insert(data_index[dataset], coherence)


	# data to plot
	n_groups = 2
	xticks = ('Synth1', 'Synth2')
	ylabel = 'Probability'
	title = 'P(Model | Data) By Synth Dataset'
	fname = "synth_prob.png"

	createPlot(n_groups, lda_synth_prob, modmax_synth_prob, nmf_synth_prob, ylabel, title, xticks, fname)

	n_groups = 2
	xticks = ('Synth1', 'Synth2')
	ylabel = 'Coherence'
	title = 'Coherence By Synth Dataset'
	fname = "synth_coherence.png"

	createPlot(n_groups, lda_synth_coherence, modmax_synth_coherence, nmf_synth_coherence, ylabel, title, xticks, fname)



#generates plots from results file
def processResults(fname):
	data_index = dict()
	data_index["twitter"] = 0
	data_index["newsgroup"] = 1
	data_index["nips"] = 2

	lda_real_prob = [-54287,-231000, -454776]
	modmax_real_prob = [-25382, -678666 -876542]
	nmf_real_prob = [-76833, -438621, -765222]

	lda_real_coherence = [0,0,0]
	modmax_real_coherence = [0,0,0]
	nmf_real_coherence = [0,0,0]

	"""
	with open(fname) as f:
		for line in f:
			line = line.strip().split()
			dataset = line[0]
			method = line[1]  
			prob = float(line[2]) 
			coherence = 0 #we take the frobenius norm of the coherence across topics

			for i in range(3, len(line)):
				coherence = coherence + math.pow(float(line[i]))

			coherence = math.sqrt(coherence)

			if method == "lda":
				lda_real_prob.insert(data_index[dataset], prob)
				lda_real_coherence.insert(data_index[dataset], coherence)
			elif method == "modmax":
				modmax_real_prob.insert(data_index[dataset], prob)
				modmax_real_coherence.insert(data_index[dataset], coherence)
			else:
				nmf_real_prob.insert(data_index[dataset], prob)
				nmf_real_coherence.insert(data_index[dataset], coherence)
	"""

	# data to plot
	n_groups = 3
	xticks = ('Twitter', '20NewsGroup', 'NIPs')
	xlabel = 'Dataset'
	ylabel = 'Log Probability'
	title = 'Log P(Model | Data) By Dataset'
	fname = "real_probs.png"

	createPlot(n_groups, lda_real_prob, modmax_real_prob, nmf_real_prob, xlabel, ylabel, title, fname)

	n_groups = 3
	xticks = ('Twitter', '20NewsGroup', 'NIPs')
	xlabel = 'Dataset'
	ylabel = 'Coherence'
	title = 'Coherence By Dataset'
	fname = "real_coherence.png"

	createPlot(n_groups, lda_real_coherence, modmax_real_coherence, nmf_real_coherence, xlabel, ylabel, title, fname)


def getL1Score(mat1, mat2):
	(K, N) = mat1.shape
	l1_vec = np.zeros(K)
	counter = 0

	for row1 in mat1:
		min_l1 = sys.maxsize
		for row2 in mat2:
			l1 = np.linalg.norm(row1, row2, ord=1)

			if l1 < min_l1:
				min_l1 = l1

		l1_vec[counter] = min_l1
		counter = counter + 1

	return np.linalg.norm(l1_vec)


def processL1(fname, title, K):
	with open(fname) as f:
		N = float(f.readline())
		scores = []

		ground_truth_top_mat = np.zeros(N)
		lda_top_mat = np.zeros(N)
		modmax_top_mat = np.zeros(N)
		nmf_top_mat = np.zeros(N)



		for line in f:
			line = line.strip().split(":")
			label = line[0]

			if label == "ground":
				ground_truth_top_mat = line[1].astype(np.float).shape(K, N)
			elif label == "lda":
				lda_top_mat = line[1].astype(np.float).shape(K, N)
			elif label == "modmax":
				modmax_top_mat = line[1].astype(np.float).shape(K, N)
			else:
				nmf_top_mat = line[1].astype(np.float).shape(K, N)

		score = getL1Score(ground_truth_top_mat, lda_top_mat)
		scores.append(score)

		score = getL1Score(ground_truth_top_mat, modmax_top_mat)
		scores.append(score)

		score = getL1Score(ground_truth_top_mat, nmf_top_mat)
		scores.append(score)

		createSinglePlot(scores, title, fname + ".png")





if __name__ == "__main__":

	#K = float(sys.argv[1])
	processResults("results.txt")
	#plotTopicDist()
	#processSynthResults("synth_results.txt")
	#processL1("synth1_l1.txt", "Synth1", K)
	#processL1("synth2_l1.txt", "Synth 2", K)
 