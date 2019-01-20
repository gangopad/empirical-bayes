"""
We run Modularity maximization with spectral clustering  on various datasets and generate 
the topic matrix A in a serialized file
"""

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn import metrics
np.random.seed(1)


#computes the spectral clusters
def computeSpectral(adj_matrix, dictionary):
	# Cluster
	sc = SpectralClustering(12, affinity='precomputed', n_init=100)
	sc.fit(adj_mat)

	# Compare ground-truth and clustering-results
	print('spectral clustering')
	print(sc.labels_)

	return sc.labels_


#runs spectral clustering over karate data and measures results
def karate():
	# Get your mentioned graph
	G = nx.karate_club_graph()

	# Get ground-truth: club-labels -> transform to 0/1 np-array
	#     (possible overcomplicated networkx usage here)
	gt_dict = nx.get_node_attributes(G, 'club')
	gt = [gt_dict[i] for i in G.nodes()]
	gt = np.array([0 if i == 'Mr. Hi' else 1 for i in gt])

	# Get adjacency-matrix as numpy-array
	adj_mat = nx.to_numpy_matrix(G)

	print("The shape of the adjacency-matrix" + str(adj_mat.shape))

	print('ground truth')
	print(gt)


	labels = computeSpectral(adj_mat, None)

	# Compare ground-truth and clustering-results
	print('spectral clustering')
	print(labels)
	print('just for better-visualization: invert clusters (permutation)')
	print(np.abs(labels - 1))

	# Calculate some clustering metrics
	print(metrics.adjusted_rand_score(gt, labels))
	print(metrics.adjusted_mutual_info_score(gt, labels))

if __name__ == "__main__":
	file = open("../adjacency_newsgroup.pickle",'rb')
    adj_mat = pickle.load(file)
    file.close()

    file = open("../dictionary_newsgroup.pickle",'rb')
    dictionary = pickle.load(file)
    file.close()

    labels = computeSpectral(adj_mat, dictionary)
    
	# Compare ground-truth and clustering-results
	print('spectral clustering')
	print(labels)

