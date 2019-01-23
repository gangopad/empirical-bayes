"""
We run Modularity maximization with spectral clustering  on various datasets and generate 
the topic matrix A in a serialized file
"""

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn import metrics
np.random.seed(1)
import pickle
import scipy.stats
import math

#computes the spectral clusters
def computeSpectral(adj_mat, dictionary):
    G=nx.from_numpy_matrix(adj_mat)
    (r, c) = adj_mat.shape


    # Cluster
    sc = SpectralClustering(12, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)

    print "Computing omega"
    (omega, degrees) = computeOmega(adj_mat, sc.labels_, r, c)
    print "Computed omega"

    probs = computeProb(omega, adj_mat, degrees, sc.labels_)
    print "The computed P(D) is " + str(probs)

    coherence = computeCoherence(adj_mat, sc.labels_, degrees, K)
    print "The coherence is "
    for topic in coherence:
        print "For topic " + str(coherence.index(topic)) + " the coherence is " + str(topic)


    return probs, coherence, sc.labels_




#computes omega 
def computeOmega(adj_mat, labels, r, c):
    degrees = dict()
    E = numpy.sum(adj_mat)
    E_cout = 0
    omega = np.zeros(r, c)
    k_ci = dict() #k_{c_{i}} which represents the degree of connections outside the community fo node i

    for node in range(0,r):
        deg = numpy.sum(adj_mat[node:])
        degrees[node] = deg
        curr_label = labels[node]
        curr_k_ci = 0

        for node2 in range(0,c):
            if labels[node2] != curr_label:
                curr_k_ci = curr_k_ci + 1
                E_cout = E_cout + 1

        k_ci[node] = curr_k_ci


    for node in range(0,r):
        for node2 in range(0,c):
            if labels[node] == labels[node2]:
                omega[node1, node2] = adj_mat[node1, node2] - float(degrees[node1] * degrees[node2])/float(2 * E)
            else:
                omega[node1, node2] = float(k_ci[node1] * k_ci[node2])/(2 * E_cout)

    return omega, degrees


#computes the coherence for each topic
def computeCoherence(adj_mat, labels, degrees, K):
    coherence = np.zeros(K)
    epsilon = 0.01

    for node1 in adj_mat:
        for node2 in adj_mat:
            if labels[node1] == labels[node2]:
                score = coherence[labels[node1]]
                curr_coherence = (adj_mat[node1, node2] + epsilon)/(degrees[node2])
                score = score + math.log(curr_coherence, 2)
                coherence[labels[node1]] = score

    return coherence


#computes P(D)
def computeProb(omega, adj_mat):
    (r,c) = adj_mat.shape
    prob = 0

    for node1 in range(0,r):
        for node2 in range(0,c):
            edge_prob = scipy.stats.norm(omega[node1, node2], 1).pdf(adj_mat[node1, node2])
            prob = prob + math.log(edge_prob, 2)

    return prob



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

#compute spectral clustering for newsgroup
def newsgroup():
    file = open("../adjacency_newsgroup.pickle",'rb')
    adj_mat = pickle.load(file)
    file.close()

    file = open("../dictionary_newsgroup.pickle",'rb')
    dictionary = pickle.load(file)
    file.close()

    probs, coherence, labels = computeSpectral(adj_mat, dictionary)

    # Compare ground-truth and clustering-results
    print('spectral clustering')
    print(labels)



#compute spectral clustering for nyt
def nyt():
    file = open("../adjacency_nyt.pickle",'rb')
    adj_mat = pickle.load(file)
    file.close()

    file = open("../dictionary_nyt.pickle",'rb')
    dictionary = pickle.load(file)
    file.close()

    probs, coherence, labels = computeSpectral(adj_mat, dictionary)

    # Compare ground-truth and clustering-results
    print('spectral clustering')
    print(labels)




#compute spectral clustering for nips
def nips():
    file = open("../adjacency_nips.pickle",'rb')
    adj_mat = pickle.load(file)
    file.close()

    file = open("../dictionary_nips.pickle",'rb')
    dictionary = pickle.load(file)
    file.close()

    probs, coherence, labels = computeSpectral(adj_mat, dictionary)

    # Compare ground-truth and clustering-results
    print('spectral clustering')
    print(labels)



if __name__ == "__main__":
    newsgroup()
    #nyt()
    #nips()

