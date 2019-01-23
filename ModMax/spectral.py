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
import sys

#computes the spectral clusters
def computeSpectral(adj_mat, dictionary, cutoff):
    G=nx.from_numpy_matrix(adj_mat)
    (r, c) = adj_mat.shape


    # Cluster
    sc = SpectralClustering(12, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)

    print "Computing omega"
    (omega, degrees) = computeOmega(adj_mat, sc.labels_, r, c)
    print "Computed omega"

    probs = computeProb(omega, adj_mat, cutoff)
    print "The computed P(D) is " + str(probs)

    coherence = computeCoherence(adj_mat, sc.labels_, degrees, K)
    print "The coherence is "
    for i in range(len(coherence)):
        print "For topic " + str(i) + " the coherence is " + str(coherence[i])


    return probs, coherence, sc.labels_




#computes omega 
def computeOmega(adj_mat, labels, r, c):
    degrees = dict()
    E = np.sum(adj_mat)
    E_cout = 0
    omega = np.zeros((r, c))
    k_ci = dict() #k_{c_{i}} which represents the degree of connections outside the community fo node i

    for node1 in range(0,r):
        deg = np.sum(adj_mat[node:])
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
def computeProb(omega, adj_mat, cutoff):
    (r,c) = adj_mat.shape
    prob = 0

    for node1 in range(0,r):
        for node2 in range(0,c):
            edge_prob = scipy.stats.norm(omega[node1, node2], 1).pdf(adj_mat[node1, node2])

            if edge_prob > cutoff:
                prob = prob + math.log(edge_prob, 2)

    return prob


#compute spectral clustering for newsgroup
def newsgroup(cutoff):
    file = open("../adjacency_newsgroup.pickle",'rb')
    adj_mat = pickle.load(file)
    file.close()

    file = open("../dictionary_newsgroup.pickle",'rb')
    dictionary = pickle.load(file)
    file.close()

    probs, coherence, labels = computeSpectral(adj_mat, dictionary, cutoff)

    # Compare ground-truth and clustering-results
    print('spectral clustering')
    print(labels)

    fout = open("../results.txt", "ab")
    fout.write("newsgroup modmax" + str(prob))

    for i in range(0,len(coherence)):
        fout.write(str(coherence[i]) + " ")

    fout.write("\n")
    fout.close()



#compute spectral clustering for nyt
def nyt(cutoff):
    file = open("../adjacency_nyt.pickle",'rb')
    adj_mat = pickle.load(file)
    file.close()

    file = open("../dictionary_nyt.pickle",'rb')
    dictionary = pickle.load(file)
    file.close()

    probs, coherence, labels = computeSpectral(adj_mat, dictionary, cutoff)

    # Compare ground-truth and clustering-results
    print('spectral clustering')
    print(labels)

    fout = open("../results.txt", "ab")
    fout.write("nyt modmax" + str(prob))

    for i in range(0,len(coherence)):
        fout.write(str(coherence[i]) + " ")

    fout.write("\n")
    fout.close()



#compute spectral clustering for nips
def nips(cutoff):
    file = open("../adjacency_nips.pickle",'rb')
    adj_mat = pickle.load(file)
    file.close()

    file = open("../dictionary_nips.pickle",'rb')
    dictionary = pickle.load(file)
    file.close()

    probs, coherence, labels = computeSpectral(adj_mat, dictionary, cutoff)

    # Compare ground-truth and clustering-results
    print('spectral clustering')
    print(labels)

    fout = open("../results.txt", "ab")
    fout.write("nips modmax" + str(prob))

    for i in range(0,len(coherence)):
        fout.write(str(coherence[i]) + " ")

    fout.write("\n")
    fout.close()



if __name__ == "__main__":

    cutoff = float(sys.argv[1]) #this parameter is intended to prevent underflow issues in probability computation
    newsgroup(cutoff)
    nyt(cutoff)
    nips(cutoff)

