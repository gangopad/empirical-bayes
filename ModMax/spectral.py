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
import os

data_dir = "/Users/anirbang/DeltaSierra/Publications/EmpiricalBayes/data/"

#computes the spectral clusters
def computeSpectral(adj_mat, dictionary, cutoff):
    G=nx.from_numpy_matrix(adj_mat)
    (r, c) = adj_mat.shape


    # Cluster
    sc = SpectralClustering(100, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)

    print "Computing omega"
    (omega, degrees) = computeOmega(adj_mat, sc.labels_, r, c)
    print "Computed omega"

    """
    probs = computeProb(omega, adj_mat, cutoff)
    print "The computed P(D) is " + str(probs)

    coherence = computeCoherence(adj_mat, sc.labels_, degrees, K)
    print "The coherence is "
    for i in range(len(coherence)):
        print "For topic " + str(i) + " the coherence is " + str(coherence[i])
    """

    probs = 0
    coherence = []

    return probs, coherence, sc.labels_




#computes omega 
def computeOmega(adj_mat, labels, r, c):
    degrees = dict()
    E = np.sum(adj_mat)
    E_cout = 0
    omega = np.zeros((r, c))
    k_ci = dict() #k_{c_{i}} which represents the degree of connections outside the community fo node i

    for node in range(0,r):
        deg = np.sum(adj_mat[node:])
        degrees[node] = deg
        curr_label = labels[node]
        curr_k_ci = 0

        for node2 in range(0,c):
            if labels[node2] != curr_label:
                curr_k_ci = curr_k_ci + 1
                E_cout = E_cout + 1

        k_ci[node] = curr_k_ci


    for node1 in range(0,r):
        for node2 in range(0,c):
            if labels[node1] == labels[node2]:
                omega[node1, node2] = adj_mat[node1, node2] - float(degrees[node1] * degrees[node2])/float(2 * E)
            else:
                omega[node1, node2] = float(k_ci[node1] * k_ci[node2])/(2 * E_cout)

    return omega, degrees


#computes the coherence for each topic
def computeCoherence(adj_mat, labels, degrees, K):
    coherence = np.zeros(K)
    epsilon = 0.01
    (r, c) = adj_mat.shape


    for node1 in range(0,r):
        for node2 in range(0,r):
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


#gets the breakdown of documents by topic
def getDocumentTopicBreakDown(labels, adj_mat, dictionary, bow_corpus):
    document_topics = []

    for doc in bow_corpus:
        denom = 0.0
        topic_prob = dict()
        topic_scores = [] #convert topic_prob to a list of (ind, float) where int is topic id and float is score
        for word in doc:
            label = labels[word[0]] #word is index (in dictionary) along with freq count

            if label in topic_prob:
                score = topic_prob[label]
            else:
                score = 0

            score = score + 1
            topic_prob[label] = score
            denom = denom + 1

        for label in topic_prob:
            score = float(topic_prob[label])/float(denom)
            topic_scores.append((label, score))

        document_topics.append(topic_scores)

    return document_topics
    

#compute spectral clustering for nyt
def compute(cutoff, dat_type):
    fname = open(os.path.join(data_dir,'processed/bow_%s.pickle'%dat_type),'rb')
    bow_corpus = pickle.load(fname)
    fname.close()


    fname = open(os.path.join(data_dir,'processed/adjacency_%s.pickle'%dat_type),'rb')
    adj_mat = pickle.load(fname)
    fname.close()

    fname = open(os.path.join(data_dir,'processed/dictionary_%s.pickle'%dat_type),'rb')
    dictionary = pickle.load(fname)
    fname.close()

    probs, coherence, labels = computeSpectral(adj_mat, dictionary, cutoff)

    # Compare ground-truth and clustering-results
    print('spectral clustering')
    print(len(set(labels)))

    document_topics = getDocumentTopicBreakDown(labels, adj_mat, dictionary, bow_corpus)

    fout = open("../results.txt", "ab")
    fout.write(dat_type + " modmax" + str(probs))

    for i in range(0,len(coherence)):
        fout.write(str(coherence[i]) + " ")

    fout.write("\n")
    fout.close()

    fname = open(os.path.join(data_dir,'processed/%s_spectral_document_topics.pickle'%dat_type), "wb")
    pickle.dump(document_topics, fname)
    fname.close()




if __name__ == "__main__":
    cutoff = float(sys.argv[1]) #this parameter is intended to prevent underflow issues in probability computation
    
    #run on twitter, newsgroup, nips    
    compute(cutoff, "nips")
    #compute(cutoff, "newsgroup")
    #compute(cutoff, "twitter")