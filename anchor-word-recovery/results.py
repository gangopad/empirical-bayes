"""
In this file, we take the output matrix and log likelihoods
of the topics and compute P(D) and the coherence
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
import scipy.stats
from scipy.stats import multivariate_normal

#computes P(D)
def computeProb(adj_mat, A, num_docs):
	A_inv = pinv(A)
	R = np.zeros((num_docs, num_docs))
	Q = 1/num_docs * np.dot(adj_mat, adj_mat.T)

	for row in adj_mat:
		w_i = np.dot(A_inv, row)

		R = R + np.dot(w_i.T, w_i)


	mu = np.dot(A, R)
	mu = np.dot(mu, A.T)


	prob = multivariate_normal.pdf(Q, mean=mu, cov=1)
	
	return np.linalg.norm(prob)



#computes the coherence for each topic
def computeCoherence(adj_mat, A, K):
    coherence = np.zeros(K)
    epsilon = 0.01
    cutoff = 0.000001

    for r in range(0,k):
    	for node1 in adj_mat:
    		deg = np.sum(adj_mat[node1:])
        	for node2 in adj_mat:
            	if A[node1, r] > cutoff and A[node2, r] > cutoff:
                	score = coherence[r]
                	curr_coherence = (adj_mat[node1, node2] + epsilon)/(deg)
                	score = score + math.log(curr_coherence, 2)
                	coherence[r] = score

    return coherence

	


#computes P(D) and coherence for the nips dataset
def nips():
	dhd

#computes P(D) and coherence for newsgroup dataset
def newsgroup():
	dd

#computes P(D) and coherence for the nyt dataset
def nyt():
	dd


if __name__ == "__main__":
	newsgroup()
    #nyt()
    #nips()