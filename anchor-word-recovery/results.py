"""
In this file, we take the output matrix and log likelihoods
of the topics and compute P(D) and the coherence
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
import scipy.stats
from scipy.stats import multivariate_normal
import pickle

#computes P(D)
def computeProb(term_doc, A):
	A_inv = pinv(A)
	(N, num_docs) = term_doc.shape
	R = np.zeros((num_docs, num_docs))
	Q = 1/num_docs * np.dot(adj_mat, adj_mat.T)

	for row in term_doc:
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
def nips(N, M, r):
	A = np.zeros((N, r))
	counter = 0

	with open("L2_out.nips.10.A") as f:
		for line in f:
			line = line.strip()
			line = line.split()
			for i in range(0,line):
				A[counter, i] = float(line[i])
			counter = counter + 1


	file = open("M_nips.full_docs.mat.trunc.mat", "rb")
	adj_mat = pickle.load(file)
	file.close()

	file = open("nips_term_doc.pickle", "rb")
	term_doc = pickle.load(file)
	file.close()

	prob = computeProb(term_doc, A)
	coherence = computeCoherence(adj_mat, A, r)

	fout = open("../results.txt", "ab")
	fout.write("nips nmf" + str(prob))

	for i in range(0,K):
		fout.write(str(coherence[i]) + " ")

	fout.write("\n")
	fout.close()



#computes P(D) and coherence for newsgroup dataset
def newsgroup(N, r):
	A = np.zeros((N, r))
	counter = 0

	with open("L2_out.newsgroup.10.A") as f:
		for line in f:
			line = line.strip()
			line = line.split()
			for i in range(0,line):
				A[counter, i] = float(line[i])
			counter = counter + 1


	file = open("M_newsgroup.full_docs.mat.trunc.mat", "rb")
	adj_mat = pickle.load(file)
	file.close()

	file = open("newsgroup_term_doc.pickle", "rb")
	term_doc = pickle.load(file)
	file.close()

	prob = computeProb(term_doc, A)
	coherence = computeCoherence(adj_mat, A, r)

	fout = open("../results.txt", "ab")
	fout.write("newsgroup nmf" + str(prob))

	for i in range(0,K):
		fout.write(str(coherence[i]) + " ")

	fout.write("\n")
	fout.close()

#computes P(D) and coherence for the nyt dataset
def nyt(N, r):
	A = np.zeros((N, r))
	counter = 0

	with open("L2_out.nyt.10.A") as f:
		for line in f:
			line = line.strip()
			line = line.split()
			for i in range(0,line):
				A[counter, i] = float(line[i])
			counter = counter + 1


	file = open("M_nyt.full_docs.mat.trunc.mat", "rb")
	adj_mat = pickle.load(file)
	file.close()

	file = open("nyt_term_doc.pickle", "rb")
	term_doc = pickle.load(file)
	file.close()

	prob = computeProb(term_doc, A)
	coherence = computeCoherence(adj_mat, A, r)

	fout = open("../results.txt", "ab")
	fout.write("nyt nmf" + str(prob))

	for i in range(0,K):
		fout.write(str(coherence[i]) + " ")

	fout.write("\n")
	fout.close()


if __name__ == "__main__":
	N = float(sys.argv[1])
	r = float(sys.argv[2])
	newsgroup(N, r)
    #nyt(N, r)
    #nips(N, r)