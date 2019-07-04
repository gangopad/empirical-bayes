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

data_dir = "/Users/anirbang/DeltaSierra/Publications/EmpiricalBayes/data/"


#gets the topic scores per document
def getDocumentTopicBreakDown(w_i):
	topic_scores = []

	for i in range(len(w_i)):
		topic_scores.append((i, w_i[i]))

	return topic_scores


#computes P(D)
def computeProb(term_doc, A):
	A_inv = pinv(A)
	(N, num_docs) = term_doc.shape
	R = np.zeros((num_docs, num_docs))
	Q = 1/num_docs * np.dot(adj_mat, adj_mat.T)
	document_topics = []

	for row in term_doc:
		w_i = np.dot(A_inv, row)
		topic_scores = getDocumentTopicBreakDown(w_i)
		document_topics.append(topic_scores)

		R = R + np.dot(w_i.T, w_i)


	mu = np.dot(A, R)
	mu = np.dot(mu, A.T)


	prob = multivariate_normal.pdf(Q, mean=mu, cov=1)

	return np.linalg.norm(prob), document_topics



#computes the coherence for each topic
def computeCoherence(adj_mat, A, K):
    coherence = np.zeros(K)
    epsilon = 0.01
    cutoff = 0.000001
	(row, col) = adj_mat.shape

    for r in range(0,K):
    	for node1 in range(0, row):
    		deg = np.sum(adj_mat[node1:])
        	for node2 in adj_mat:
            	if A[node1, r] > cutoff and A[node2, r] > cutoff:
                	score = coherence[r]
                	curr_coherence = (adj_mat[node1, node2] + epsilon)/(deg)
                	score = score + math.log(curr_coherence, 2)
                	coherence[r] = score

    return coherence

	

#computes P(D) and coherence for the nyt dataset
def compute(N, r, dat_type):
	A = np.zeros((N, r))
	counter = 0

	with open("L2_out." + dat_type + ".10.A") as f:
		for line in f:
			line = line.strip()
			line = line.split()
			for i in range(0,line):
				A[counter, i] = float(line[i])
			counter = counter + 1


	fname = open("M_" + dat_type + ".full_docs.mat.trunc.mat", "rb")
	adj_mat = pickle.load(fname)
	fname.close()

    fname = open(os.path.join(data_dir,'processed/%s_term_doc.pickle'%dat_type),'rb')
	term_doc = pickle.load(fname)
	fname.close()

	(prob, document_topics) = computeProb(term_doc, A)
	coherence = computeCoherence(adj_mat, A, r)

    fout = open(os.path.join(data_dir,'processed/results.txt'), "ab")
	fout.write(dat_type + " nmf" + str(prob))

	for i in range(0,r):
		fout.write(str(coherence[i]) + " ")

	fout.write("\n")
	fout.close()

	fname = open(os.path.join(data_dir,'processed/%s_nmf_document_topics.pickle'%dat_type), "wb")
    pickle.dump(document_topics, fname)
    pickle.close()


if __name__ == "__main__":
	N = float(sys.argv[1]) #vocab size
	r = float(sys.argv[2]) #number of topics
	compute(N, r, "nips")
    nyt(N, r, "newsgroup")
    nips(N, r, "twitter")
  