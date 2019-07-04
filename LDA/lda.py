"""
We run LDA with online variational bayes on various datasets and generate the topic matrix 
A in a serialized file
"""

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
np.random.seed(400)
import pandas as pd
import pickle
import math
import sys
import os

data_dir = "/Users/anirbang/DeltaSierra/Publications/EmpiricalBayes/data/"


#computes LDA given bag-of-words
def computeLDA(bow_corpus, dictionary, cutoff, K):
    N = len(bow_corpus)
    train = int(0.7 * N)

    """
    # LDA mono-core -- fallback code in case LdaMulticore throws an error on your machine
    lda_model = gensim.models.LdaModel(bow_corpus, num_topics = 10, id2word = dictionary, passes = 50)
    """

    """
    # LDA multicore 
    Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
    """
    lda_model =  gensim.models.LdaMulticore(bow_corpus[:train], num_topics = K, id2word = dictionary, passes = 10, workers = 2)

    """
    For each topic, we will explore the words occuring in that topic and its relative weight
    """
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic ))
        print("\n")

    (probs, document_topics) = computeProb(lda_model, bow_corpus, cutoff)


    print "P(D) is represented as: " + str(probs)

    """
    #held out documents
    unseen_lda = lda_model[bow_corpus[train:]]
    inference = lda_model.inference(bow_corpus)
    print "Inference"
    print inference[0]
    print inference[0].shape
    """

    coherence = lda_model.top_topics(bow_corpus)
   
    """
    print "Coherence"

    for topic in coherence:
        print topic[1]
    """

    return probs, coherence, document_topics, lda_model



#computes P(d_{i}) given the multinomial and beta LDA parameters
def computeDocProb(mult, beta, cutoff):
    doc_prob = 0

    #we first convert mult to hashmap of topic index to prob
    hash_mult = dict()
    for topic in mult:
        topic_index = topic[0]
        topic_prob = topic[1]
        hash_mult[topic_index] = topic_prob

    for word in beta:
        word_prob = 0.0
        word_index = word[0]
        word_topic_probs = word[1]

        for prob in word_topic_probs:
            topic_index = prob[0]
            word_topic_prob = prob[1]
            if topic_index in hash_mult:
                #word_prob = word_prob + (float(hash_mult[topic_index]) * float(word_topic_prob))
                word_prob = word_prob + math.log(float(hash_mult[topic_index]) * float(word_topic_prob), 2)

        
        if word_prob > cutoff:
            #doc_prob = doc_prob + math.log(word_prob, 2)
            doc_prob = doc_prob + word_prob

    return doc_prob


#computes the log probability of the corpus P(D)
def computeProb(lda_model, bow_corpus, cutoff):
    log_prob = 0.0
    document_topics = []
    for doc in bow_corpus:
        params = lda_model.get_document_topics(doc, per_word_topics=True)
        mult = params[0]
        beta = params[2]
        document_topics.append(mult)

        doc_prob = computeDocProb(mult, beta, cutoff)
        log_prob = log_prob + doc_prob

    return log_prob, params


#runs LDA for NYT dataset
def compute(cutoff, K, dat_type):
    fname = open(os.path.join(data_dir,'processed/bow_%s.pickle'%dat_type),'rb')
    bow_corpus = pickle.load(fname)
    fname.close()

    fname= open(os.path.join(data_dir,'processed/dictionary_%s.pickle'%dat_type),'rb')
    dictionary = pickle.load(fname)
    fname.close()

    (probs, coherence, document_topics, lda_model) = computeLDA(bow_corpus, dictionary, cutoff, K)

    #write results to file
    fout = open(os.path.join(data_dir,'processed/results.txt'), "ab")
    fout.write(dat_type + " lda" + str(probs))

    for topic in coherence:
        fout.write(str(topic[1]) + " ")

    fout.write("\n")
    fout.close()

    fname = open(os.path.join(data_dir,'processed/%s_lda_document_topics.pickle'%dat_type), "wb")
    pickle.dump(document_topics, fname)
    fname.close()



if __name__ =="__main__":
    cutoff = float(sys.argv[1]) #this parameter is intended to prevent underflow issues in probability computation
    
    #run on twitter, newsgroup, nips    
    compute(cutoff, 100, "nips")
    #compute(cutoff, 100, "newsgroup")
    #compute(cutoff, 12, "twitter")




