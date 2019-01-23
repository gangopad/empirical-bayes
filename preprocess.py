"""
Preprocesses the input datasets and outputs serialized 
objects represented cleaned data
1) Adjacency matrix
2) Bag of words 
"""

from sklearn.datasets import fetch_20newsgroups
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
import nltk
nltk.download('wordnet')
import pandas as pd
stemmer = SnowballStemmer("english")
import pickle
import os


#Write a function to perform the pre processing steps on the entire dataset
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result


#takes the preprocessed data and returns the appropriate objects
def serialize(processed_docs):
    #preview preprocessed docs
    #print(processed_docs[:2])

    #create dictionary with frequency counts
    dictionary = gensim.corpora.Dictionary(processed_docs)

    
    #Checking dictionary created
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break


    """
    OPTIONAL STEP
    Remove very rare and very common words:
    
    - words appearing less than 15 times
    - words appearing in more than 10% of all documents
    """
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)

    """
    Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
    words and how many times those words appear. Save this to 'bow_corpus'
    """
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


    """
    Preview BOW for our sample preprocessed document
    """
    document_num = 20
    bow_doc_x = bow_corpus[document_num]

    for i in range(len(bow_doc_x)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
                                                     dictionary[bow_doc_x[i][0]], 
                                                     bow_doc_x[i][1]))


    """
    Convert BOW to adjacency matrix and store documents by index
    """
    adj_mat = np.zeros((len(dictionary), len(dictionary)))
    for doc in bow_corpus:
        for word1 in doc:
            for word2 in doc:
                val = adj_mat.item((word1[0], word2[0]))
                val = val + 1
                adj_mat[word1[0], word2[0]] = val


    return dictionary, bow_corpus, adj_mat


#consider the NYT dataset
def NYT():
    processed_docs = []

    with open("../data/nytimes_news_articles.txt") as f:
        for line in f:
            if "URL:" not in line:
                doc = line.strip()
                processed_docs.append(preprocess(doc))

    (dictionary, bow_corpus, adj_mat) = serialize(processed_docs)

    #dump objects via pickle
    fname = open("bow_nyt.pickle", "wb")
    pickle.dump(bow_corpus, fname)
    fname.close()

    fname = open("dictionary_nyt.pickle", "wb")
    pickle.dump(dictionary, fname)
    fname.close()

    fname = open("adjacency_nyt.pickle", "wb")
    pickle.dump(adj_mat, fname)
    fname.close()

    #dumping into textfile format for anchor-word-recovery scripts
    fout = open("docword.nyt.txt", "wb")
    fout.write(str(len(bow_corpus)))
    fout.write(str(len(vocabulary)))
    fout.write(str(np.count_nonzero(adj_mat)))

    (r,c) = adj_mat.shape

    for i in range(0, r):
        for j in range(0, c):
            fout.write(str(i) + " " + str(j) + " " + str(adj_mat[i, j]) + "\n")

    fout.close()


    #writing vocabulary file
    fout = open("vocab.nyt.txt", "wb")
    for term in dictionary:
        fout.write(term + "\n")
    fout.close()

    #write term document matrix to file
    termdoc = np.zeros((r, len(bow_corpus)))
    for counter in range(len(bow_corpus)):
        doc = bow_corpus[counter]

        for i in range(doc):
            word_index = doc[i][0]
            freq = doc[i][1]

            termdoc[counter, i] = freq

    fname = open("nyt_term_doc.pickle", "wb")
    pickle.dump(termdoc)
    pickle.close()


#consider the set of NIPs abstracts
def nips():
    processed_docs = []
    rootdir = "../data/nipstxt"
    for dir in os.listdir(rootdir):
        if "nips" in dir:
            for file in os.listdir(rootdir + "/" + dir):
                with open(rootdir + "/" + dir + "/" + file) as f:
                    doc = f.read().strip()
                    processed_docs.append(preprocess(doc))

    (dictionary, bow_corpus, adj_mat) = serialize(processed_docs)

    #dump objects via pickle
    fname = open("bow_nips.pickle", "wb")
    pickle.dump(bow_corpus, fname)
    fname.close()

    fname = open("dictionary_nips.pickle", "wb")
    pickle.dump(dictionary, fname)
    fname.close()

    fname = open("adjacency_nips.pickle", "wb")
    pickle.dump(adj_mat, fname)
    fname.close()


    #dumping into textfile format for anchor-word-recovery scripts
    fout = open("docword.nips.txt", "wb")
    fout.write(str(len(bow_corpus)))
    fout.write(str(len(vocabulary)))
    fout.write(str(np.count_nonzero(adj_mat)))

    (r,c) = adj_mat.shape

    for i in range(0, r):
        for j in range(0, c):
            fout.write(str(i) + " " + str(j) + " " + str(adj_mat[i, j]) + "\n")

    fout.close()


    #writing vocabulary file
    fout = open("vocab.nips.txt", "wb")
    for term in dictionary:
        fout.write(term + "\n")
    fout.close()

    #write term document matrix to file
    termdoc = np.zeros((r, len(bow_corpus)))
    for counter in range(len(bow_corpus)):
        doc = bow_corpus[counter]

        for i in range(doc):
            word_index = doc[i][0]
            freq = doc[i][1]

            termdoc[counter, i] = freq

    fname = open("nips_term_doc.pickle", "wb")
    pickle.dump(termdoc)
    pickle.close()



#we consider the 20newsgroup dataset
def newsgroup():
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
    newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)

    """
    #dataset info
    print(list(newsgroups_train.target_names))
    print(newsgroups_train.data[:2])
    print(newsgroups_train.filenames.shape, newsgroups_train.target.shape)
    """


    #test
    #print(WordNetLemmatizer().lemmatize('went', pos = 'v')) # past tense to present tense

    #preprocess the data
    processed_docs = []
    for doc in newsgroups_train.data:
        processed_docs.append(preprocess(doc))
        
    
    (dictionary, bow_corpus, adj_mat) = serialize(processed_docs)


    #dump objects via pickle
    fname = open("bow_newsgroup.pickle", "wb")
    pickle.dump(bow_corpus, fname)
    fname.close()

    fname = open("dictionary_newsgroup.pickle", "wb")
    pickle.dump(dictionary, fname)
    fname.close()

    fname = open("adjacency_newsgroup.pickle", "wb")
    pickle.dump(adj_mat, fname)
    fname.close()

    #dumping into textfile format for anchor-word-recovery scripts
    fout = open("docword.newsgroup.txt", "wb")
    fout.write(str(len(bow_corpus)))
    fout.write(str(len(vocabulary)))
    fout.write(str(np.count_nonzero(adj_mat)))

    (r,c) = adj_mat.shape

    for i in range(0, r):
        for j in range(0, c):
            fout.write(str(i) + " " + str(j) + " " + str(adj_mat[i, j]) + "\n")

    fout.close()


    #writing vocabulary file
    fout = open("vocab.newsgroup.txt", "wb")
    for term in dictionary:
        fout.write(term + "\n")
    fout.close()

    #write term document matrix to file
    termdoc = np.zeros((r, len(bow_corpus)))
    for counter in range(len(bow_corpus)):
        doc = bow_corpus[counter]

        for i in range(doc):
            word_index = doc[i][0]
            freq = doc[i][1]

            termdoc[counter, i] = freq

    fname = open("newsgroup_term_doc.pickle", "wb")
    pickle.dump(termdoc)
    pickle.close()




if __name__ =="__main__":
    newsgroup()
    nyt()
    nips()
