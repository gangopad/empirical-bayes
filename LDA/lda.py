"""
We run LDA on various datasets and generate the topic matrix A in a serialized
file
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


#computes LDA given bag-of-words
def computeLDA(bow_corpus, dictionary):

    """
    # LDA mono-core -- fallback code in case LdaMulticore throws an error on your machine
    lda_model = gensim.models.LdaModel(bow_corpus, num_topics = 10, id2word = dictionary, passes = 50)
    """

    """
    # LDA multicore 
    Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
    """
    lda_model =  gensim.models.LdaMulticore(bow_corpus, num_topics = 8, id2word = dictionary, passes = 10, workers = 2)

    """
    For each topic, we will explore the words occuring in that topic and its relative weight
    """
    for idx, topic in lda_model.print_topics(-1):
        print("Topic: {} \nWords: {}".format(idx, topic ))
        print("\n")


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
        
    #preview preprocessed docs
    print(processed_docs[:2])

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

    computelDA(bow_corpus, dictionary)


if __name__ =="__main__":
	newsgroup()
