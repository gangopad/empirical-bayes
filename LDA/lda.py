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





if __name__ =="__main__":
	newsgroup()