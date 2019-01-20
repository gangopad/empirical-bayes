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


#we consider the 20newsgroup dataset
def newsgroup():
	newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
	newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)

	#print(list(newsgroups_train.target_names))
	#print(newsgroups_train.data[:2])
	#print(newsgroups_train.filenames.shape, newsgroups_train.target.shape)

	print(WordNetLemmatizer().lemmatize('went', pos = 'v')) # past tense to present tense


if __name__ =="__main__":
	newsgroup()