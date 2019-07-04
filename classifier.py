"""
We ingest a list of documents broken by topic probability in (idx, prob) format along
with a label specifying the category of the data. We train a random forest to predict
the label and compute precision/recall on held out data
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from random import shuffle
from sklearn import metrics
import os
import pickle

data_dir = "/Users/anirbang/DeltaSierra/Publications/EmpiricalBayes/data/"

#runs the random forest model and returns precision/recall
def run_model(X, y):
    ratio = 0.7 #indicates percentage we want to train on
    (X_train, X_test, y_train, y_test) = split(X, y, ratio)

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    return accuracy, precision, recall

#splits by ratio
def split(X, y, ratio):
    X_train = []
    X_test = [] 
    y_train = []
    y_test = []
    cutoff = int(ratio * len(X))

    indices = list(range(len(X)))
    shuffle(indices)

    count = 0
    for i in indices:

        if count < cutoff:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

        count = count + 1

    return X_train, X_test, y_train, y_test



#puts the data into ingestible format
def clean(document_topics, labels, N):
    r = len(document_topics)
    X = np.zeros((r,N))
    y = np.array(labels)

    for i in range(len(document_topics)):
        doc = document_topics[i]
        print("doc: " + str(doc))

        for entry in doc:
            (ind, prob) = entry
            print("i: " + str(i))
            print("ind: " + str(ind))
            print("prob: " + str(prob))
            X[i,ind] = prob

    return (X,y)


    

#reads the data from disk and returns the object
def read_from_disk(dat_type):    
    with open(os.path.join(data_dir,'processed/%s_lda_document_topics.pickle'%dat_type), "rb") as f:
    #with open(os.path.join(data_dir,'processed/%s_spectral_document_topics.pickle'%dat_type), "rb") as f:
        document_topics = pickle.load(f)

    with open(os.path.join(data_dir,'processed/labels_%s.pickle'%dat_type), "rb") as f:
        labels = pickle.load(f)

    with open(os.path.join(data_dir,'processed/dictionary_%s.pickle'%dat_type),'rb') as f:
        dictionary = pickle.load(f)

    return document_topics, labels, len(dictionary)


def runRF(dat_type):
    (document_topics, labels, N) = read_from_disk(dat_type)
    (X,y) = clean(document_topics, labels, N)
    (accuracy, precision, recall) = run_model(X, y)
    
    #write to disk
    print("accuracy: " + str(accuracy))
    print("precision: " + str(precision))
    print("recall: " + str(recall))


if __name__ == "__main__":
    runRF("nips")
    #runRF("twitter")
    #runRF("newsgroup")
