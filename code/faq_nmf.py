import faq_clean as cleaned
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

from sklearn import metrics
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.feature_extraction import text as txt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import decomposition

#### FILE PATHS AND NAMES ##########
pathname = "/Users/wubr2000/Dropbox/Data Science Courses/Zipfian project Ideas/FAQ/FAQ/data/"
filename = "statsLearningForum2.csv"
coursename = "Medicine/HRP258/Statistics_in_Medicine"
####################################

##### MODEL PARAMETERS #############
#n_features = 10000
n_top_words = 15
n_top_questions = 5
####################################

def additional_stopwords(filename = pathname+"faq_stopwords.txt"):
    '''
    INPUT: filename for user-defined stopwords
    OUTPUT: list of user-defined stopwords 
    '''
    return [line.strip() for line in open(filename)]

def vectorize(comments):
    '''
    INPUT: comments
    OUTPUT: X

    Takes in all comments for a course's discussion forum and
    outputs a bag of words sparse matrix called X
    '''
    n_samples = len(comments)

    tfidf = TfidfVectorizer(min_df = 10, 
                             #max_df =.99,
                             #max_features = n_features, 
                             stop_words = txt.ENGLISH_STOP_WORDS.union(additional_stopwords()),
                             sublinear_tf=True,
                             ngram_range=(1,2),
                             )

    counts = tfidf.fit_transform(comments[:n_samples])
    X = TfidfTransformer().fit_transform(counts)
    # Inverse the vectorizer vocabulary 
    feature_names = tfidf.get_feature_names()
    
    return X, feature_names

def nmf_model(X):
    '''
    INPUT: X (bag of words)
    OUTPUT: nmf (fitted model)

    calculates the optimal number of topics depending on number of comments 
    '''
    #Start timer
    t0 = time()

    #Set number of "optimal" topics - depends on number of comments
    n_topics = int(round(X.shape[0]*2/100))

    # Fit the NMF model
    print("Fitting the NMF model on with n_samples=%d, n_features=%d, n_topics=%d..."
          % (X.shape[0], X.shape[1], n_topics))
    
    nmf = decomposition.NMF(n_components = n_topics).fit(X)
    
    print("done in %0.3fs." % (time() - t0))

    return nmf

def FAQ(X, nmf, questions, n_top_questions):
    '''
    INPUT: X, nmf, questions (X matrix, fitted model, all questions in forum)
    OUTPUT: FAQ (top questions associated with each topic)


    '''
    #Get the component W and H matrices for NMF model where X = dot(W , H)
    W = nmf.fit_transform(X)
    H = nmf.components_

    #Get the top 5 questions associated with each topic
    FAQ = []
    for t in xrange(W.shape[1]):
        FAQ.append([questions[i] for i in np.argsort(W.T[t])[:-n_top_questions-1:-1]])

    return FAQ



def print_nmf(FAQ, nmf, feature_names, n_top_words):
    #Print out each topics and 5 questions that are most closely associated with that topic
    for topic_idx, topic in enumerate(nmf.components_):
        print "******************************"
        print("Important words for Topic #%d:" % (topic_idx+1))
        print
        print(" ".join( [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] ))
        print 
        print "******************************"
        print
        # for v in FAQ[topic_idx]:
        #     print "Question"
        #     print "============="
        #     print v
        #     print
        print FAQ[topic_idx]


f = cleaned.pull_data(pathname, filename, coursename)
all_questions, all_comments = cleaned.questions_comments(f)
X, feature_names = vectorize(all_comments)
nmf = nmf_model(X)
faq = FAQ(X, nmf, all_questions, n_top_questions)

print_nmf(faq, nmf, feature_names, n_top_words)



