import faq_clean as cleaned
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import vincent

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
n_answers = 5
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

    Calculates the optimal number of topics depending on number of comments 
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

def Q_and_A(forum, X, nmf, questions, n_top_questions, n_answers):
    '''
    INPUT: X, nmf, questions (X matrix, fitted model, all questions in forum)
    OUTPUT: FAQ (top questions associated with each topic)

    Figures out the top questions for each topic as determined by the NMF model
    Returns the top 'n_top_questions' number of FAQ for each topic
    Returns an 'n_answers' number of comments for each FAQ
    '''
    #Get the component W and H matrices for NMF model where X = dot(W , H)
    W = nmf.fit_transform(X)
    H = nmf.components_

    #Group the answers by questions
    concat_comments = forum[forum.type=="Comment"].sort("comment_thread_id").groupby("comment_thread_id").body.sum()

    #Initialize variables
    FAQ = []
    top_answers = [[["" for i in range(n_answers+1)] for j in range(n_top_questions)] for k in range(nmf.n_components_)]

    for t in xrange(W.shape[1]):
        #Get top n_questions closest questions to each topic
        FAQ.append([questions[i] for i in np.argsort(W.T[t])[:-n_top_questions-1:-1]])

        #Loop through top n_questions closest questions to the topic
        for q, i in enumerate(np.argsort(W.T[t])[:-n_top_questions-1:-1]):
            #All answers for each of the closest questions
            thread = forum[forum.comment_thread_id == concat_comments.index[i]]
            #Append n_answers for each question
            for a in range(len(thread)):
                #Ranked answers based on number of "upvotes"
                if a <= n_answers: 
                    #Ranked answers based on number of "upvotes"
                    #top_answers[t][n] = thread.sort("up_count", ascending = False).body.iloc[n]
                    #Unranked/sequential answers
                    top_answers[t][q][a] = thread.body.iloc[a]

    return FAQ, top_answers

def word_cloud(nmf, feature_names):
    '''
    Generates word clouds using the `vincent` library
    '''

    topics_dicts = []
    H = nmf.components_
    n_topics = nmf.n_components_

    for i in xrange(n_topics):
        # n_top_words of keys and values
        keys, values = zip(*sorted(zip(feature_names, H[i]), key = lambda x: x[1])[:-n_top_words:-1])
        val_arr = np.array(values)
        norms = val_arr / np.sum(val_arr)
        #normalize = lambda x: int(x / (max(counter.values()) - min(counter.values())) * 90 + 10)
        topics_dicts.append(dict(zip(keys, np.rint(norms* 300))))

    vincent.core.initialize_notebook()

    for i in xrange(n_topics):
        word_cloud = vincent.Word(topics_dicts[i])
        word_cloud.width = 400
        word_cloud.height = 400
        word_cloud.padding = 0
        word_cloud.display()


def histograms(nmf, feature_names, n_top_words):
    '''
    Generates historgrams of words for each topic 
    '''
    for topic_idx, topic in enumerate(nmf.components_):

        words = [feature_names[i] for i in topic.argsort()[:-(n_top_words):-1]]
        scores = [topic[i] for i in topic.argsort()[:-(n_top_words):-1]]

        dicta = {word: score for score, word in zip(scores, words)}

        s=pd.Series(dicta)
        plt.figure()
        plt.title("Topic #%d:" % topic_idx)
        s.plot(kind='bar');





f = cleaned.pull_data(pathname, filename, coursename)
all_questions, all_comments = cleaned.questions_comments(f)
X, feature_names = vectorize(all_comments)
nmf = nmf_model(X)

faq, top_answers = Q_and_A(f, X, nmf, all_questions, n_top_questions, n_answers)

#print faq[0][0]
#print top_answers[0][0][0]

#word_cloud(nmf, feature_names)
#histograms(nmf, feature_names, n_top_words)

