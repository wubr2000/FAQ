# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
from scipy import io
from scipy import sparse
import sys
import logging
import seaborn 
import matplotlib.pyplot as plt
from IPython.display import display, Math, Latex
from time import time

from sklearn import metrics
from sklearn.metrics.pairwise import linear_kernel,pairwise_distances
from sklearn.feature_extraction import text as txt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer, HashingVectorizer
from sklearn import decomposition

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation

from optparse import OptionParser

import bs4 as bs4
import re, sys

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.regexp import WordPunctTokenizer
import nltk

import gensim
from gensim import corpora, similarities, models
from gensim import matutils

%pylab inline

# <codecell>

forum = pd.read_csv("statsLearningForum4.csv", quoting = 2, header = 0, 
                    escapechar = "\\", encoding = "ISO-8859-1")

# <codecell>

forum.columns = [c.replace("'","") for c in forum.columns]

# <codecell>

pd.options.display.max_colwidth=110
#forum.head()

# <codecell>

del forum["anon_screen_name"]
del forum["at_position_list"]
del forum["course_display_name"]

# <codecell>

forum.head()

# <markdowncell>

# ##Text Cleaning Functions

# <codecell>

#Text Cleaning Functions
def mrclean(text):
    '''
    NAME
            mrclean
    SYNOPSIS
            Cleans incoming text
    DESCRIPTION
            Removes all untranslatable unicode
    '''
    c = ['\xe2','\x80','\x93','\xc2','\xa0','\n', '\r', '\u2019', '\n\n', '\x96', 
         'zipredac', 'phoneredac', 'nameRedac_anon_screen_name_redacted']  
    return re.sub('[%s]' % ''.join(c), '', text)

def soupText(eText):
    '''
    NAME
            soupText

    SYNOPSIS
            Removes all the HTML tags using BeautifulSoup

    DESCRIPTION
            Explicitly calls utf-8, replaces all double quotes with single and calls on helper function mrclean
    '''
    eText = str(eText.encode("utf-8").replace("'",'"') )
    soup = bs4.BeautifulSoup(eText)
    cleanText = soup.findAll(text=True)  #.get_text()
    return mrclean(''.join(cleanText))

# <markdowncell>

# ##Group all comments and associated questions together

# <codecell>

#Concatenate all comments together that has the same "comment_thread_id" 
#- i.e. jam all the comments on one question together - this should be better for analysis
concat_comments = forum[forum.type=="Comment"].sort("comment_thread_id").groupby("comment_thread_id").body.sum()

grouped_comments=[]
for c in concat_comments:
    unicode(c)
    soupText(c)
    grouped_comments.append(c)

grouped_questions = []
for i in concat_comments.index:
    q = forum.body[forum.forum_post_id == i]
    grouped_questions.append(q.iloc[0])

#grouped_questions[:5]

# <codecell>

grouped_comments[:5]

# <markdowncell>

# ##Stopwords, Lemmatizer, and TfidfVectorizer

# <codecell>

#Discussion forums specific stopwords
additional_stop_words = [
                         "zipredac","phoneredac","nameRedac_anon_screen_name_redacted", 
                         "explanation", "helpful", "yes",
                         "agree","excellent","dear", "thanks",
                         "ok", "cool", "idea", "guys",
                         "hello", "student", "sorry", "makes",
                         "sense", "nice", "meet", "yeah", "wow",
                         "hint", "post", "understand", "helped",
                         "www","http","https","org","web","en","wiki","wikipedia","youtube","edu","com","://"
                         ]

# <markdowncell>

# >Token-level analysis such as stemming, lemmatizing, compound splitting, filtering based on part-of-speech, etc. are not included in the scikit-learn codebase, but can be added by customizing either the tokenizer or the analyzer. Here’s a TfidfVectorizer with a tokenizer and lemmatizer using NLTK:

# <codecell>

class LemmaTokenizer(object):
    '''Scikit learn requires the tokenizer to be a callable class if custom made'''
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# <markdowncell>

# > Adjusting for "outliers"
# - Adjusting `min_df` and `max_df` affects the results ALOT!!! They will essentially get rid of "OUTLIERS". Tell function to ignore these words because they occurred in either too many (`max_df`) or in too few (`min_df`) documents.
# 
# - Setting `min_df`= X means that calculation will include only words that appear in at least X documents. This will throw out many really weird, idiosyncratic, or made-up words that appear rarely and that are not germane to the comment.
# 
# - Setting `max_df` = Y means that calculation will exclude words that appear over and over again in every documents. For example, in these forum discussions, there are alot of "thanks", "good", "agree","helpful","helps", "thanks alot" type of words that will throw off calculation. Again, they are not germane to the discussion but will throw off the calcuation. `max_df` is not that relevant if `sublinear_tf` is also turned on since they both do similar things.
# 
# - The `sublinear_tf` parameter, when True, transforms the term frequency calculation to: tf = 1 + log(tf).
# 
# - This transformation, as the name implies, is sublinear, which in practice lessens the effect of frequently occurring words. Since this calculation is only for the term frequency—and thus only affects the numerator of the tf-idf calculation—having a sublinear transform is a good way to reduce the effect of very common words.
# 
# - Bigrams and higher tend to make the results worse.
# 
# - NLTK WordNetLemmatizer really screwed things up...

# <markdowncell>

# >Need to throw away URLs in bag of words 

# <codecell>

n_samples = len(grouped_comments)
n_features = 10000

tfidf = TfidfVectorizer(min_df = 10, 
                         #max_df =.99,
                         max_features = n_features, 
                         stop_words = txt.ENGLISH_STOP_WORDS.union(additional_stop_words),
                         #tokenizer = LemmaTokenizer(),
                         sublinear_tf=True,
                         ngram_range=(1,2),
                         )

counts = tfidf.fit_transform(grouped_comments[:n_samples])
X = TfidfTransformer().fit_transform(counts)

print X.shape
print len(grouped_comments)
print type(X)
X_transformed = X.T
print X_transformed.shape

# <codecell>

#Saving the sparse matrix in a file
io.savemat('X.mat', {'M' : X}, oned_as='column')
io.savemat('X_transformed.mat', {'M' : X_transformed}, oned_as='column')

# <codecell>

#Saving the bag of words in a file
bag_of_words = tfidf.get_feature_names()
outfile = open('vocab.txt', 'w')
outfile.write("\n".join(bag_of_words))

# <codecell>

len(bag_of_words)

# <markdowncell>

# ##K Means Clustering

# <codecell>

#K-means clustering analysis

km = KMeans(n_clusters=30, init='k-means++', max_iter=100, n_init=1,
                verbose=True)
km_fit = km.fit_transform(X)
km_fit.shape

# <markdowncell>

# Below is from a [Stackoverflow article](http://stackoverflow.com/questions/12497252/how-can-i-cluster-document-using-kmean-flann-with-python):
# 
# There is one big problem here: K-means is designed for Euclidean distance.
# 
# The key problem is the mean function. The mean will reduce variance for Euclidean distance, but it might not do so for a different distance function. So in the worst case, k-means will no longer converge, but run in an infinite loop (although most implementations support stopping at a maximum number of iterations).
# 
# Furthermore, the mean is not very sensible for sparse data, and text vectors tend to be very sparse. Roughly speaking the problem is that the mean of a large number of documents will no longer look like a real document, and this way become dissimilar to any real document, and more similar to other mean vectors. So the results to some extend degenerate.
# 
# For text vectors, you probably will want to use a different distance function such as cosine similarity.
# 
# And of course you first need to compute number vectors. For example by using relative term frequencies, normalizing them via TF-IDF.
# 
# There is a variation of the k-means idea known as k-medoids. It can work with arbitrary distance functions, and it avoids the whole "mean" thing by using the real document that is most central to the cluster (the "medoid"). But the known algorithms for this are much slower than k-means.

# <markdowncell>

# ##K-Medoids Clustering
# > Good for very sparse matrix and also can use other distance measures
# 
# > Rather than using the mean (which is usually not a real datapoint), K-medoids uses a particular datapoint as the centroid. This makes K-medoids less sensitive to outliers. And in case of text, can give an actual "representative" document of a topic.
# 
# Valid values for metric are:
# from scikit-learn: [‘euclidean’, ‘l2’, ‘l1’, ‘manhattan’, ‘cityblock’]
# from scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘cosine’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’] See the documentation for scipy.spatial.distance for details on these metrics.

# <codecell>

#K-Medoid
#Can pass in other distances for K-Medoids
import Pycluster
from Pycluster import kmedoids

distances = pairwise_distances(X, metric='euclidean', n_jobs=1)
nb_clusters = 20
clusterid, error, nfound = Pycluster.kmedoids(distances, nclusters= nb_clusters, npass=100)

# <codecell>

#Look at the actual comments that are central to each cluster:
medoids = list(set(clusterid))

for m in medoids:
    print "------------------------"
    print "medoid ", m, ":", grouped_questions[m]

# <markdowncell>

# ##Non-Negative Matrix Factorization

# <codecell>

#NMF for grouped comments
n_topics = 30
n_top_words = 15

print("size of corpus: {}".format(len(grouped_comments)))

t0 = time()

# Fit the NMF model
print("Fitting the NMF model on with n_samples=%d and n_features=%d..."
      % (n_samples, X.shape[1]))
nmf = decomposition.NMF(n_components=n_topics).fit(X)
print("done in %0.3fs." % (time() - t0))

# Inverse the vectorizer vocabulary to be able
feature_names = tfidf.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join( [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] ))
    print

# <codecell>

#Getting the W and H for NMF model where X = dot(W , H)
W = nmf.fit_transform(X)
H = nmf.components_

# <codecell>

for topic_idx, topic in enumerate(nmf.components_):

    words = [feature_names[i] for i in topic.argsort()[:-(n_top_words):-1]]
    scores = [topic[i] for i in topic.argsort()[:-(n_top_words):-1]]

    dicta = {word: score for score, word in zip(scores, words)}

    s=pd.Series(dicta)
    plt.figure()
    plt.title("Topic #%d:" % topic_idx)
    s.plot(kind='bar');

# <codecell>

topics_dicts = []

for i in xrange(n_topics):
    # n_top_words of keys and values
    keys, values = zip(*sorted(zip(feature_names, H[i]), key = lambda x: x[1])[:-n_top_words:-1])
    val_arr = np.array(values)
    norms = val_arr / np.sum(val_arr)
    #normalize = lambda x: int(x / (max(counter.values()) - min(counter.values())) * 90 + 10)
    topics_dicts.append(dict(zip(keys, np.rint(norms* 300))))

topics_dicts

# <codecell>

import vincent

vincent.core.initialize_notebook()

for i in xrange(n_topics):
    word_cloud = vincent.Word(topics_dicts[i])
    word_cloud.width = 400
    word_cloud.height = 400
    word_cloud.padding = 0
    word_cloud.display()

#word_cloud.grammar();

# <markdowncell>

# ##Get the top 5 questions corresponding to each NMF topic

# <codecell>

FAQ = []
for t in xrange(n_topics):
    FAQ.append([grouped_questions[i] for i in np.argsort(W.T[t])[:-6:-1]])
    

# <codecell>

#Print out each topics and 5 questions that are most closely associated with that topic
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join( [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] ))
    print "---------------------------------------------"
    for q in FAQ[topic_idx]:
        print q
        print

# <markdowncell>

# ##Find cosine similarity between comments and the various topics as determined by NMF
# > Output: questions are "closest" to these NMF topics (as measured by cosine distances?)

# <codecell>

topics_bag = []
for topic_idx, topic in enumerate(nmf.components_):
    words = [feature_names[i] for i in topic.argsort()[::-1]]
    scores = [topic[i] for i in topic.argsort()[::-1]]
    topics_bag.append(zip(scores, words))

# topics_array = pd.DataFrame(topics_bag[1])
# topics_array.T
# sparse.csr_matrix(topics_array)
print len(topics_bag[1])
print "-----------------------------"
#print X[1]

# <codecell>

# cosine_similarities = linear_kernel(tfid, tfid)

# most_similar_index = []
# for similar in cosine_similarities:
#     most_similar_index.append(similar.argsort()[-10:][::-1])

# print len(most_similar_index)

# <codecell>

#Word clouds
# import vincent
# vincent.core.initialize_notebook()

# for topic_idx, topic in enumerate(nmf.components_):
#     print("Topic #%d:" % topic_idx)
#     tmpDic = {}
#     for i in topic.argsort()[:-50 - 1:-1]:
#         tmpDic[feature_names[i]] = topic[i]*100
#     vincent.Word(tmpDic).display();

# <codecell>

#Write file to EXCEL for labeling each comment by hand
# import csv
# reload(sys)
# sys.setdefaultencoding("utf-8")

# myfile = csv.writer(open("grouped_comments.csv", 'wb'))
# for g in grouped_comments:
#     myfile.writerow([g])

# <markdowncell>

# ##Latent Dirichlet Allocation (LDA) by using `gensim` library
# 
# > Gensim unfortunately isn't sklearn friendly.
# 
# > Take corpus again and run it through the mmcorpus. Then use it for LDA model in gensim.

# <codecell>

#NLTK-generated corpus vector
stop_words = txt.ENGLISH_STOP_WORDS.union(additional_stop_words)

def cleanDoc(doc):
    stopset = stop_words
    stemmer = nltk.PorterStemmer()
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = WordPunctTokenizer().tokenize(doc)
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
    stemmed = [stemmer.stem(word) for word in clean]
    final = [lemmatizer.lemmatize(word) for word in stemmed]
    return final


cleaned = []
for line in grouped_comments:
    cleaned.append(' '.join(cleanDoc(line)))

class MyCorpus(object):
    def __iter__(self):
        for line in cleaned:
            yield dictionary.doc2bow(line.lower().split())

corpus = MyCorpus()
len(cleaned)

# <codecell>

#corpus = gensim.matutils.Sparse2Corpus(X.T) #corpus vector directly from scipy sparse matrix
stop_words = txt.ENGLISH_STOP_WORDS.union(additional_stop_words)

# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in cleaned)

# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stop_words 
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() 
            if docfreq == 1 or docfreq == 2 or docfreq == 3 or docfreq == 4]
dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed
dictionary.save('questions.dict')
print(dictionary)

corpora.MmCorpus.serialize('questions.mm', corpus)
corpora.SvmLightCorpus.serialize('questions.svmlight', corpus)
corpora.BleiCorpus.serialize('questions.lda-c', corpus)
#corpora.LowCorpus.serialize('questions.low', corpus)

# ##Then the dictionary and corpus can be used to train using LDA

mm = corpora.MmCorpus('questions.mm')
blei = corpora.BleiCorpus('questions.lda-c')
#low = corpora.LowCorpus('questions.low')

# <codecell>

print(mm)

# <codecell>

#LDA model
import warnings
warnings.filterwarnings('ignore')

lda = gensim.models.ldamodel.LdaModel(corpus = blei, 
                                      id2word = dictionary, 
                                      num_topics = n_topics, 
                                      #update_every = 1, 
                                      #chunksize = 10000, 
                                      passes = 4,
                                      )

# <codecell>

for i in range(0, lda.num_topics-1):
    print lda.print_topic(i)

# <codecell>

#Matching NMF topics with LDA topics

nmf_lda=[]
for topic_idx, topic in enumerate(nmf.components_):
    words = [feature_names[i] for i in topic.argsort()[:-(n_top_words):-1]]
    nmf_lda.append(lda[dictionary.doc2bow(words)])

for k, v in enumerate(nmf_lda): 
    print "NMF topic: #", k, "matching LDA topic: #", max(nmf_lda[k])[0]

# <markdowncell>

# ## Topic Model using Anchor word algorithm
# 
# - "Separability requires that each topic has some near-perfect indicator word – a word that we call the anchor word for this topic— that appears with reasonable probability in that topic but with negligible probability in all other topics (e.g., “soccer” could be an anchor word for the topic “sports”). We give a formal definition in Section 1.1. This property is particularly natural in the context of topic modeling, where the number of distinct words (dictionary size) is very large compared to the number of topics. In a typical application, it is common to have a dictionary size in the thousands or tens of thousands, but the number of topics is usually somewhere in the range from 50 to 100. Note that separability does not mean that the anchor word always occurs (in fact, a typical document may be very likely to contain no anchor words). Instead, it dictates that when an anchor word does occur, it is a strong indicator that the corresponding topic is in the mixture used to generate the document." -- [A Practical Algorithm for Topic Modeling with Provable Guarantees](http://arxiv.org/pdf/1212.4777.pdf) by Sanjeev Arora; Rong Ge; Yonatan Halpern; David Mimno; Ankur Moitra; David Sontag; Yichen Wu; Michael Zhu

# <codecell>

import os

os.popen("/usr/local/Cellar/stanford-parser/3.3.1/libexec/lexparser.sh /usr/local/Cellar/stanford-parser/3.3.1/libexec/stupid.txt > ~/Desktop/parsed.txt")

# <codecell>


