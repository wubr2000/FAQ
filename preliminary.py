# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
import sys
import logging
import seaborn 
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics.pairwise import linear_kernel,pairwise_distances
from sklearn.feature_extraction import text as txt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer, HashingVectorizer
from IPython.display import display, Math, Latex

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation

from optparse import OptionParser

%pylab inline

# <codecell>

#Discussion forums specific stopwords
additional_stop_words = [
                         "zipredac","phoneredac","nameRedac_anon_screen_name_redacted",
                         "explanation", "helpful", "yes",
                         "agree","excellent","dear",
                         "ok", "cool", "idea", "guys",
                         "hello", "student", "sorry", "makes",
                         "sense", "nice", "meet", "yeah", "wow",
                         "hint", "post", "understand", "helped",
                         ]

# <codecell>

forum = pd.read_csv("statsLearningForum.csv", quoting = 2, header = None, 
                    escapechar = "\\", encoding = "ISO-8859-1")

# <codecell>

forum.columns = ["anon_screen_name", "type", "anonymous", "anonymous_to_peers", 
                 "at_position_list", "user_int_id", "body", "course_display_name", 
                 "created_at", "votes", "count", "down_count", "up_count", "up", "down",
                 "comment_thread_id", "parent_id",  "parent_ids", "sk","extra"]

# <codecell>

pd.options.display.max_colwidth=110
#forum.head()

# <codecell>

del forum["anon_screen_name"]
del forum["at_position_list"]
del forum["extra"]
del forum["course_display_name"]

# <codecell>

forum.body.head()

# <codecell>

#Create "final_corpus", "all_questions", and "all_comments"

#"all_questions" are not only rows labeled "CommentThread" but also rows where there is a question mark - i.e.
#clearly a question and not just someone making a statement to initiate a chain.

corpus = []
for body in forum.body:
    corpus.append(''.join(body))    

#Clean junk in strings
final_corpus = []
for c in corpus:
    unicode(c)
    final_corpus.append(c.replace("\n"," ").replace("\n\n"," "))

print len(final_corpus)

questions = [ f for f in forum.body[forum.type=="CommentThread"] if "?" in f ]
#Clean junk in strings
all_questions = []
for q in questions:
    unicode(q)
    all_questions.append(q.replace("\n"," ").replace("\n\n"," "))

print len(all_questions)

all_comments = [ comment for comment in final_corpus if comment not in all_questions ]

print len(all_comments)

# <codecell>

#TF-IDF calculation on "all_questions"
#Prior to any topic modeling or clustering
#This doesn't work very well - take a look at the questions that are grouped together

tfidf = TfidfVectorizer(encoding='utf-8', lowercase = True, 
                        stop_words = txt.ENGLISH_STOP_WORDS.union(additional_stop_words))
tfid = tfidf.fit_transform(all_questions)

print tfid.shape

# <codecell>

#Get the top 10 most similar questions using cosine similarity for each question

cosine_similarities = linear_kernel(tfid, tfid)

most_similar_index = []
for similar in cosine_similarities:
    most_similar_index.append(similar.argsort()[-10:][::-1])

print len(most_similar_index)

# <codecell>

for questions in most_similar_index[:10]:
    print "------------------------"
    for q in questions:
        print q, ":", all_questions[q]

# <markdowncell>

# >Using pure cosine similarity between questions doesn't work too well. 

# <markdowncell>

# > Adjusting `min_df` and `max_df` affects the results ALOT!!! They will essentially get rid of "OUTLIERS". Tell function to ignore these words because they occurred in either too many (`max_df`) or in too few (`min_df`) documents.
# 
# > Setting `min_df`= X means that calculation will include only words that appear in at least X documents. This will throw out many really weird, idiosyncratic, or made-up words that appear rarely and that are not germane to the comment.
# 
# > Setting `max_df` = Y means that calculation will exclude words that appear over and over again in every documents. For example, in these forum discussions, there are alot of "thanks", "good", "agree","helpful","helps", "thanks alot" type of words that will throw off calculation. Again, they are not germane to the discussion but will throw off the calcuation.
# 
# > My approach here was to include as many "features" or words as possible but use these to "adjust" for outliers.
# 
# > Bigrams and higher tend to make the results worse.
# 
# > Need to include additional stopwords like "dear", "hint", "makes", "sense", "helpful", "explanation", etc. that dominates a topic.
#  

# <markdowncell>

# ##Group all comments together first by `comment_thread_id`

# <codecell>

#Need to figure out how comments and questions are grouped together
#forum[forum.type == "CommentThread"].head()

#Concatenate all comments together that has the same "comment_thread_id" - i.e. jam all the comments on one 
#question together - this should be better for analysis
concat_comments = forum[forum.type=="Comment"].sort("comment_thread_id").groupby("comment_thread_id").body.sum()

grouped_comments=[]
for c in concat_comments:
    unicode(c)
    grouped_comments.append(c.replace("\n"," ").replace("\n\n"," "))

# <markdowncell>

# ##K Means Clustering

# <codecell>

#K-means clustering analysis

tfidf = TfidfVectorizer(encoding='utf-8', lowercase = True, 
                        stop_words = txt.ENGLISH_STOP_WORDS.union(additional_stop_words))
X = tfidf.fit_transform(grouped_comments)
print X.shape
print len(grouped_comments)

# <codecell>

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
nb_clusters = 50
clusterid, error, nfound = Pycluster.kmedoids(distances, nclusters= nb_clusters, npass=100)

# <codecell>

#Look at the actual comments that are central to each cluster:
medoids = list(set(clusterid))

for m in medoids:
    print "------------------------"
    print "medoid ", m, ":", grouped_comments[m]

# <markdowncell>

# ##Non-Negative Matrix Factorization

# <codecell>

#NMF for grouped comments

from sklearn import decomposition
from time import time

print("size of corpus: {}".format(len(grouped_comments)))

n_samples = 1000
n_features = 10000
n_topics = 50
n_top_words = 20

t0 = time()

vectorizer = TfidfVectorizer(min_df=10, max_df=150,
                             max_features=n_features, 
                             stop_words = txt.ENGLISH_STOP_WORDS.union(additional_stop_words))

counts = vectorizer.fit_transform(grouped_comments[:n_samples])
tfidf = TfidfTransformer().fit_transform(counts)
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model on with n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
nmf = decomposition.NMF(n_components=n_topics).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

# Inverse the vectorizer vocabulary to be able
feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join( [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] ))
    print

# <codecell>

for topic_idx, topic in enumerate(nmf.components_):

    words = [feature_names[i] for i in topic.argsort()[:-(n_top_words-12):-1]]
    scores = [topic[i] for i in topic.argsort()[:-(n_top_words-12):-1]]

    dicta = {word: score for score, word in zip(scores, words)}

    s=pd.Series(dicta)
    plt.figure()
    plt.title("Topic #%d:" % topic_idx)
    s.plot(kind='bar');

# <codecell>

#Word clouds
import vincent
vincent.core.initialize_notebook()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    tmpDic = {}
    for i in topic.argsort()[:-50 - 1:-1]:
        tmpDic[feature_names[i]] = topic[i]*100
    vincent.Word(tmpDic).display();

# <codecell>

#Write file to EXCEL for labeling each comment by hand
# import csv
# reload(sys)
# sys.setdefaultencoding("utf-8")

# myfile = csv.writer(open("grouped_comments.csv", 'wb'))
# for g in grouped_comments:
#     myfile.writerow([g])

# <codecell>


