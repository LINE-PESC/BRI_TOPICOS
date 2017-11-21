#!/usr/bin/python3

########################################
## import packages
########################################
from time import time

from tqdm import tqdm
from gensim import corpora, models


########################################
## Configurations
########################################
n_features = 500
n_components = 500
n_top_words = 20


########################################
## Support Functions
########################################


## print top 10 topics
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

########################################
## Applying LDA
########################################

#tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
#
#t0 = time()
#tf = tf_vectorizer.fit_transform(stopped_tokens)
#print("Vectorizing done in %0.3fs." % (time() - t0))
#
#lda = LatentDirichletAllocation(n_topics=n_components, max_iter=10, learning_method='online', learning_offset=50., random_state=0)
#
#t0 = time()
#lda.fit(tf)
#print("LDA done in %0.3fs." % (time() - t0))
#
#
#tf_feature_names = tf_vectorizer.get_feature_names()
#print_top_words(lda, tf_feature_names, n_top_words)
