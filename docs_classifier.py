#!/usr/bin/python3

########################################
## import packages
########################################
from __future__ import print_function

from tqdm import tqdm
from time import time

import os

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

########################################
## print top 10 topics
########################################
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

STOP_WORDS = set(stopwords.words('english'))

DOCS_DIR = './docs/'
RAILS_DOCS_DIR = DOCS_DIR + 'rails/'
PYTHON_DOCS_DIR = DOCS_DIR + 'python/'

def vectorize_docs(lang):
    docs_vector = []
    if lang == 'ruby':
        with open(RAILS_DOCS_DIR+'gettingstarted.txt') as f:
            raw = tqdm(f.readlines(), desc='Reading ruby document lines')
            raw = [x.strip() for x in raw]
            return raw
    elif lang == 'python':
        python_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(PYTHON_DOCS_DIR)
             for name in files]
        for file in python_files:
            with open(file) as f:
                raw = tqdm(f.readlines(), desc='Reading ' + file + ' document lines')
                raw = [x.strip().lower() for x in raw]
                docs_vector.append(raw)
        return docs_vector
    else:
        return 0

n_features = 10000
n_components = 10
n_top_words = 20

raw = vectorize_docs('python')

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

t0 = time()
tokens = tokenizer.tokenize(str(raw))
print("Tokenizing done in %0.3fs." % (time() - t0))

print('Number of words: ' + str(len(tokens)))

t0 = time()
stopped_tokens = [word for word in tokens if not word in STOP_WORDS]
print("Removing stop words done in %0.3fs." % (time() - t0))

t0 = time()
stemmed_tokens = [stemmer.stem(word) for word in stopped_tokens]
print("Stemming words done in %0.3fs." % (time() - t0))


tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')

t0 = time()
tf = tf_vectorizer.fit_transform(tokens)
print("Vectorizing done in %0.3fs." % (time() - t0))

lda = LatentDirichletAllocation(n_topics=n_components, max_iter=15, learning_method='online', learning_offset=50., random_state=0)

t0 = time()
lda.fit(tf)
print("LDA done in %0.3fs." % (time() - t0))


tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
