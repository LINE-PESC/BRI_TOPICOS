#!/usr/bin/python3

########################################
## import packages
########################################
from __future__ import print_function

from time import time
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20

with open(RAILS_DOCS_DIR+'gettingstarted.txt') as f:
    raw = tqdm(f.readlines(), desc='Reading document lines')
    raw = [x.strip() for x in raw]

tokenizer = RegexpTokenizer(r'\w+')
tokens = tqdm(tokenizer.tokenize(str(raw)), desc='Tokenizing words')
filtered_words = [word for word in tokens if word not in STOP_WORDS]

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(filtered_words)

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')

tf = tf_vectorizer.fit_transform(filtered_words)

nmf = NMF(n_components=n_components, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()

nmf = NMF(n_components=n_components, random_state=1, solver='cd', max_iter=1000, alpha=.1, l1_ratio=.5).fit(tfidf)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()

lda = LatentDirichletAllocation(n_topics=n_components, max_iter=10, learning_method='online', learning_offset=50., random_state=0)

lda.fit(tf)

tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
