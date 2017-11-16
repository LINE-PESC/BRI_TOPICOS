#!/usr/bin/python3

########################################
## import packages
########################################
from __future__ import print_function

import os
import csv
import enchant
import requests
from time import time

from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

########################################
## Configurations
########################################
DOCS_DIR = './docs/'
RAILS_DOCS_DIR = DOCS_DIR + 'rails/'
PYTHON_DOCS_DIR = DOCS_DIR + 'python/'
STOP_WORDS = set(stopwords.words('english'))

n_features = 1000
n_components = 10
n_top_words = 20

## Stackoverflow page request parameters
pagesize = 100
site = 'stackoverflow'
key = 'U5pNnFT8KfJCieV9wGb5uQ(('

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

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
                raw = [x.lower() for x in raw]
                docs_vector.append(raw)
        return docs_vector
    else:
        return 0

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def clearList(inputList):
    clearedList = [word for word in inputList if len(str(word)) > 2 and not hasNumbers(str(word))]
    return clearedList

def get_stackoverflow_tags():
    if not os.path.isfile('./stackoverflow_tags.csv'):
        has_more = True
        tags = []
        page = 1
        while(has_more):
            request = requests.get('https://api.stackexchange.com/2.2/tags',params={'pagesize' : pagesize,'site' : site, 'page' : page  ,'key' : key}, verify=True, stream=True).json()
            if 'has_more' in request:
                page = page + 1 
                for item in request['items']:
                    tags.append(item['name'])
            else:
                has_more = False
        with open('./stackoverflow_tags.csv', 'wb') as outfile:
            writer = csv.writer(outfile,delimiter=';')
            writer.writerow(tags)
    else:
        tags = pd.read_csv('./stackoverflow_tags.csv', header = None, delimiter = ';')    
    return tags

########################################
## Pre processing
########################################

raw = vectorize_docs('python')
tags = get_stackoverflow_tags()
clearedTags = clearList(tags)

dictionary = corpora.Dictionary(clearedTags)
dictionary.save('./terms.dict')

print(raw[0])

t0 = time()
tokenized_docs = []
for document in raw:
    for token in document:
        #tmp.append(''.join(tokenizer.tokenize(str(token)))
        break

    tokenized_docs.append(tmp)

print("Tokenizing done in %0.3fs." % (time() - t0))

clearedDocs = clearList([document for document in tokenized_docs])
print(clearedDocs)
frequency = defaultdict(int)
for document in clearedDocs:
    for token in document:
        frequency[token] += 1

clearedDocs = [[token.lower() for token in document if frequency[token] > 1] for document in clearedDocs]

########################################
## Applying LSI
########################################
#dictionary = corpora.dictionary(stopped_tokens)


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
