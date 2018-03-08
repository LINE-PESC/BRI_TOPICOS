#!/usr/bin/python3

########################################
## import packages
########################################
import os
import re
import csv
import logging
import warnings
import requests

import pandas as pd

from operator import itemgetter
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel, LsiModel, TfidfModel, CoherenceModel

import matplotlib.pyplot as plt
import pyLDAvis.gensim

########################################
## Configurations
########################################

DOCS_DIR = './docs/'
RAILS_DOCS_DIR = DOCS_DIR + 'rails/'
PYTHON_DOCS_DIR = DOCS_DIR + 'python/'
STOP_WORDS = set(stopwords.words('english'))

## Stackoverflow page request parameters
pagesize = 100
site = 'stackoverflow'
key = 'U5pNnFT8KfJCieV9wGb5uQ(('
wordnet_lemmatizer = WordNetLemmatizer()

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

warnings.filterwarnings('ignore')

#%matplotlib inline

########################################
## Support Functions
########################################
def vectorize_docs(lang):
    docs_vector = []
    if lang == 'ruby':
        with open(RAILS_DOCS_DIR+'gettingstarted.txt') as f:
            raw = f.read().replace('\n', '')
            docs_vector.append(raw)
            return docs_vector
    elif lang == 'python':
        python_files = [os.path.join(root, name)
             for root, dirs, files in os.walk(PYTHON_DOCS_DIR)
             for name in files]
        for file in python_files:
            with open(file) as f:
                raw = f.read().replace('\n', ' ')
                docs_vector.append(raw)
        return docs_vector
    else:
        return 0

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def removeSpecialCharacters(documents):
   return [[re.sub('[^A-Za-z0-9]+', '', str(word)) for word in document] for document in documents]

def removeNumbers(documents):
    return [[word for word in document if (len(str(word)) > 2 and not hasNumbers(str(word)))] for document in documents]

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
        return tags.values.tolist()
    return tags


def evaluate_graph(dictionary, corpus, texts, limit):
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, update_every=1, chunksize=100, passes=1)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())

    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()

    return lm_list, c_v
########################################
## Pre processing
########################################

documents = vectorize_docs('python')

## Stackoverflow tags pre processing
tags = get_stackoverflow_tags()
tags = removeSpecialCharacters(tags)
tags = removeNumbers(tags)
cleared_tags = tags[0]

## Create tokens from documents and check if not in STOP_WORDS
tokenized_docs = []
tokenized_docs = [[word for word in document.lower().split() if word not in STOP_WORDS] for document in documents]

## Remove special characters and numbers
tokenized_docs = removeSpecialCharacters(tokenized_docs)
cleared_docs = removeNumbers(tokenized_docs)

## Lemmatize docs
tokenized_docs = [[wordnet_lemmatizer.lemmatize(word) for word in document] for document in tokenized_docs]

## Remove all words that are not stackoverflow tags
cleared_docs = [[word for word in document if word in cleared_tags] for document in cleared_docs]

## Remove all words that appear only once
frequency = defaultdict(int)

for document in cleared_docs:
    for token in document:
        frequency[token] += 1

cleared_docs = [[token.lower() for token in document if frequency[token] > 1] for document in cleared_docs]

trash_tokens = []

for k, v in sorted(frequency.items(), key=itemgetter(1), reverse=True):
    trash_tokens.append(k)

cleared_tokens = trash_tokens[int(len(trash_tokens) * 0.10) : int(len(trash_tokens) * .90)]

cleared_docs = [[token for token in document if token in cleared_tokens] for document in cleared_docs]

## Save dictionary in serialized form
dictionary = Dictionary(cleared_docs)
dictionary.save('./dictionaries/python_tags.dict')
corpus = [dictionary.doc2bow(document) for document in cleared_docs]
MmCorpus.serialize('./dictionaries/python_tags.mm', corpus)

########################################
## Load Data
########################################
if (os.path.exists("./dictionaries/python_tags.dict")):
    dictionary = Dictionary.load('./dictionaries/python_tags.dict')
    corpus = MmCorpus('./dictionaries/python_tags.mm')
    print("Used dictionary generated")
else:
    print("Please run the preprocessing to generate a dictionary file")

########################################
## Create Model
########################################
print(corpus)
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

########################################
## Applying LSI
########################################
lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=400, decay=1, onepass=False, extra_samples=20)
corpus_lsi = lsi[corpus_tfidf]
print(lsi.show_topics(num_topics=10))
lsitopics = lsi.show_topics(formatted=False)

########################################
## Applying LDA
########################################
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=400, update_every=1, chunksize=100, passes=1)
print(lda.show_topics(num_topics=10))
ldatopics = lda.show_topics(formatted=False)

########################################
## Generate Intertropic Distance Map
########################################
#pyLDAvis.enable_notebook()
#pyLDAvis.gensim.prepare(lda, corpus, dictionary)

########################################
## Generate Topic Coherence Graph
########################################
#lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=cleared_docs, limit=50)