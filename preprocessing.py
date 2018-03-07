#!/usr/bin/python3

########################################
## import packages
########################################
from __future__ import print_function

import os
import re
import csv
import requests
import logging

from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from gensim import corpora
from nltk.corpus import stopwords

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


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

########################################
## Pre processing
########################################

documents = vectorize_docs('python')

## Stackoverflow tags pre processing
tags = get_stackoverflow_tags()
tags = removeSpecialCharacters(tags)
tags = removeNumbers(tags)
cleared_tags = tags[0]

## Create tokens from documents
tokenized_docs = []
tokenized_docs = [[word for word in document.lower().split() if word not in STOP_WORDS] for document in documents]


## Remove special characters and numbers
tokenized_docs = removeSpecialCharacters(tokenized_docs)
cleared_docs = removeNumbers(tokenized_docs)

frequency = defaultdict(int)

## Remove all words that appear only once
for document in cleared_docs:
    for token in document:
        frequency[token] += 1

cleared_docs = [[token.lower() for token in document if frequency[token] > 1] for document in cleared_docs]

## Remove all words that are not stackoverflow tags
cleared_docs = [[word for word in document if word in cleared_tags] for document in cleared_docs]


## Save dictionary in serialized form
dictionary = corpora.Dictionary(cleared_docs)
dictionary.save('./dictionaries/python_tags.dict')
corpus = [dictionary.doc2bow(document) for document in cleared_docs]
corpora.MmCorpus.serialize('./dictionaries/python_tags.mm', corpus)