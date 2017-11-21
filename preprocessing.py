#!/usr/bin/python3

########################################
## import packages
########################################
from __future__ import print_function

import os
import csv
import requests

from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

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

tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()

########################################
## Support Functions
########################################
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
                raw = f.read().replace('\n', '')
                docs_vector.append(raw)
        return docs_vector
    else:
        return 0

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def clearList(inputList):
    clearedList = [word for word in inputList if (len(str(word)) > 2 and not hasNumbers(str(word)))]
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
        return tags.values.tolist()
    return tags

########################################
## Pre processing
########################################

raw = vectorize_docs('python')
tags = get_stackoverflow_tags()
clearedTags = clearList(tags[0])

tokenized_docs = []
for text in raw:
    tokenized_docs.append(tokenizer.tokenize(str(text.lower())))

tokenized_docs = [[word for word in document if word not in STOP_WORDS] for document in tokenized_docs]

tokenized_docs = [[word for word in document if word not in clearedTags] for document in tokenized_docs]

cleared_docs = []
for document in tokenized_docs:
    cleared_docs.append(clearList(document))

frequency = defaultdict(int)

## Remove all words that appear only once
for document in cleared_docs:
    for token in document:
        frequency[token] += 1

cleared_docs = [[token.lower() for token in document if frequency[token] > 1] for document in cleared_docs]

dictionary = corpora.Dictionary(cleared_docs)
dictionary.save('./text.dict')
corpus = [dictionary.doc2bow(doc) for doc in cleared_docs]
corpora.MmCorpus.serialize('./text.mm', corpus)