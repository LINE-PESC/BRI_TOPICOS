#!/usr/bin/python3

########################################
## import packages
########################################

import os
import logging

from time import time

from gensim import corpora, models, similarities


########################################
## Configurations
########################################
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


########################################
## Load Data
########################################
if (os.path.exists("./dictionaries/training_wo_tags.dict")):
    dictionary = corpora.Dictionary.load('./dictionaries/training_wo_tags.dict')
    corpus = corpora.MmCorpus('./dictionaries/training_wo_tags.mm')
    print("Used dictionary generated")
else:
    print("Please run the preprocessing to generate a dictionary file")

########################################
## Create Model
########################################
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

for doc in corpus_tfidf:
    print(doc)

########################################
## Applying LSI
########################################
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(2)