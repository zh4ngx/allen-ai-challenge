#!/usr/local/bin/python

"""
  Apply Latent Semantic Analysis to Wikipedia dataset

  Run this after running `python -m gensim.scripts.make_wiki` on enwiki-latest-pages-articles.xml.bz2
  Use LSA to extract latent vectors
  Use latent vectors to calculate cosine distance between Questions/Answers
  Pick the answer with the smallest cosine distance
  Check against correct answer
"""

import logging, gensim, bz2

logging.basicConfig(
  format='%(asctime)s : %(levelname)s : %(message)s',
  level=logging.INFO
)

# load id->word mapping (the dictionary)
id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File('./data/wiki_en_wordids.txt.bz2'))

# load corpus iterator
mm = gensim.corpora.MmCorpus('./data/wiki_en_tfidf.mm')

print(mm)
