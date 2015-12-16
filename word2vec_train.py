"""
  Apply Word2Vec to Wikipedia dataset

  Reference: https://code.google.com/p/word2vec/

    Args (unlabeled):
    input_articles: Path to articles.xml
    output_model: somewhere to save the LSA model
"""

import argparse
import logging
import multiprocessing
import os

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from utils import generate_timestamp

timestamp = generate_timestamp()

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
)

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--articles", help="path to enwiki-latest-pages-articles.xml.bz2")
parser.add_argument("-m", "--model", help="path to model dir")
parser.add_argument("-d", "--demo", help="path to question-words.txt analogies")
parser.add_argument("-l", "--lines", help="path to wiki-lines.txt")
args = parser.parse_args()

# Load or create wiki-lines.txt
if not (os.path.isfile(args.lines)):
    wiki_corpus = WikiCorpus(args.articles, lemmatize=False)
    wiki_lines = wiki_corpus.get_texts()

    # Write wiki_lines out for future use
    lines_file = open(args.lines, 'w')
    for text in wiki_lines:
        lines_file.write(" ".join(text) + "\n")
    lines_file.close()
else:
    wiki_lines = open(args.lines)

model = Word2Vec(
        sentences=LineSentence(wiki_lines),
        size=400,
        hs=1,
        window=5,
        min_count=5,
        sample=1e-5,
        iter=2,
        workers=multiprocessing.cpu_count()
)

model.save("%s/%s.model" % (args.model, timestamp))

# Evaluate using analogy file:
# https://word2vec.googlecode.com/svn/trunk/questions-words.txt
if args.demo:
    model.accuracy(open(args.demo))
