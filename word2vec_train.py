"""
  Apply Word2Vec to Wikipedia dataset

  Reference: https://code.google.com/p/word2vec/

    Args (unlabeled):
    input_articles: Path to articles.xml
    output_model: somewhere to save the LSA model
"""

import datetime
import logging
import multiprocessing
import os
import sys

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec, Phrases
from gensim.models.word2vec import LineSentence

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
)

input_articles = sys.argv[1]
output_model = sys.argv[2]
demo_questions = sys.argv[3]  # question-words.txt analogy example
output_lines = sys.argv[4]

# Load or create wiki-lines.txt
if not (os.path.isfile(output_lines)):
    wiki_corpus = WikiCorpus(input_articles, lemmatize=False)
    wiki_lines = wiki_corpus.get_texts()

    # Write wiki_lines out for future use
    lines_output = open(output_lines, 'w')
    for text in wiki_lines:
        lines_output.write(" ".join(text) + "\n")
    lines_output.close()
else:
    wiki_lines = open(output_lines)

# Load or create bigram transformer
if not (os.path.isfile("%s/bigram_transformer" % output_model)):
    bigram_transformer = Phrases(LineSentence(wiki_lines))
else:
    bigram_transformer = Phrases.load("%s/bigram_transformer" % output_model)

model = Word2Vec(
        sentences=bigram_transformer[LineSentence(wiki_lines)],
        size=400,
        hs=1,
        sample=1e-5,
        window=5,
        min_count=5,
        workers=multiprocessing.cpu_count()
)

model.save("%s/%s.model" % (output_model, timestamp))
bigram_transformer.save("%s/bigram_transformer" % output_model)

# Evaluate using analogy file:
# https://word2vec.googlecode.com/svn/trunk/questions-words.txt
model.accuracy(open(bigram_transformer[demo_questions]))
