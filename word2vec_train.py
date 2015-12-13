"""
  Apply Word2Vec to Wikipedia dataset

    Args (unlabeled):
    input_articles: Path to articles.xml
    output_model: somewhere to save the LSA model
"""

import logging, sys, datetime

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logging.basicConfig(
  format='%(asctime)s : %(levelname)s : %(message)s',
  level=logging.INFO
)

input_articles = sys.argv[1]
output_model = sys.argv[2]
demo_questions = sys.argv[3] # question-words.txt analogy example

wiki_lines = WikiCorpus(input_articles, lemmatize=False)

model = Word2Vec(LineSentence(wiki_lines), window=5, min_count=5, workers=multiprocessing.cpu_count())

# Try out with word2vec google page's question-words
model.accuracy(demo_questions)

model.save("%s/%s" % (output_model, timestamp))
