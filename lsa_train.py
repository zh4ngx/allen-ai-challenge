"""
  Apply Latent Semantic Analysis to Wikipedia dataset

  Run this after running `python -m gensim.scripts.make_wiki` on enwiki-latest-pages-articles.xml.bz2
  Args (unlabeled):
    input_dictionary: Path to wiki_en_wordids.txt.bz2
    input_corpus: path to wiki_en_tfidf.mm
    output_model: somewhere to save the LSA model
  Use LSA to extract latent vectors
"""
import argparse
import bz2
import logging

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LsiModel

from utils import generate_timestamp

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
)
timestamp = generate_timestamp()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dictionary", help="path to wiki_en_wordids.txt")
parser.add_argument("-c", "--corpus", help="path to wiki_en_tfidf.mm")
parser.add_argument("-m", "--model", help="path to model output")
args = parser.parse_args()

# load id->word mapping (the dictionary)
id2word = Dictionary.load_from_text(bz2.BZ2File(args.dictionary))

# load corpus iterator
mm = MmCorpus(args.corpus)

print(mm)
# MmCorpus(3933461 documents, 100000 features, 612118814 non-zero entries)

# extract num_topics LSI topics; use the default one-pass algorithm
num_topics = 400
model = LsiModel(corpus=mm, id2word=id2word, num_topics=num_topics)

# print the most contributing words (both positively and negatively) for each of the first ten topics
model.print_topics(10)

model.save("%s/%s.model" % (args.model, timestamp))
