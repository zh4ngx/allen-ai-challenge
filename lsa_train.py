"""
  Apply Latent Semantic Analysis to Wikipedia dataset

  Run this after running `python -m gensim.scripts.make_wiki` on enwiki-latest-pages-articles.xml.bz2
  Args (unlabeled):
    input_dicttionary: Path to wiki_en_wordids.txt.bz2
    input_corpus: path to wiki_en_tfidf.mm
    output_model: somewhere to save the LSA model
  Use LSA to extract latent vectors
"""

import logging, gensim, bz2, sys

logging.basicConfig(
  format='%(asctime)s : %(levelname)s : %(message)s',
  level=logging.INFO
)

input_dictionary = sys.argv[1]
input_corpus = sys.argv[2]
output_model = sys.argv[3]

# load id->word mapping (the dictionary)
id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File(input_dictionary))

# load corpus iterator
mm = gensim.corpora.MmCorpus(input_corpus)

print(mm)
# MmCorpus(3933461 documents, 100000 features, 612118814 non-zero entries)

# extract num_topics LSI topics; use the default one-pass algorithm
num_topics = 400
model = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=num_topics)

# print the most contributing words (both positively and negatively) for each of the first ten topics
model.print_topics(10)

model.save(output_model)
