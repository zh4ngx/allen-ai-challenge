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
# MmCorpus(3933461 documents, 100000 features, 612118814 non-zero entries)

# extract num_topics LSI topics; use the default one-pass algorithm
num_topics = 400
lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=num_topics)

# print the most contributing words (both positively and negatively) for each of the first ten topics
lsi.print_topics(10)

# 2015-11-17 00:21:23,576 : INFO : topic #0(160.363): 0.145*"album" + 0.098*"song" + 0.097*"population" + 0.097*"league" + 0.090*"band" + 0.083*"town" + 0.083*"station" + 0.080*"district" + 0.079*"village" + 0.076*"chart"
# 2015-11-17 00:21:23,579 : INFO : topic #1(124.548): 0.283*"population" + 0.242*"median" + 0.236*"census" + -0.216*"album" + 0.214*"households" + 0.200*"income" + 0.171*"township" + 0.165*"females" + 0.164*"average" + 0.164*"males"
# 2015-11-17 00:21:23,583 : INFO : topic #2(114.006): 0.379*"album" + 0.226*"song" + 0.225*"chart" + -0.193*"league" + 0.185*"band" + 0.149*"vocals" + 0.137*"guitar" + -0.132*"football" + 0.122*"median" + 0.115*"track"
# 2015-11-17 00:21:23,586 : INFO : topic #3(104.613): -0.393*"league" + -0.251*"football" + -0.234*"cup" + -0.218*"club" + -0.159*"goals" + 0.129*"station" + -0.120*"player" + -0.115*"games" + -0.113*"apps" + -0.106*"game"
# 2015-11-17 00:21:23,590 : INFO : topic #4(90.690): 0.573*"station" + 0.248*"railway" + -0.215*"party" + -0.173*"election" + 0.141*"village" + 0.117*"fm" + 0.099*"river" + 0.093*"road" + 0.093*"radio" + -0.088*"elected"
# 2015-11-17 00:21:23,676 : INFO : topic #0(160.363): 0.145*"album" + 0.098*"song" + 0.097*"population" + 0.097*"league" + 0.090*"band" + 0.083*"town" + 0.083*"station" + 0.080*"district" + 0.079*"village" + 0.076*"chart"
# 2015-11-17 00:21:23,679 : INFO : topic #1(124.548): 0.283*"population" + 0.242*"median" + 0.236*"census" + -0.216*"album" + 0.214*"households" + 0.200*"income" + 0.171*"township" + 0.165*"females" + 0.164*"average" + 0.164*"males"
# 2015-11-17 00:21:23,683 : INFO : topic #2(114.006): 0.379*"album" + 0.226*"song" + 0.225*"chart" + -0.193*"league" + 0.185*"band" + 0.149*"vocals" + 0.137*"guitar" + -0.132*"football" + 0.122*"median" + 0.115*"track"
# 2015-11-17 00:21:23,686 : INFO : topic #3(104.613): -0.393*"league" + -0.251*"football" + -0.234*"cup" + -0.218*"club" + -0.159*"goals" + 0.129*"station" + -0.120*"player" + -0.115*"games" + -0.113*"apps" + -0.106*"game"
# 2015-11-17 00:21:23,690 : INFO : topic #4(90.690): 0.573*"station" + 0.248*"railway" + -0.215*"party" + -0.173*"election" + 0.141*"village" + 0.117*"fm" + 0.099*"river" + 0.093*"road" + 0.093*"radio" + -0.088*"elected"
# 2015-11-17 00:21:23,695 : INFO : topic #5(85.940): -0.306*"district" + -0.280*"village" + -0.243*"party" + -0.221*"election" + -0.163*"municipality" + -0.145*"church" + -0.131*"album" + -0.111*"gmina" + 0.110*"station" + -0.108*"population"
# 2015-11-17 00:21:23,699 : INFO : topic #6(84.064): -0.477*"station" + -0.294*"party" + -0.247*"election" + 0.197*"church" + 0.188*"village" + 0.171*"species" + -0.163*"railway" + -0.128*"fm" + -0.121*"radio" + -0.108*"elections"
# 2015-11-17 00:21:23,706 : INFO : topic #7(78.752): 0.495*"species" + -0.275*"church" + 0.190*"genus" + 0.114*"party" + -0.106*"village" + 0.103*"mm" + -0.092*"historic" + 0.092*"river" + 0.086*"plant" + 0.084*"election"
# 2015-11-17 00:21:23,711 : INFO : topic #8(76.934): -0.328*"championships" + 0.235*"league" + -0.207*"medal" + -0.199*"olympics" + 0.193*"species" + -0.154*"olympic" + -0.146*"event" + -0.140*"men" + -0.130*"metres" + -0.129*"village"
# 2015-11-17 00:21:23,715 : INFO : topic #9(75.469): 0.277*"church" + 0.261*"historic" + -0.252*"village" + 0.178*"register" + -0.164*"gmina" + -0.152*"district" + 0.147*"building" + -0.138*"population" + 0.134*"township" + 0.131*"championships"

lsi.save('./data/lsa_model_1')
