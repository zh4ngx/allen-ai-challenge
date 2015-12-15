Tackling Kaggle's [Allen AI Challenge](https://www.kaggle.com/c/the-allen-ai-science-challenge)

Progress on my [blog](http://zhangbanger.github.io/)

# Run Examples

In both cases, examples are given assuming you're doing everything in the current directory. The models and submissions are automatically timestamped in the `--model` and `--project` directories, respectively.

## LSA

```bash
$ python -m gensim.scripts.make_wiki

$ python lsa_train.py \
    --dictionary wiki_en_wordsids.txt \
    --corpus wiki_en_tfidf.mm \
    --model model/lsa

$ python lsa_evaluate.py \
    --model model/lsa/timestamp.model \
    --data training.tsv

$ python lsa_submit.py \
    --model model/lsa/timestamp.model \
    --project $(pwd)
```

## Word2Vec

```bash
$ wget https://word2vec.googlecode.com/svn/trunk/questions-words.txt

$ python word2vec_train.py \
    --articles enwiki-latest-pages-articles.xml.bz2 \
    --model model/word2vec \
    --demo question-words.txt \
    --lines wiki-lines.txt

$ python word2vec_evaluate.py \
    --model model/word2vec/timestamp.model \
    --data training.tsv

$ python word2vec_submit.py \
    --model model/word2vec/timestamp.model \
    --project $(pwd)
```
