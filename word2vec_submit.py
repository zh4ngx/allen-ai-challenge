"""
  Generate csv submission for Kaggle contest
"""

import logging, gensim, bz2, sys, string

from utils import idx2answerchar

logging.basicConfig(
  format='%(asctime)s : %(levelname)s : %(message)s',
  level=logging.INFO
)

input_model = sys.argv[1]
input_validation = sys.argv[2]
output_file = sys.argv[3]

# Load model
model = gensim.models.Word2Vec.load(input_model, mmap='r')

# Load validation set and advance 1 line
validation_set = open(input_validation)
validation_set.readline()

output = open(output_file, "w")
output.write("id,correctAnswer\n")

for line in validation_set:
  elements = line.split("\t")
  question_id = elements.pop(0)

  # clean question and filter by vocab
  question_words = [
    word
    for word
    in elements.pop(0).lower().translate(None, string.punctuation).split()
    if model.__contains__(word)
  ]
  answers = elements # the only items left

  answer_words_dict = {
    idx2answerchar(idx): [
      word
      for word
      in answer.lower().translate(None, string.punctuation).split()
      if model.__contains__(word)
    ] # clean answer and filter by vocab
    for idx, answer
    in enumerate(answers)
  }

  similarities = {
    answer_char:
      model.n_similarity(
        question_words,
        answer_words
      ) if answer_words else 0. # presume 0 similarity for missing
    for answer_char, answer_words
    in answer_words_dict.iteritems()
  }
  chosen_answer = max(similarities.iteritems(), key=lambda item: item[1])[0]

  output.write("%s,%s\n" % (question_id, chosen_answer))

output.close()
