"""
  Generate csv submission for Kaggle contest
"""

import gensim
import logging
import string
import sys

from utils import idx2answer_label

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

  question = elements.pop(0)
  answers = elements

  answer_dict = {idx2answer_label(idx): answer for idx, answer in enumerate(answers)}

  # Score log probability of question + answer text
  scores = {
    answer_label: model.score((question + answer).lower().translate(None, string.punctuation).split())
    for answer_label, answer
    in answer_dict.iteritems()
  }

  chosen_answer = max(scores.iteritems(), key=lambda item: item[1])[0]

  output.write("%s,%s\n" % (question_id, chosen_answer))

output.close()
