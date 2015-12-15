"""
  Evaluate efficacy of Wikipedia Word2Vec Model on Allen AI dataset

  Use latent vectors to calculate cosine distance between Questions/Answers
  Pick the answer with the smallest cosine distance
  Check against correct answer
  Measure accuracy by simple percentage correct
"""

import logging, gensim, sys, random, string

from utils import idx2answerlabel, preprocess_text_gensim, choose_answer

logging.basicConfig(
  format='%(asctime)s : %(levelname)s : %(message)s',
  level=logging.INFO
)

input_model = sys.argv[1]
input_training = sys.argv[2]

# Load model
model = gensim.models.Word2Vec.load(input_model, mmap='r')

# Load 'training' data
training_data = open(input_training)
training_data.readline() # advance past header line

correct = 0
total = 0

for line in training_data:
  elements = line.strip().split("\t")
  question_id = elements.pop(0)
  correct_answer = elements.pop(1)

  # Get canonical elements for comparison
  question = elements.pop(0)
  answers = elements

  answer_dict = {idx2answerlabel(idx): answer for idx, answer in enumerate(answers)}

  # Score log probability of question + answer text
  scores = {
    answer_label: model.score([preprocess_text_gensim(model, "%s %s" % (question, answer))])
    for answer_label, answer
    in answer_dict.iteritems()
  }

  chosen_answer = choose_answer(scores)

  got_right_answer = correct_answer == chosen_answer
  correct += got_right_answer
  total += 1

  if not got_right_answer and (random.random() > 0.95):
    print("Question: %s" % question)
    print("Correct Answer: %s" % answer_dict[correct_answer])
    print("Correct Answer similarity: %.4f" % scores[correct_answer])
    print("Chosen Answer: %s" % answer_dict[chosen_answer])
    print("Correct Answer similarity: %.4f" % scores[chosen_answer])

print("Correct: %d" % correct)
print("Total: %d" % total)
print("Accuracy: %.4f" % (float(correct) / total))
