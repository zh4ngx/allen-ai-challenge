"""
  Evaluate efficacy of Wikipedia Word2Vec Model on Allen AI dataset

  Use latent vectors to calculate cosine distance between Questions/Answers
  Pick the answer with the smallest cosine distance
  Check against correct answer
  Measure accuracy by simple percentage correct
"""

import logging, gensim, sys, random, string

from utils import idx2answerchar

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
  question = elements.pop(0) # keep original question text
  # clean question and filter by vocab
  question_words = [
    word
    for word
    in question.lower().translate(None, string.punctuation).split()
    if model.__contains__(word)
  ]
  answers = elements # the only items left

  # Calculate cosine similarity between mean projection weight vectors in question and answer
  answer_dict = {idx2answerchar(idx): answer for idx, answer in enumerate(answers)} # keep original answer txt

  # Intermediate dictionary for list of words in answer
  answer_words_dict = {
    answer_char: [
      word
      for word
      in answer.lower().translate(None, string.punctuation).split()
      if model.__contains__(word)
    ] # clean answer and filter by vocab
    for answer_char, answer
    in answer_dict.iteritems()
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

  got_right_answer = correct_answer == chosen_answer
  correct += got_right_answer
  total += 1

  if not got_right_answer and (random.random() > 0.95):
    print("Question: %s" % question)
    print("Correct Answer: %s" % answer_dict[correct_answer])
    print("Correct Answer similarity: %.4f" % similarities[correct_answer])
    print("Chosen Answer: %s" % answer_dict[chosen_answer])
    print("Correct Answer similarity: %.4f" % similarities[chosen_answer])

print("Correct: %d" % correct)
print("Total: %d" % total)
print("Accuracy: %.4f" % (float(correct) / total))
