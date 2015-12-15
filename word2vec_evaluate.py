"""
  Evaluate efficacy of Wikipedia Word2Vec Model on Allen AI dataset

  Use latent vectors to calculate cosine distance between Questions/Answers
  Pick the answer with the smallest cosine distance
  Check against correct answer
  Measure accuracy by simple percentage correct
"""

import logging
import random
import sys

from gensim.models import Word2Vec, Phrases

from utils import idx2answer_label, preprocess_for_model, choose_answer

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
)

input_model = sys.argv[1]
input_training = sys.argv[2]
input_transformer = sys.argv[3]

# Load model and bigram transformer
model = Word2Vec.load(input_model, mmap='r')
bigram_transformer = Phrases.load(input_transformer, mmap='r')

# Load 'training' data
training_data = open(input_training)
training_data.readline()  # advance past header line

correct = 0
total = 0

for line in bigram_transformer[training_data]:
    elements = line.strip().split("\t")
    question_id = elements.pop(0)
    correct_answer = elements.pop(1)

    # Extract question and answers
    question = elements.pop(0)
    answers = {idx2answer_label(idx): answer for idx, answer in enumerate(elements)}

    # Preprocess question and answers
    # Run text through model vocab filter
    question_preprocessed = preprocess_for_model(model, question)
    answers_preprocessed = {
        answer_label: preprocess_for_model(model, answer)
        for answer_label, answer
        in answers.iteritems()}

    # Calculate cosine similarity between mean projection weight vectors in question and answer
    similarities = {
        answer_label:
            model.n_similarity(
                    question_preprocessed,
                    answer_preprocessed
            ) if answer_preprocessed else 0.  # presume 0 similarity for missing
        for answer_label, answer_preprocessed
        in answers_preprocessed.iteritems()}

    chosen_answer = choose_answer(similarities)

    got_right_answer = correct_answer == chosen_answer
    correct += got_right_answer
    total += 1

    # Randomly display info for wrong answer
    if not got_right_answer and (random.random() > 0.95):
        print("Question: %s" % question)
        print("Correct Answer: %s" % answers[correct_answer])
        print("Correct Answer similarity: %.4f" % similarities[correct_answer])
        print("Chosen Answer: %s" % answers[chosen_answer])
        print("Correct Answer similarity: %.4f" % similarities[chosen_answer])

print("Correct: %d" % correct)
print("Total: %d" % total)
print("Accuracy: %.4f" % (float(correct) / total))
