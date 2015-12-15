"""
  Evaluate efficacy of Wikipedia Word2Vec Model on Allen AI dataset

  Use latent vectors to calculate cosine distance between Questions/Answers
  Pick the answer with the smallest cosine distance
  Check against correct answer
  Measure accuracy by simple percentage correct
"""
import argparse
import logging
import random

from gensim.models import Word2Vec, Phrases

from utils import extract_elements, preprocess_for_model, choose_answer

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="path to word2vec/model/timestamp.model")
parser.add_argument("-d", "--data", help="path to training.tsv")
parser.add_argument("-t", "--transformer", help="path to word2vec/model/bigram_transformer")
args = parser.parse_args()

# Load model and bigram transformer
model = Word2Vec.load(args.model, mmap='r')
bigram_transformer = Phrases.load(args.transformer, mmap='r') if args.transformer else None

# Load 'training' data
training_data = open(args.training)
training_data.readline()  # advance past header line

correct = 0
total = 0

for line in training_data:
    question_id, question, answers, correct_answer = extract_elements(line)

    # Preprocess question and answers
    # Run text through model vocab filter
    question_preprocessed = preprocess_for_model(model, question, bigram_transformer)
    answers_preprocessed = {
        answer_label: preprocess_for_model(model, answer, bigram_transformer)
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
