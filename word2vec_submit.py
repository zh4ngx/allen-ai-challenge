"""
  Generate csv submission for Kaggle contest
"""

import logging
import sys

from gensim.models import Word2Vec, Phrases

from utils import extract_elements, choose_answer, preprocess_for_model

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
)

input_model = sys.argv[1]
input_validation = sys.argv[2]
output_file = sys.argv[3]
input_transformer = sys.argv[4]

# Load model
model = Word2Vec.load(input_model, mmap='r')
bigram_transformer = Phrases.load(input_transformer, mmap='r')

# Load validation set and advance 1 line
validation_set = open(input_validation)
validation_set.readline()

output = open(output_file, "w")
output.write("id,correctAnswer\n")

for line in validation_set:
    question_id, question, answers, _ = extract_elements(line)

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

    output.write("%s,%s\n" % (question_id, chosen_answer))

output.close()
