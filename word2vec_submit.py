"""
  Generate csv submission for Kaggle contest
"""
import argparse
import datetime
import logging

from gensim.models import Word2Vec, Phrases

from utils import extract_elements, choose_answer, preprocess_for_model

logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO
)
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="path to word2vec/model/timestamp.model")
parser.add_argument("-p", "--project", help="path to allen-ai & validation.tsv")
parser.add_argument("-t", "--transformer", help="path to word2vec/model/bigram_transformer")
args = parser.parse_args()

# Load model
model = Word2Vec.load(args.model, mmap='r')
bigram_transformer = Phrases.load(args.transformer, mmap='r') if args.transformer else None

# Load validation set and advance 1 line
validation_set = open("%s/validation.tsv" % args.project)
validation_set.readline()

output = open("%s/%s_submission.tsv" % (args.project, timestamp), "w")
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
